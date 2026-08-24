#!/usr/bin/env bash
#
# BPE trainer verification harness (Task 6, step 6).
#
# Runs the four gates that decide whether std/bpe_train.mlr + the CLI actually
# reproduce HuggingFace `tokenizers`, and how fast:
#
#   G1  byte-exact       MLRift's tokenizer.json vs the HF oracle
#                        (model.merges + model.vocab, and whole-document bytes)
#   G2  1-thread speed   MLRift best-of-3 vs the recorded HF 1-thread bar
#   G3  full speed       MLRift best-of-3 at $(nproc) vs the HF full bar
#   G4  round-trip       decode(encode(x)) == x through std/tokenizer.mlr
#   M   peak RSS         maximum resident set of the full-throttle run
#
# Usage:  bash scripts/bpe_verify.sh [CORPUS]
#         CORPUS defaults to the WZMA embedder corpus (276 MB).
#
# Environment overrides:
#   HF_BAR_1T / HF_BAR_FULL   the HF wall-clock seconds to compare against.
#                             The defaults below were MEASURED ON THIS MACHINE
#                             (24 cores) with tokenizers 0.23.1 on 2026-08-24,
#                             three independent best-of-3 sweeps:
#                               1-thread  RAYON_NUM_THREADS=1
#                                 50.514/50.364/50.069   best 50.069
#                                 50.668/49.926/50.462   best 49.926
#                                 50.346/50.079/50.166   best 50.079
#                               full      (all cores)
#                                  7.088/ 7.128/ 7.124   best  7.088
#                                  7.086/ 7.162/ 6.961   best  6.961
#                                  7.077/ 7.055/ 7.020   best  7.020
#                             An independent reviewer measured 50.120 (1-thread)
#                             and 7.084 (full) on the same box, consistent with
#                             the above. The enforced defaults (50.07 / 7.09)
#                             sit at the median of those bests; the single
#                             fastest samples ever seen were 49.926 and 6.961,
#                             so the bars are at most ~0.15s looser than the
#                             most demanding observation and never flatter
#                             MLRift by more than run-to-run noise.
#                             An older pair (54.59 / 8.64) is NOT used: a fresh
#                             measurement could not reproduce it and the gap is
#                             machine state, not clock boundary (HF's
#                             whole-process overhead beyond train() is 0.021s).
#                             They are machine-specific: re-measure with
#                             HF_TIME=1 before trusting them elsewhere.
#   HF_TIME=1                 re-measure both HF bars on this machine and
#                             DECIDE G2/G3 against those numbers instead of the
#                             defaults (adds ~3 minutes). A measurement taken
#                             back-to-back with the MLRift runs beats a stored
#                             constant, so it replaces the bar outright.
#   REPS                      timing repetitions per gate (default 3).
#
# Exits non-zero if any gate FAILs.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

CORPUS="${1:-/home/pantelis/Desktop/Projects/Work/AtlasLM/checkpoints/wzma_embedder/corpus.txt}"
VOCAB_SIZE=8192
NTHREADS_FULL="$(nproc)"
REPS="${REPS:-3}"
# Bars, in whole-process wall-clock seconds. Provenance — every sweep behind
# these two numbers, and why the older 54.59/8.64 pair was dropped — is in the
# HF_BAR_1T / HF_BAR_FULL entry of the usage block at the top of this file.
# Keep the two in sync: an evidence harness that documents one bar and enforces
# another is the same self-contradiction the summary logic was fixed for.
# With HF_TIME=1 the bars are re-derived from this run and those measurements,
# not these constants, decide G2/G3.
HF_BAR_1T="${HF_BAR_1T:-50.07}"
HF_BAR_FULL="${HF_BAR_FULL:-7.09}"

PY=/home/pantelis/Desktop/Projects/Work/AtlasLM/.venv/bin/python3
MLRC=./build/mlrc

WORK=/tmp/bpe_verify
REF_DIR=/tmp/hf_ref_full
REF_JSON="$REF_DIR/tokenizer_ref.json"
REF_MIN="$REF_DIR/tokenizer_ref.min.json"
OUT_JSON="$WORK/tokenizer_mlrift.json"
CLI_BIN="$WORK/bpe_train_cli"
RT_BIN="$WORK/bpe_roundtrip"
DIGESTS="$WORK/digests.txt"

mkdir -p "$WORK" "$REF_DIR" || exit 1
: > "$DIGESTS"

# --- result bookkeeping -----------------------------------------------------
declare -A VERDICT
declare -A DETAIL
FAILED=0

# A gate's verdict lives ONLY in VERDICT[]; the final exit status is derived
# from that map at the end. Accumulating a monotonic FAILED flag here would
# mean a gate re-decided later (see `regate`) could print PASS in the summary
# while the script still exited non-zero — the same class of self-contradiction
# as printing a warning and a PASS side by side. FAILED is reserved for
# non-gate problems (a crashed profiling run, a digest mismatch).
record() {            # record <gate> <PASS|FAIL> <detail...>
    VERDICT["$1"]="$2"
    DETAIL["$1"]="${*:3}"
    printf '  -> %s %s: %s\n' "$2" "$1" "${*:3}"
}

hdr() { printf '\n=== %s ===\n' "$*"; }

# Wall-clock seconds (3 decimals) of a command run with the given env.
# Prints the elapsed time on stdout; the command's own output goes to $2.
timed_run() {         # timed_run <threads> <logfile>
    local threads="$1" log="$2" t0 t1
    t0=$(date +%s%N)
    MLRIFT_BPE_CORPUS="$CORPUS" \
    MLRIFT_BPE_OUT="$OUT_JSON" \
    MLRIFT_BPE_VOCAB="$VOCAB_SIZE" \
    MLRIFT_BPE_THREADS="$threads" \
    "$CLI_BIN" > "$log" 2>&1
    local rc=$?
    t1=$(date +%s%N)
    if [ $rc -ne 0 ]; then
        echo "     the trainer exited $rc:" >&2
        sed 's/^/     | /' "$log" >&2
        echo "FAILED"
        return 1
    fi
    # Every run must produce the SAME tokenizer.json regardless of thread
    # count; the digest log is checked once at the end.
    printf '%s %s\n' "$threads" "$(md5sum < "$OUT_JSON" | cut -d' ' -f1)" >> "$DIGESTS"
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e9}'
}

# Min of the whitespace-separated numbers on stdin. The `grep -v '^$'` is
# load-bearing: the accumulator strings are built as " a b c" with a leading
# space, so without it `sort -g` returns the EMPTY first field and every
# downstream `awk 'a<b'` comparison sees a=0 and passes vacuously.
best_of() { tr ' ' '\n' | grep -v '^$' | sort -g | head -1; }

# Dies rather than let an empty/garbage "best" turn into a free PASS. This is
# the last guard between the harness and a vacuous PASS, so it rejects by
# EXCLUSION (any byte that is not a digit or '.') rather than by a prefix
# glob: `[0-9]*` would happily accept "16abc" and "1e3", both of which awk
# then truncates to a smaller number and compares as a pass.
require_number() {    # require_number <value> <what>
    case "$1" in
        ''|*[!0-9.]*|*.*.*|.) ;;
        *) return 0 ;;
    esac
    echo "FATAL: $2 is not a number (got '$1')" >&2
    exit 1
}

# --- 0. environment ---------------------------------------------------------
hdr "environment"
echo "corpus:        $CORPUS"
if [ ! -r "$CORPUS" ]; then
    echo "FATAL: corpus is missing or unreadable: $CORPUS" >&2
    exit 1
fi
echo "corpus bytes:  $(stat -c %s "$CORPUS")"
echo "cores:         $NTHREADS_FULL"
echo "vocab_size:    $VOCAB_SIZE"
echo "reps:          $REPS"
echo "HF bars:       1-thread ${HF_BAR_1T}s   full ${HF_BAR_FULL}s"
echo "memory before the run:"
free -h | sed 's/^/  /'

# --- 1. build ---------------------------------------------------------------
hdr "build"
if [ ! -x "$MLRC" ]; then
    echo "FATAL: $MLRC not found — build the compiler first" >&2
    exit 1
fi
# $WORK persists between runs and mlrc leaves the PREVIOUS output in place when
# it rejects the source, so an existence check after a failed compile would
# happily hand the gates a stale binary that no longer corresponds to the tree.
# Delete first, then gate on mlrc's own exit status (which `| tail -1` would
# discard, pipefail or not, because the pipeline's status is never tested).
rm -f "$CLI_BIN" "$RT_BIN"
"$MLRC" --arch=x86_64 examples/bpe_train_cli.mlr -o "$CLI_BIN" > "$WORK/build_cli.log" 2>&1 \
    || { echo "FATAL: CLI build failed" >&2; tail -5 "$WORK/build_cli.log" >&2; exit 1; }
tail -1 "$WORK/build_cli.log"
[ -x "$CLI_BIN" ] || { echo "FATAL: mlrc reported success but produced no CLI binary" >&2; exit 1; }
"$MLRC" --arch=x86_64 tests/bpe/roundtrip_probe.mlr -o "$RT_BIN" > "$WORK/build_rt.log" 2>&1 \
    || { echo "FATAL: round-trip probe build failed" >&2; tail -5 "$WORK/build_rt.log" >&2; exit 1; }
tail -1 "$WORK/build_rt.log"
[ -x "$RT_BIN" ] || { echo "FATAL: mlrc reported success but produced no probe binary" >&2; exit 1; }

# --- 2. the HF oracle -------------------------------------------------------
hdr "HF oracle"
if [ -s "$REF_JSON" ] && [ -s "$REF_MIN" ]; then
    echo "reusing $REF_JSON"
else
    echo "generating the HF reference (this takes ~1 minute)..."
    "$PY" - "$CORPUS" "$REF_JSON" <<'PYGEN' || { echo "FATAL: HF reference generation failed" >&2; exit 1; }
import sys, tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
corpus, out = sys.argv[1], sys.argv[2]
print("tokenizers", tokenizers.__version__)
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.post_processor = processors.ByteLevel(trim_offsets=False)
tr = trainers.BpeTrainer(vocab_size=8192,
                         special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tk.train([corpus], tr)
tk.save(out, pretty=True)
PYGEN
fi
"$PY" - "$REF_JSON" "$REF_MIN" <<'PYMIN' || { echo "FATAL: could not minify the HF reference" >&2; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
PYMIN
"$PY" - "$REF_JSON" <<'PYSUM' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
m, v = d["model"]["merges"], d["model"]["vocab"]
print("oracle: vocab %d  merges %d  added_tokens %d  alphabet %d" %
      (len(v), len(m), len(d["added_tokens"]), sum(1 for t in v if len(t) == 1)))
print("oracle: first merges", [ "".join(x) for x in m[:5] ])
PYSUM

# --- 3. G1: byte-exact ------------------------------------------------------
hdr "G1 — byte-exact vs the HF oracle"
G1LOG="$WORK/g1.log"
if ! timed_run "$NTHREADS_FULL" "$G1LOG" > /dev/null; then
    record G1 FAIL "the trainer did not complete"
else
    sed 's/^/  | /' "$G1LOG"
    "$PY" - "$OUT_JSON" "$REF_JSON" <<'PYDIFF'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
am, bm = a["model"]["merges"], b["model"]["merges"]
av, bv = a["model"]["vocab"], b["model"]["vocab"]
ok = True
print("  merges: mlrift %d  hf %d" % (len(am), len(bm)))
if len(am) != len(bm):
    ok = False
    print("  MERGE COUNT MISMATCH")
for i, (x, y) in enumerate(zip(am, bm)):
    if list(x) != list(y):
        ok = False
        print("  FIRST MERGE DIVERGENCE at index %d" % i)
        lo, hi = max(0, i - 3), min(len(am), i + 4)
        for j in range(lo, hi):
            mark = "->" if j == i else "  "
            print("   %s [%5d] mlrift=%-24r hf=%r"
                  % (mark, j, list(am[j]), list(bm[j])))
        break
print("  vocab: mlrift %d  hf %d" % (len(av), len(bv)))
if len(av) != len(bv):
    ok = False
    print("  VOCAB SIZE MISMATCH")
diffs = [(t, av.get(t), bv.get(t)) for t in sorted(set(av) | set(bv)) if av.get(t) != bv.get(t)]
if diffs:
    ok = False
    print("  %d VOCAB ID MISMATCHES, first 10:" % len(diffs))
    for t, x, y in diffs[:10]:
        print("    %r mlrift=%s hf=%s" % (t, x, y))
sys.exit(0 if ok else 1)
PYDIFF
    payload_rc=$?
    if cmp -s "$OUT_JSON" "$REF_MIN"; then
        doc="whole-document bytes IDENTICAL to the minified oracle"
        doc_ok=1
    else
        doc="whole-document bytes DIFFER from the minified oracle ($(cmp "$OUT_JSON" "$REF_MIN" 2>&1 | head -1))"
        doc_ok=0
    fi
    echo "  $doc"
    if [ $payload_rc -eq 0 ] && [ $doc_ok -eq 1 ]; then
        record G1 PASS "model.merges + model.vocab identical; $doc"
    elif [ $payload_rc -eq 0 ]; then
        record G1 FAIL "payload identical but $doc"
    else
        record G1 FAIL "model.merges/model.vocab DIVERGE (see the first divergence above)"
    fi
fi

# --- 4. G2: single-thread speed --------------------------------------------
hdr "G2 — single-thread speed (best of $REPS, bar = HF 1-thread ${HF_BAR_1T}s)"
G2_TIMES=""
for i in $(seq 1 "$REPS"); do
    t=$(timed_run 1 "$WORK/g2_$i.log") || { record G2 FAIL "run $i did not complete"; break; }
    echo "  run $i: ${t}s"
    G2_TIMES="$G2_TIMES $t"
done
if [ -z "${VERDICT[G2]:-}" ]; then
    G2_BEST=$(echo "$G2_TIMES" | best_of)
    require_number "$G2_BEST" "the G2 best-of-$REPS time"
    if awk -v a="$G2_BEST" -v b="$HF_BAR_1T" 'BEGIN{exit !(a<b)}'; then
        record G2 PASS "best ${G2_BEST}s <  HF ${HF_BAR_1T}s (runs:$G2_TIMES)"
    else
        record G2 FAIL "best ${G2_BEST}s >= HF ${HF_BAR_1T}s (runs:$G2_TIMES)"
    fi
fi

# --- 5. G3 + M: full-throttle speed and peak RSS ----------------------------
hdr "G3 — ${NTHREADS_FULL}-thread speed (best of $REPS, bar = HF full ${HF_BAR_FULL}s)"
G3_TIMES=""
for i in $(seq 1 "$REPS"); do
    t=$(timed_run "$NTHREADS_FULL" "$WORK/g3_$i.log") || { record G3 FAIL "run $i did not complete"; break; }
    echo "  run $i: ${t}s"
    G3_TIMES="$G3_TIMES $t"
done
if [ -z "${VERDICT[G3]:-}" ]; then
    G3_BEST=$(echo "$G3_TIMES" | best_of)
    require_number "$G3_BEST" "the G3 best-of-$REPS time"
    if awk -v a="$G3_BEST" -v b="$HF_BAR_FULL" 'BEGIN{exit !(a<b)}'; then
        record G3 PASS "best ${G3_BEST}s <  HF ${HF_BAR_FULL}s (runs:$G3_TIMES)"
    else
        record G3 FAIL "best ${G3_BEST}s >= HF ${HF_BAR_FULL}s (runs:$G3_TIMES)"
    fi
fi

hdr "M — peak RSS of the ${NTHREADS_FULL}-thread run"
# A separate, untimed run: /usr/bin/time -v adds noise the speed gates must not
# absorb. Measurement (not assumption) puts the high-water mark inside
# bpe_train_merges, not in the counter: counting tops out near 366 MiB with
# every partial WordTable live, and the 1-thread peak is marginally HIGHER
# than the 24-thread peak, so the mark is thread-count independent and one
# full-throttle run captures it for the whole pipeline.
MEMLOG="$WORK/mem.log"
MLRIFT_BPE_CORPUS="$CORPUS" MLRIFT_BPE_OUT="$OUT_JSON" \
MLRIFT_BPE_VOCAB="$VOCAB_SIZE" MLRIFT_BPE_THREADS="$NTHREADS_FULL" \
    /usr/bin/time -v "$CLI_BIN" > "$MEMLOG" 2>&1
MEM_RC=$?
PEAK_KB=$(awk -F': *' '/Maximum resident set size/{print $2}' "$MEMLOG")
if [ $MEM_RC -ne 0 ]; then
    # $OUT_JSON still holds the PREVIOUS run's output, so recording a digest
    # for it here would credit a crashed run with a passing determinism check.
    echo "  the memory-profiling run exited $MEM_RC — no digest recorded"
    # The trainer's own output comes FIRST in $MEMLOG; /usr/bin/time -v appends
    # its 20-line report afterwards, so `tail` here would show only that
    # boilerplate and hide the actual error.
    head -6 "$MEMLOG" | sed 's/^/    | /'
    PEAK_HUMAN="unavailable (run exited $MEM_RC)"
    FAILED=1
elif [ -z "$PEAK_KB" ]; then
    printf '%s %s\n' "$NTHREADS_FULL" "$(md5sum < "$OUT_JSON" | cut -d' ' -f1)" >> "$DIGESTS"
    echo "  could not read the peak RSS from /usr/bin/time -v"
    PEAK_HUMAN="unknown"
    FAILED=1
else
    printf '%s %s\n' "$NTHREADS_FULL" "$(md5sum < "$OUT_JSON" | cut -d' ' -f1)" >> "$DIGESTS"
    PEAK_HUMAN=$(awk -v k="$PEAK_KB" 'BEGIN{printf "%d KB (%.2f GiB)", k, k/1048576}')
    echo "  peak RSS: $PEAK_HUMAN"
fi

# --- 5b. determinism across thread counts -----------------------------------
hdr "D — every run produced the same tokenizer.json"
REF_MD5=$(md5sum < "$REF_MIN" | cut -d' ' -f1)
sort "$DIGESTS" | uniq -c | awk '{printf "  %d run(s) at threads=%s -> %s\n", $1, $2, $3}'
NRUNS=$(wc -l < "$DIGESTS")
NDIGESTS=$(cut -d' ' -f2 "$DIGESTS" | sort -u | wc -l)
if [ "$NRUNS" = "0" ]; then
    # Zero successful runs is a different failure from disagreeing runs, and
    # saying "the trainer is not thread-count independent" here would blame
    # the wrong thing entirely.
    echo "  no successful runs to compare — every trainer invocation failed"
    FAILED=1
elif [ "$NDIGESTS" != "1" ]; then
    echo "  RUNS DISAGREE — the trainer is not thread-count independent"
    FAILED=1
elif [ "$(cut -d' ' -f2 "$DIGESTS" | head -1)" != "$REF_MD5" ]; then
    echo "  all runs agree with each other but NOT with the oracle ($REF_MD5)"
    FAILED=1
else
    echo "  all $NRUNS runs -> $REF_MD5 == the minified oracle"
fi

# --- 6. G4: round-trip ------------------------------------------------------
hdr "G4 — round-trip through std/tokenizer.mlr"
if MLRIFT_BPE_OUT="$OUT_JSON" "$RT_BIN"; then
    record G4 PASS "decode(encode(x)) == x for every probe (incl. non-ASCII)"
else
    record G4 FAIL "a probe did not round-trip (see the output above)"
fi

# --- 7. optional: re-measure the HF bars ------------------------------------
if [ "${HF_TIME:-0}" = "1" ]; then
    hdr "HF re-measurement (best of $REPS each)"
    # Timed exactly the way MLRift is: whole-process wall clock of a run that
    # trains AND writes the tokenizer.json. Timing only `train()` would flatter
    # HF by excluding interpreter startup, import and serialization, which the
    # MLRift number necessarily includes.
    cat > "$WORK/hf_train.py" <<'PYHF'
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
corpus, out = sys.argv[1], sys.argv[2]
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.post_processor = processors.ByteLevel(trim_offsets=False)
tr = trainers.BpeTrainer(vocab_size=8192,
                         special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tk.train([corpus], tr)
tk.save(out)
PYHF
    hf_timed() {      # hf_timed <rayon-threads-or-empty>
        local t0 t1
        t0=$(date +%s%N)
        if [ -n "$1" ]; then
            RAYON_NUM_THREADS="$1" "$PY" "$WORK/hf_train.py" "$CORPUS" "$WORK/hf_timed.json" >/dev/null 2>&1
        else
            "$PY" "$WORK/hf_train.py" "$CORPUS" "$WORK/hf_timed.json" >/dev/null 2>&1
        fi
        local rc=$?
        t1=$(date +%s%N)
        # Same contract as timed_run: this body runs inside $(...), so `exit`
        # would kill only the subshell and leave an EMPTY entry that best_of
        # silently skips — reporting the best of the surviving runs as if
        # nothing had gone wrong. Signal through the return status instead.
        if [ $rc -ne 0 ]; then
            echo "     the HF timing run exited $rc" >&2
            echo "FAILED"
            return 1
        fi
        awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e9}'
    }
    hf1=""; hfn=""
    for i in $(seq 1 "$REPS"); do
        t=$(hf_timed 1)  || { echo "FATAL: an HF 1-thread timing run failed" >&2; exit 1; }
        hf1="$hf1 $t"
        t=$(hf_timed "") || { echo "FATAL: an HF full-throttle timing run failed" >&2; exit 1; }
        hfn="$hfn $t"
    done
    HF1_BEST=$(echo "$hf1" | best_of)
    HFN_BEST=$(echo "$hfn" | best_of)
    require_number "$HF1_BEST" "the HF 1-thread best time"
    require_number "$HFN_BEST" "the HF full-throttle best time"
    echo "  HF 1-thread runs:$hf1  best ${HF1_BEST}s   (default bar ${HF_BAR_1T}s)"
    echo "  HF full runs:$hfn  best ${HFN_BEST}s   (default bar ${HF_BAR_FULL}s)"
    echo "  MLRift best:      1-thread ${G2_BEST:-n/a}s   full ${G3_BEST:-n/a}s"

    # A measurement taken THIS run, on THIS machine, back-to-back with the
    # MLRift runs, is strictly better evidence than a stored constant — so it
    # REPLACES the bar rather than being printed as a footnote beside a verdict
    # that contradicts it. Printing "MLRift does NOT beat HF" and then "G3 PASS"
    # in the same output is worse than either verdict alone.
    echo "  the G2/G3 verdicts are re-decided below against these measurements."
    regate() {        # regate <gate> <mlrift-best> <hf-best>
        local g="$1" mine="$2" theirs="$3"
        # A gate whose trainer run crashed has already been recorded FAIL and
        # never assigned a best time. Feeding that empty string to
        # require_number would abort the script HERE — before the SUMMARY is
        # printed — and tell the operator the harness has a number-parsing
        # problem rather than that the trainer died. Keep the existing verdict
        # and carry on to the summary instead; there is nothing to re-decide.
        if [ -z "$mine" ]; then
            echo "  $g has no completed run to re-decide — keeping ${VERDICT[$g]:-FAIL}"
            VERDICT["$g"]="${VERDICT[$g]:-FAIL}"
            return
        fi
        require_number "$mine" "the $g best time"
        if awk -v a="$mine" -v b="$theirs" 'BEGIN{exit !(a<b)}'; then
            record "$g" PASS "best ${mine}s <  HF re-measured ${theirs}s (this run)"
        else
            record "$g" FAIL "best ${mine}s >= HF re-measured ${theirs}s (this run)"
        fi
    }
    regate G2 "${G2_BEST:-}" "$HF1_BEST"
    regate G3 "${G3_BEST:-}" "$HFN_BEST"
fi

# --- summary ----------------------------------------------------------------
hdr "SUMMARY"
printf '%-4s %-6s %s\n' "GATE" "RESULT" "DETAIL"
printf '%-4s %-6s %s\n' "----" "------" "------"
for g in G1 G2 G3 G4; do
    v="${VERDICT[$g]:-FAIL}"
    # A gate that never ran is a failure, not a silent omission.
    [ "$v" = "PASS" ] || FAILED=1
    printf '%-4s %-6s %s\n' "$g" "$v" "${DETAIL[$g]:-not run}"
done
printf '%-4s %-6s %s\n' "M" "INFO" "peak RSS $PEAK_HUMAN (${NTHREADS_FULL} threads)"
echo
if [ $FAILED -ne 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: ALL GATES PASS"
