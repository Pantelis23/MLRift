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
#                             (24 cores) with tokenizers 0.23.1, best of 3:
#                               1-thread  RAYON_NUM_THREADS=1 -> 55.14/54.70/54.59
#                               full      (all cores)         ->  8.64 best
#                             They are machine-specific: re-measure with
#                             HF_TIME=1 before trusting them elsewhere.
#   HF_TIME=1                 re-measure both HF bars on this machine instead
#                             of using the recorded ones (adds ~3 minutes).
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
HF_BAR_1T="${HF_BAR_1T:-54.59}"
HF_BAR_FULL="${HF_BAR_FULL:-8.64}"

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

record() {            # record <gate> <PASS|FAIL> <detail...>
    VERDICT["$1"]="$2"
    DETAIL["$1"]="${*:3}"
    [ "$2" = "FAIL" ] && FAILED=1
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

# Dies rather than let an empty/garbage "best" turn into a free PASS.
require_number() {    # require_number <value> <what>
    case "$1" in
        [0-9]*.[0-9]*|[0-9]*) return 0 ;;
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
"$MLRC" --arch=x86_64 examples/bpe_train_cli.mlr -o "$CLI_BIN" 2>&1 | tail -1
[ -x "$CLI_BIN" ] || { echo "FATAL: CLI build failed" >&2; exit 1; }
"$MLRC" --arch=x86_64 tests/bpe/roundtrip_probe.mlr -o "$RT_BIN" 2>&1 | tail -1
[ -x "$RT_BIN" ] || { echo "FATAL: round-trip probe build failed" >&2; exit 1; }

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
# absorb. All $NTHREADS_FULL partial WordTables coexist until the merge frees
# them, so this is the high-water mark for the whole pipeline.
MEMLOG="$WORK/mem.log"
MLRIFT_BPE_CORPUS="$CORPUS" MLRIFT_BPE_OUT="$OUT_JSON" \
MLRIFT_BPE_VOCAB="$VOCAB_SIZE" MLRIFT_BPE_THREADS="$NTHREADS_FULL" \
    /usr/bin/time -v "$CLI_BIN" > "$MEMLOG" 2>&1
printf '%s %s\n' "$NTHREADS_FULL" "$(md5sum < "$OUT_JSON" | cut -d' ' -f1)" >> "$DIGESTS"
PEAK_KB=$(awk -F': *' '/Maximum resident set size/{print $2}' "$MEMLOG")
if [ -z "$PEAK_KB" ]; then
    echo "  could not read the peak RSS from /usr/bin/time -v"
    PEAK_HUMAN="unknown"
    FAILED=1
else
    PEAK_HUMAN=$(awk -v k="$PEAK_KB" 'BEGIN{printf "%d KB (%.2f GiB)", k, k/1048576}')
    echo "  peak RSS: $PEAK_HUMAN"
fi

# --- 5b. determinism across thread counts -----------------------------------
hdr "D — every run produced the same tokenizer.json"
REF_MD5=$(md5sum < "$REF_MIN" | cut -d' ' -f1)
sort "$DIGESTS" | uniq -c | awk '{printf "  %d run(s) at threads=%s -> %s\n", $1, $2, $3}'
NDIGESTS=$(cut -d' ' -f2 "$DIGESTS" | sort -u | wc -l)
if [ "$NDIGESTS" != "1" ]; then
    echo "  RUNS DISAGREE — the trainer is not thread-count independent"
    FAILED=1
elif [ "$(cut -d' ' -f2 "$DIGESTS" | head -1)" != "$REF_MD5" ]; then
    echo "  all runs agree with each other but NOT with the oracle ($REF_MD5)"
    FAILED=1
else
    echo "  all $(wc -l < "$DIGESTS") runs -> $REF_MD5 == the minified oracle"
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
        [ $rc -ne 0 ] && { echo "FATAL: the HF timing run exited $rc" >&2; exit 1; }
        awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)/1e9}'
    }
    hf1=""; hfn=""
    for i in $(seq 1 "$REPS"); do
        hf1="$hf1 $(hf_timed 1)"
        hfn="$hfn $(hf_timed "")"
    done
    HF1_BEST=$(echo "$hf1" | best_of)
    HFN_BEST=$(echo "$hfn" | best_of)
    require_number "$HF1_BEST" "the HF 1-thread best time"
    require_number "$HFN_BEST" "the HF full-throttle best time"
    echo "  HF 1-thread runs:$hf1  best ${HF1_BEST}s   (gate bar was ${HF_BAR_1T}s)"
    echo "  HF full runs:$hfn  best ${HFN_BEST}s   (gate bar was ${HF_BAR_FULL}s)"
    echo "  MLRift best:      1-thread ${G2_BEST:-n/a}s   full ${G3_BEST:-n/a}s"
    echo "  NOTE: the G2/G3 verdicts above used the RECORDED bars, not these."
    if awk -v a="${G3_BEST:-0}" -v b="$HFN_BEST" 'BEGIN{exit !(a>=b)}'; then
        echo "  WARNING: against THIS re-measurement MLRift does NOT beat HF at full throttle."
    fi
    if awk -v a="${G2_BEST:-0}" -v b="$HF1_BEST" 'BEGIN{exit !(a>=b)}'; then
        echo "  WARNING: against THIS re-measurement MLRift does NOT beat HF single-threaded."
    fi
fi

# --- summary ----------------------------------------------------------------
hdr "SUMMARY"
printf '%-4s %-6s %s\n' "GATE" "RESULT" "DETAIL"
printf '%-4s %-6s %s\n' "----" "------" "------"
for g in G1 G2 G3 G4; do
    printf '%-4s %-6s %s\n' "$g" "${VERDICT[$g]:-FAIL}" "${DETAIL[$g]:-not run}"
done
printf '%-4s %-6s %s\n' "M" "INFO" "peak RSS $PEAK_HUMAN (${NTHREADS_FULL} threads)"
echo
if [ $FAILED -ne 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: ALL GATES PASS"
