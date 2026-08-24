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
#                             SETTING EITHER FROM THE ENVIRONMENT DOES NOT MAKE
#                             IT A BAR: the harness cannot attribute a number it
#                             did not measure, so a supplied value is displayed
#                             for reference and the run falls back to measuring
#                             HF itself. Otherwise `HF_BAR_FULL=999` would print
#                             "ALL GATES PASS" and exit 0.
#   HF_TIME=1                 re-measure both HF bars on this machine and
#                             DECIDE G2/G3 against those numbers instead of the
#                             defaults (adds ~3 minutes). A measurement taken
#                             back-to-back with the MLRift runs beats a stored
#                             constant, so it replaces the bar outright. Forced
#                             automatically whenever the stored bars do not
#                             apply to this run's input.
#   VOCAB_SIZE                target vocabulary (default 8192). Flows to the
#                             CLI, the oracle, the oracle's provenance stamp and
#                             the HF timing runs together, and any value other
#                             than the bars' invalidates them.
#   REPS                      timing repetitions per gate (default 3).
#
# Exits non-zero if any gate FAILs.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# The stored HF bars, and the cached HF oracle, are measurements of ONE
# (corpus, vocab_size) pair. Both are named here so every later use can be
# checked against what is actually being tested rather than assumed to match.
DEFAULT_CORPUS=/home/pantelis/Desktop/Projects/Work/AtlasLM/checkpoints/wzma_embedder/corpus.txt
BAR_BYTES=276539257       # the corpus CONTENT SIZE they were measured on. The
                          # path alone is not enough: corpus.txt regenerated,
                          # truncated or appended in place is different work at
                          # the same realpath, and the bars would silently
                          # decide gates about it (A-1).
BAR_VOCAB=8192            # the vocab size HF_BAR_1T / HF_BAR_FULL were measured at
BAR_CORES=24              # the core count they were measured on — HF_BAR_FULL is
                          # a function of it, so a box with a different nproc
                          # invalidates the full-throttle bar just as surely as
                          # a different corpus does
CORPUS="${1:-$DEFAULT_CORPUS}"
# Overridable so the $BAR_VOCAB guard below is a live check rather than a
# branch that can never fire. It flows to the CLI, to the oracle generator, to
# the oracle's provenance stamp and to the HF timing runs, so changing it moves
# all four together and invalidates the stored bars automatically. Bounds match
# examples/bpe_train_cli.mlr's own so a bad value is rejected here rather than
# deeper in.
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
case "$VOCAB_SIZE" in
    *[!0-9]*) echo "FATAL: VOCAB_SIZE must be a positive integer (got '$VOCAB_SIZE')" >&2; exit 1 ;;
esac
if [ "${#VOCAB_SIZE}" -gt 7 ] || [ "$VOCAB_SIZE" -lt 5 ] || [ "$VOCAB_SIZE" -gt 1000000 ]; then
    echo "FATAL: VOCAB_SIZE must be between 5 and 1000000 (got '$VOCAB_SIZE')" >&2
    exit 1
fi
# B-6: normalise leading zeros. "08192" is numerically the bars' vocab size but
# string-compares unequal to $BAR_VOCAB, which would declare the bars
# inapplicable and spend three minutes re-measuring HF for nothing. 10# forces
# base 10 so "0012" is 12, not an octal parse error.
VOCAB_SIZE=$((10#$VOCAB_SIZE))
NTHREADS_FULL="$(nproc)"
# B-1: the only knob that was never validated. `[ "${HF_TIME:-0}" = "1" ]` reads
# HF_TIME=true / yes / TRUE as OFF, so an operator asking for a re-measurement
# silently got stored bars instead — while REPS and VOCAB_SIZE FATAL on garbage.
# Normalised and validated here so every knob behaves the same way.
HF_TIME="${HF_TIME:-0}"
case "$HF_TIME" in
    1|true|TRUE|True|yes|YES|on|ON)   HF_TIME=1 ;;
    0|false|FALSE|False|no|NO|off|OFF) HF_TIME=0 ;;
    *)
        echo "FATAL: HF_TIME must be 1/true/yes/on or 0/false/no/off (got '$HF_TIME')" >&2
        exit 1 ;;
esac

REPS="${REPS:-3}"
# Validated HERE, before any gate runs, rather than tolerated downstream:
# REPS=0 makes `seq 1 0` iterate zero times, so best_of sees no samples and the
# run dies in require_number with a "not a number" FATAL *after* the gates but
# *before* the summary — reporting a parse problem instead of the operator's
# actual mistake, and losing the report. A misconfiguration should be rejected
# at the point it is read.
# The `:-` default above guarantees REPS is non-empty here (an unset OR empty
# REPS becomes 3), so the patterns below only have to reject non-digits and
# out-of-range values. The 6-digit cap keeps `-lt` away from values that
# overflow the shell's integer parser and turn a clear message into a bash
# error; 999999 repetitions is already about three years of run time.
case "$REPS" in
    *[!0-9]*)
        echo "FATAL: REPS must be a positive integer (got '$REPS')" >&2
        exit 1 ;;
esac
if [ "${#REPS}" -gt 6 ]; then
    echo "FATAL: REPS is implausibly large (got '$REPS')" >&2
    exit 1
fi
if [ "$REPS" -lt 1 ]; then
    echo "FATAL: REPS must be at least 1 (got '$REPS') — a timing gate needs at least one run" >&2
    exit 1
fi
# Bars, in whole-process wall-clock seconds. Provenance — every sweep behind
# these two numbers, and why the older 54.59/8.64 pair was dropped — is in the
# HF_BAR_1T / HF_BAR_FULL entry of the usage block at the top of this file.
# Keep the two in sync: an evidence harness that documents one bar and enforces
# another is the same self-contradiction the summary logic was fixed for.
# With HF_TIME=1 the bars are re-derived from this run and those measurements,
# not these constants, decide G2/G3.
# S-2: remember whether these came from the environment BEFORE defaulting.
# A number the harness did not measure and cannot attribute must never be the
# thing a PASS rests on -- `HF_BAR_FULL=999` would otherwise print
# "ALL GATES PASS", exit 0, and describe the bar as "valid for this input".
# Operator-supplied bars are therefore treated as not-applicable below, which
# forces HF to be re-measured on the input under test; the supplied values are
# still displayed, but they decide nothing.
BARS_FROM_ENV=0
BAR_1T_SRC=default
BAR_FULL_SRC=default
if [ -n "${HF_BAR_1T+set}" ]; then BARS_FROM_ENV=1; BAR_1T_SRC=env; fi
if [ -n "${HF_BAR_FULL+set}" ]; then BARS_FROM_ENV=1; BAR_FULL_SRC=env; fi
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

# A-3: $WORK and $REF_DIR are fixed paths, so two overlapping invocations would
# interleave $OUT_JSON, $DIGESTS and the oracle cache and produce a false FAIL
# — this project has already lost time once to exactly that shape (a shared
# scratch dir manufacturing a phantom smoke failure).
#
# Fail fast rather than unique-per-run paths: unique paths would defeat the
# oracle cache entirely (every run paying a ~60s HF retrain) and would leave
# stale $WORK trees behind, whereas a clear "another run holds this" message
# tells the operator the true situation immediately. Fail fast rather than
# blocking, too — a queued second run would silently take minutes with no
# indication it was waiting.
#
# Note the ordinary flock-in-shell property: fd 9 is inherited by children, so
# if a run is KILLED mid-flight the lock stays held until its surviving child
# (an mlrc or python invocation) exits — normally seconds, up to about a minute
# for an in-progress HF training. That is why the message says "wait for it to
# finish" rather than claiming the holder is necessarily still a live harness.
exec 9>"$WORK/.lock" || exit 1
if ! flock -n 9; then
    echo "FATAL: another bpe_verify run holds $WORK" >&2
    echo "       (fixed work dir; wait for it to finish, or point WORK elsewhere)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# INVARIANT: every gate asserts only about artifacts produced by THIS run.
#
# $WORK persists between runs, so anything left there by a previous, possibly
# successful, run is a live hazard: a gate that reads it reports a verdict
# about a file this run never created. That has bitten this harness twice --
# a stale CLI binary surviving a failed compile (I-1), and G4 round-tripping
# the previous run's tokenizer.json after a total crash (P-1).
#
# Every reusable artifact is therefore deleted, and each gate treats "missing"
# as a failure rather than skipping the check. Destroying a previous run's
# artifacts is itself a side effect, so each deletion happens only AFTER every
# check that could still abort this run -- otherwise a typo'd corpus path or a
# failed compile takes a good tokenizer down with it:
#   $CLI_BIN   deleted before mlrc       (see the build section)
#   $RT_BIN    deleted before mlrc       (see the build section)
#   $DIGESTS   truncated after the build (never carries a prior run's md5)
#   $OUT_JSON  deleted after the build   (G1 recreates it; G4 refuses without
#                                         it) -- deleting before the corpus is
#                                         validated would destroy the previous
#                                         run's tokenizer over a typo'd path.
#
# The invariant covers MEASUREMENTS and CONSTANTS too, not just files -- a
# stored number compared against a different input is the same defect as a
# stale file. Two such bindings exist, and both are checked rather than assumed:
#   $REF_JSON        the HF oracle. Generated BY THIS HARNESS from $CORPUS at
#                    $VOCAB_SIZE (it is not external ground truth), so it is
#                    cached under a five-field stamp -- corpus realpath, corpus
#                    content size, vocab size, tokenizers version, and an md5
#                    of the generator recipe itself -- and regenerated whenever
#                    any of those differ.
#   HF_BAR_1T/_FULL  measured on $DEFAULT_CORPUS at $BAR_BYTES bytes,
#                    $BAR_VOCAB vocab, $BAR_CORES cores, and only from the
#                    built-in constants. If this run differs in ANY of those,
#                    or the bars came from the environment, they cannot decide
#                    a gate: HF_TIME=1 is forced and the bars are re-measured
#                    on the input actually under test.
# ---------------------------------------------------------------------------

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
CORPUS_REAL="$(readlink -f "$CORPUS")"
# S-1: -L dereferences. Without it a symlinked corpus reports the LINK's own
# size (e.g. 34 bytes), which is both a false "corpus bytes:" line and — far
# worse — a stamp whose size field never changes when the target's content
# does, silently disarming the oracle cache's content guard for symlinked
# datasets. Symlinked corpora are ordinary, so this is not a corner case.
CORPUS_BYTES="$(stat -Lc %s "$CORPUS")"
echo "corpus bytes:  $CORPUS_BYTES"
echo "cores:         $NTHREADS_FULL"
echo "vocab_size:    $VOCAB_SIZE"
echo "reps:          $REPS"

# R-2: the stored bars are a measurement of ($DEFAULT_CORPUS, $BAR_VOCAB).
# Comparing this run's seconds against them on any other input would print
# "MLRift beats HuggingFace" backed by a measurement of different work -- a
# 6.5 KB corpus's 0.005s against a 276 MB corpus's 50.07s. So the bars only
# apply to the pair they were measured on; for anything else HF is re-measured
# on the input actually under test, and the stored constants decide nothing.
# B-3: EVERY applicable reason is collected, not just the first. A single-reason
# chain let env-provenance mask a simultaneously foreign corpus, vocab or core
# count, so the operator fixed one cause and was surprised by the next.
BARS_APPLY=1
BARS_WHY=""
bars_reject() {       # bars_reject <reason>
    BARS_APPLY=0
    if [ -z "$BARS_WHY" ]; then BARS_WHY="$1"; else BARS_WHY="$BARS_WHY; $1"; fi
}
if [ "$BARS_FROM_ENV" = "1" ]; then
    bars_reject "HF_BAR_* were supplied from the environment — provenance unknown"
fi
if [ "$CORPUS_REAL" != "$(readlink -f "$DEFAULT_CORPUS")" ]; then
    bars_reject "corpus is not the one the bars were measured on"
fi
# A-1: content, not just path. Without this the chain checked WHERE the corpus
# is and never WHAT it contains, so regenerating, truncating or appending to
# corpus.txt in place left G2/G3 deciding against constants measured on
# different work — while the oracle cache, which does key on size, regenerated
# for that very same change. A gate could print PASS and exit 0 on input the
# bars had never seen.
if [ "$CORPUS_BYTES" != "$BAR_BYTES" ]; then
    bars_reject "corpus is $CORPUS_BYTES bytes, the bars were measured on $BAR_BYTES"
fi
if [ "$VOCAB_SIZE" != "$BAR_VOCAB" ]; then
    bars_reject "vocab_size $VOCAB_SIZE != the bars' $BAR_VOCAB"
fi
if [ "$NTHREADS_FULL" != "$BAR_CORES" ]; then
    bars_reject "this box has $NTHREADS_FULL cores, the bars were measured on $BAR_CORES"
fi
if [ "$BARS_APPLY" = "1" ]; then
    # This line states exactly what the checks above established and nothing
    # more: same corpus by realpath AND content size, same vocab size, same
    # core count. It is not a claim that the box is otherwise identical, nor
    # that the corpus bytes are the same ones (same path, same length).
    echo "HF bars:       1-thread ${HF_BAR_1T}s   full ${HF_BAR_FULL}s"
    echo "               (same corpus path and size, same vocab, same core count"
    echo "                as the stored measurement)"
else
    echo "HF bars:       NOT APPLICABLE — $BARS_WHY"
    if [ "$BARS_FROM_ENV" = "1" ]; then
        # B-2: naming a built-in default as a "supplied value" is a small lie
        # that sends the operator looking for an override they never set.
        echo "               1-thread ${HF_BAR_1T}s ($BAR_1T_SRC), full ${HF_BAR_FULL}s ($BAR_FULL_SRC)"
        echo "               shown for reference only; they decide nothing."
    fi
    echo "               forcing HF_TIME=1 so G2/G3 are decided against HF"
    echo "               measured on THIS corpus at THIS vocab size."
    HF_TIME=1
fi
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

# Only here -- past the corpus check AND past the build, the two stages that
# can still abort -- is it safe to destroy the previous run's results. Earlier
# placements meant a typo'd corpus path (N-4) or a rejected source file took a
# good tokenizer down with them on the way to a FATAL.
: > "$DIGESTS"
rm -f "$OUT_JSON"

# --- 2. the HF oracle -------------------------------------------------------
hdr "HF oracle"
# R-1: this file is GENERATED BY THIS HARNESS from $CORPUS at $VOCAB_SIZE, so
# it is only a valid oracle for that exact pair. Reusing it on `[ -s ]` alone
# meant a 6.5 KB corpus could be gated against a 276 MB corpus's oracle. The
# cache is therefore stamped with what produced it and regenerated on any
# mismatch. (Corpus size rather than a content hash: hashing 276 MB on every
# run costs more than it buys. A same-path, same-LENGTH content change is
# therefore not detected -- the one gap left open here, and deliberately so.)
REF_STAMP="$REF_DIR/tokenizer_ref.stamp"
# The tokenizers version is part of the key: the oracle IS HuggingFace's
# output, so an upgraded library can legitimately produce a different one and
# a cache from the old version would silently become the thing G1 compares
# against — byte-exactness against a version we are no longer running.
HF_VER="$("$PY" -c 'import tokenizers; print(tokenizers.__version__)' 2>/dev/null)"
if [ -z "$HF_VER" ]; then
    echo "FATAL: cannot import tokenizers with $PY — no oracle is possible" >&2
    exit 1
fi
# A-2: the oracle's own RECIPE is part of the key. The stamp used to record the
# inputs (corpus, size, vocab, library version) but not the configuration --
# pre-tokenizer, decoder, post-processor, special tokens -- so editing any of
# those and re-running silently reused an oracle built by the OLD recipe, and
# G1 then reported byte-exactness about a configuration this run never used.
# PYSUM only notices a changed COUNT of added tokens, not e.g. a flipped
# add_prefix_space. The recipe is written to a file and hashed so the hash is
# provably of the exact text executed, not of a second copy that could drift.
REF_RECIPE="$WORK/hf_oracle_recipe.py"
cat > "$REF_RECIPE" <<'PYGEN'
import sys, tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
corpus, out = sys.argv[1], sys.argv[2]
# Taken from the shell's $VOCAB_SIZE, never hardcoded: a second copy of the
# number here would silently train the oracle at a different size from the one
# the CLI is run at, and G1 would be comparing two different experiments.
vocab_size = int(sys.argv[3])
print("tokenizers", tokenizers.__version__, "vocab_size", vocab_size)
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.post_processor = processors.ByteLevel(trim_offsets=False)
tr = trainers.BpeTrainer(vocab_size=vocab_size,
                         special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tk.train([corpus], tr)
tk.save(out, pretty=True)
PYGEN
RECIPE_MD5="$(md5sum < "$REF_RECIPE" | cut -d' ' -f1)"
WANT_STAMP="corpus=$CORPUS_REAL bytes=$CORPUS_BYTES vocab=$VOCAB_SIZE tokenizers=$HF_VER recipe=$RECIPE_MD5"
# $REF_MIN is deliberately NOT part of the validity test: it is re-derived from
# $REF_JSON a few lines below on every run, so requiring it here would turn a
# missing derived file into a needless ~60s HF retrain.
if [ -s "$REF_JSON" ] && [ -f "$REF_STAMP" ] \
   && [ "$(cat "$REF_STAMP")" = "$WANT_STAMP" ]; then
    echo "reusing $REF_JSON"
    echo "  stamp: $WANT_STAMP"
else
    if [ -s "$REF_JSON" ] && [ -f "$REF_STAMP" ]; then
        echo "cached oracle does not match this run — regenerating"
        echo "  cached: $(cat "$REF_STAMP")"
        echo "  wanted: $WANT_STAMP"
    elif [ -s "$REF_JSON" ]; then
        echo "cached oracle has no provenance stamp — regenerating"
        echo "  wanted: $WANT_STAMP"
    fi
    echo "generating the HF reference (~1 minute for the full corpus)..."
    rm -f "$REF_STAMP" "$REF_MIN"
    "$PY" "$REF_RECIPE" "$CORPUS" "$REF_JSON" "$VOCAB_SIZE" \
        || { echo "FATAL: HF reference generation failed" >&2; exit 1; }
    printf '%s' "$WANT_STAMP" > "$REF_STAMP"
fi
"$PY" - "$REF_JSON" "$REF_MIN" <<'PYMIN' || { echo "FATAL: could not minify the HF reference" >&2; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
open(sys.argv[2], "w").write(json.dumps(d, separators=(",", ":"), ensure_ascii=False))
PYMIN
# The stamp above proves the oracle was generated from THIS corpus at THIS
# vocab size; these assertions prove it is also structurally whole, which a
# stamp cannot show (an interrupted generation leaves a matching stamp with a
# truncated file only if the stamp is written first — it is written last, but
# the file can still be damaged afterwards). They are corpus-agnostic on
# purpose (no hardcoded 8192/7980), so the harness stays usable on any corpus:
#   - exactly the 4 special tokens we train with,
#   - at least one merge (an oracle with none would make G1 near-vacuous),
#   - and the ByteLevel BPE size identity vocab == 4 + alphabet + merges,
#     which pins alphabet and merge counts to the vocab in one check and fails
#     on any truncation or substitution that disturbs their relationship.
"$PY" - "$REF_JSON" <<'PYSUM' || { echo "FATAL: the HF oracle failed its self-consistency check" >&2; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
m, v, at = d["model"]["merges"], d["model"]["vocab"], d["added_tokens"]
alpha = sum(1 for t in v if len(t) == 1)
print("oracle: vocab %d  merges %d  added_tokens %d  alphabet %d"
      % (len(v), len(m), len(at), alpha))
print("oracle: first merges", ["".join(x) for x in m[:5]])
ok = True
if len(at) != 4:
    ok = False; print("  ORACLE INVALID: expected 4 added_tokens, got %d" % len(at))
if len(m) < 1:
    ok = False; print("  ORACLE INVALID: no merges — G1 would assert almost nothing")
if len(v) != 4 + alpha + len(m):
    ok = False
    print("  ORACLE INVALID: vocab %d != 4 + alphabet %d + merges %d = %d"
          % (len(v), alpha, len(m), 4 + alpha + len(m)))
sys.exit(0 if ok else 1)
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
if [ "$BARS_APPLY" = "1" ]; then
    hdr "G2 — single-thread speed (best of $REPS, bar = HF 1-thread ${HF_BAR_1T}s)"
else
    hdr "G2 — single-thread speed (best of $REPS, bar = measured below)"
fi
G2_TIMES=""
for i in $(seq 1 "$REPS"); do
    t=$(timed_run 1 "$WORK/g2_$i.log") || { record G2 FAIL "run $i did not complete"; break; }
    echo "  run $i: ${t}s"
    G2_TIMES="$G2_TIMES $t"
done
if [ -z "${VERDICT[G2]:-}" ]; then
    G2_BEST=$(echo "$G2_TIMES" | best_of)
    require_number "$G2_BEST" "the G2 best-of-$REPS time"
    if [ "$BARS_APPLY" = "1" ]; then
        if awk -v a="$G2_BEST" -v b="$HF_BAR_1T" 'BEGIN{exit !(a<b)}'; then
            record G2 PASS "best ${G2_BEST}s <  HF ${HF_BAR_1T}s (runs:$G2_TIMES)"
        else
            record G2 FAIL "best ${G2_BEST}s >= HF ${HF_BAR_1T}s (runs:$G2_TIMES)"
        fi
    else
        # No verdict is recorded here on purpose. Leaving VERDICT[G2] unset
        # means the summary's `${VERDICT[G2]:-FAIL}` default applies if the
        # re-measurement below never runs, so the failure mode is a FAIL, never
        # a PASS backed by a bar measured on other input.
        echo "  best ${G2_BEST}s (runs:$G2_TIMES) — decided by the HF re-measurement below"
    fi
fi

# --- 5. G3 + M: full-throttle speed and peak RSS ----------------------------
if [ "$BARS_APPLY" = "1" ]; then
    hdr "G3 — ${NTHREADS_FULL}-thread speed (best of $REPS, bar = HF full ${HF_BAR_FULL}s)"
else
    hdr "G3 — ${NTHREADS_FULL}-thread speed (best of $REPS, bar = measured below)"
fi
G3_TIMES=""
for i in $(seq 1 "$REPS"); do
    t=$(timed_run "$NTHREADS_FULL" "$WORK/g3_$i.log") || { record G3 FAIL "run $i did not complete"; break; }
    echo "  run $i: ${t}s"
    G3_TIMES="$G3_TIMES $t"
done
if [ -z "${VERDICT[G3]:-}" ]; then
    G3_BEST=$(echo "$G3_TIMES" | best_of)
    require_number "$G3_BEST" "the G3 best-of-$REPS time"
    if [ "$BARS_APPLY" = "1" ]; then
        if awk -v a="$G3_BEST" -v b="$HF_BAR_FULL" 'BEGIN{exit !(a<b)}'; then
            record G3 PASS "best ${G3_BEST}s <  HF ${HF_BAR_FULL}s (runs:$G3_TIMES)"
        else
            record G3 FAIL "best ${G3_BEST}s >= HF ${HF_BAR_FULL}s (runs:$G3_TIMES)"
        fi
    else
        # See the matching note in G2: no verdict here means the summary
        # defaults to FAIL if the re-measurement never runs.
        echo "  best ${G3_BEST}s (runs:$G3_TIMES) — decided by the HF re-measurement below"
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
    # $OUT_JSON here is absent, or an EARLIER run-of-this-invocation's output,
    # or -- if this run died between the trainer's truncating open and its
    # completed write -- a partial file. None of those is a result this run
    # earned, and on the absent path md5sum would fail outright. So the digest
    # is recorded only on the success branches below.
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
# $OUT_JSON was deleted before the first training run, so if it is absent now
# no trainer invocation in this run ever produced a tokenizer. Round-tripping
# whatever happened to be on disk would report a green gate backed by a file
# this run did not create — the stale-artifact failure this harness has already
# been bitten by twice.
# -s, not -f: a zero-byte leftover is not a tokenizer, and letting one through
# would send the probe to tokenizer_load only to fail with a less clear message.
if [ ! -s "$OUT_JSON" ]; then
    record G4 FAIL "no tokenizer produced by this run — nothing to round-trip"
elif MLRIFT_BPE_OUT="$OUT_JSON" "$RT_BIN"; then
    record G4 PASS "decode(encode(x)) == x for every probe (incl. non-ASCII)"
else
    record G4 FAIL "a probe did not round-trip (see the output above)"
fi

# --- 7. optional: re-measure the HF bars ------------------------------------
if [ "$HF_TIME" = "1" ]; then
    hdr "HF re-measurement (best of $REPS each)"
    # Timed exactly the way MLRift is: whole-process wall clock of a run that
    # trains AND writes the tokenizer.json. Timing only `train()` would flatter
    # HF by excluding interpreter startup, import and serialization, which the
    # MLRift number necessarily includes.
    cat > "$WORK/hf_train.py" <<'PYHF'
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
corpus, out = sys.argv[1], sys.argv[2]
# From the shell's $VOCAB_SIZE, for the same reason as PYGEN: a bar measured at
# a different vocab size than the MLRift runs is a bar for different work.
vocab_size = int(sys.argv[3])
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.post_processor = processors.ByteLevel(trim_offsets=False)
tr = trainers.BpeTrainer(vocab_size=vocab_size,
                         special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tk.train([corpus], tr)
tk.save(out)
PYHF
    hf_timed() {      # hf_timed <rayon-threads-or-empty>
        local t0 t1
        t0=$(date +%s%N)
        if [ -n "$1" ]; then
            RAYON_NUM_THREADS="$1" "$PY" "$WORK/hf_train.py" "$CORPUS" "$WORK/hf_timed.json" "$VOCAB_SIZE" >/dev/null 2>&1
        else
            "$PY" "$WORK/hf_train.py" "$CORPUS" "$WORK/hf_timed.json" "$VOCAB_SIZE" >/dev/null 2>&1
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
    # S-3: this block is now REACHED WITHOUT THE OPERATOR OPTING IN whenever the
    # stored bars do not apply, so an HF failure here must not abort the run.
    # Aborting would lose the SUMMARY on any foreign-corpus run — the same
    # "report lost" shape fixed on the MLRift side in round 2. Record the
    # affected gates FAIL instead and let the summary print.
    hf1=""; hfn=""; HF_OK=1
    for i in $(seq 1 "$REPS"); do
        t=$(hf_timed 1)  || { HF_OK=0; break; }
        hf1="$hf1 $t"
        t=$(hf_timed "") || { HF_OK=0; break; }
        hfn="$hfn $t"
    done
    if [ "$HF_OK" = "0" ]; then
        # B-4: with applicable stored bars the gates already HAVE verdicts, so
        # "no bar could be established" would contradict the summary printed
        # moments later. Say what is true for each combination.
        if [ "$BARS_APPLY" = "1" ]; then
            echo "  the HF re-measurement did not complete — G2/G3 keep the verdicts"
            echo "  already decided against the applicable stored bars"
        else
            echo "  the HF re-measurement did not complete — no bar could be established"
        fi
        for g in G2 G3; do
            # Only fill in gates that have no verdict yet: one already recorded
            # FAIL for a crashed trainer keeps that more specific reason.
            if [ -z "${VERDICT[$g]:-}" ]; then
                record "$g" FAIL "HF re-measurement failed; no applicable bar for this input"
            fi
        done
    else
    HF1_BEST=$(echo "$hf1" | best_of)
    HFN_BEST=$(echo "$hfn" | best_of)
    require_number "$HF1_BEST" "the HF 1-thread best time"
    require_number "$HFN_BEST" "the HF full-throttle best time"
    # Tidiness 5: naming a stored bar beside a measurement it has no bearing on
    # invites the reader to compare them. Show it only when it actually applies.
    if [ "$BARS_APPLY" = "1" ]; then
        echo "  HF 1-thread runs:$hf1  best ${HF1_BEST}s   (stored bar ${HF_BAR_1T}s)"
        echo "  HF full runs:$hfn  best ${HFN_BEST}s   (stored bar ${HF_BAR_FULL}s)"
    else
        echo "  HF 1-thread runs:$hf1  best ${HF1_BEST}s"
        echo "  HF full runs:$hfn  best ${HFN_BEST}s"
    fi
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
    fi   # HF_OK
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
