#!/bin/bash
# examples/lif_compare.sh — runs the MLRift and Python LIF neurons
# and diffs their spike-step outputs. Any difference is a correctness
# signal: either the compiler lost precision or the two implementations
# drifted apart.
#
# Usage:
#   bash examples/lif_compare.sh
# Env:
#   MLRC=<path>   compiler to use (default: ../build/mlrc from this dir)

set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
MLRC="${MLRC:-$REPO/build/mlrc}"
if [ ! -x "$MLRC" ]; then echo "lif_compare: mlrc not found ($MLRC)" >&2; exit 2; fi
command -v python3 >/dev/null 2>&1 || { echo "lif_compare: python3 required" >&2; exit 2; }

BIN=$(mktemp /tmp/lif_neuron_XXXX); rm -f "$BIN"
if ! "$MLRC" --arch=x86_64 "$DIR/lif_neuron.kr" -o "$BIN" >/dev/null 2>&1; then
    echo "lif_compare: compile failed" >&2
    "$MLRC" --arch=x86_64 "$DIR/lif_neuron.kr" -o "$BIN"
    exit 1
fi
chmod +x "$BIN"

KR_OUT=$(mktemp); PY_OUT=$(mktemp)
"$BIN" > "$KR_OUT"
python3 "$DIR/lif_neuron_reference.py" > "$PY_OUT"

KR_SPIKES=$(wc -l < "$KR_OUT")
PY_SPIKES=$(wc -l < "$PY_OUT")

if cmp -s "$KR_OUT" "$PY_OUT"; then
    echo "PASS: MLRift and Python agree on all ${KR_SPIKES} spike step-indices"
    echo "      first spike: step $(head -1 "$KR_OUT")"
    echo "      last  spike: step $(tail -1 "$KR_OUT")"
    rm -f "$BIN" "$KR_OUT" "$PY_OUT"
    exit 0
fi

echo "FAIL: MLRift (${KR_SPIKES} spikes) disagrees with Python (${PY_SPIKES} spikes)"
echo "--- first differing lines (- MLRift, + Python) ---"
diff "$KR_OUT" "$PY_OUT" | head -20
echo ""
echo "kr output preserved at: $KR_OUT"
echo "py output preserved at: $PY_OUT"
rm -f "$BIN"
exit 1
