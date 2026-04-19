#!/bin/bash
# examples/compare.sh <base> — compile examples/<base>.kr with mlrc,
# run examples/<base>_reference.py, diff the two outputs.
#
#   bash examples/compare.sh lif_neuron
#   bash examples/compare.sh two_neuron_synapse
#
# Any stdout divergence is a correctness failure — either the compiler
# lost precision on f64 math, or the kr/py implementations drifted.

set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
MLRC="${MLRC:-$REPO/build/mlrc}"

if [ $# -ne 1 ]; then
    echo "usage: $0 <base>   (expects $DIR/<base>.mlr + $DIR/<base>_reference.py)" >&2
    exit 2
fi
BASE="$1"
KR_SRC="$DIR/${BASE}.mlr"
PY_SRC="$DIR/${BASE}_reference.py"
[ -r "$KR_SRC" ] || { echo "compare: missing $KR_SRC" >&2; exit 2; }
[ -r "$PY_SRC" ] || { echo "compare: missing $PY_SRC" >&2; exit 2; }
[ -x "$MLRC"   ] || { echo "compare: missing mlrc ($MLRC)" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "compare: python3 required" >&2; exit 2; }

BIN=$(mktemp /tmp/mlrift_${BASE}_XXXX); rm -f "$BIN"
if ! "$MLRC" --arch=x86_64 "$KR_SRC" -o "$BIN" >/dev/null 2>&1; then
    echo "[$BASE] compile failed:" >&2
    "$MLRC" --arch=x86_64 "$KR_SRC" -o "$BIN"
    exit 1
fi
chmod +x "$BIN"

KR_OUT=$(mktemp); PY_OUT=$(mktemp)
"$BIN" > "$KR_OUT"
python3 "$PY_SRC" > "$PY_OUT"

KR_LINES=$(wc -l < "$KR_OUT")
PY_LINES=$(wc -l < "$PY_OUT")

if cmp -s "$KR_OUT" "$PY_OUT"; then
    echo "PASS [$BASE]: ${KR_LINES} lines agree byte-for-byte"
    rm -f "$BIN" "$KR_OUT" "$PY_OUT"
    exit 0
fi

echo "FAIL [$BASE]: MLRift (${KR_LINES} lines) disagrees with Python (${PY_LINES} lines)"
echo "--- first differing lines (- MLRift, + Python) ---"
diff "$KR_OUT" "$PY_OUT" | head -20
echo ""
echo "kr: $KR_OUT"
echo "py: $PY_OUT"
rm -f "$BIN"
exit 1
