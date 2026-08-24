#!/usr/bin/env python3
# Emit compact \p{L} and \p{N} codepoint RANGES as MLRift arrays.
# Uses the `regex` module to match Oniguruma's property semantics.
#
# M-6, and it is not hypothetical. This header used to claim the unicodedata
# fallback "agrees for BMP+SMP here". MEASURED on 2026-08-24, it does not:
#
#     regex 2026.7.19          uni_L_count=684  uni_N_count=146
#     unicodedata 15.0.0       uni_L_count=659  uni_N_count=137
#
# The committed std/bpe_unicode.mlr has 684/146, i.e. it was produced by the
# `regex` backend. The gap is a Unicode-VERSION gap (the `regex` wheel bundles
# its own, newer UCD; CPython's unicodedata is pinned to whatever that
# interpreter shipped), not a category-semantics gap — but it is a real gap:
# regenerating on a box without `regex` would have silently replaced the
# tables (25 fewer letter ranges, 9 fewer number ranges, and boundary shifts
# inside ranges that survive — e.g. U+0889..U+088F becomes U+0889..U+088E),
# changing where words split, with nothing in the diff naming the cause.
#
# Two things close that:
#   1. The generated file now carries a PROVENANCE line naming the backend,
#      its version, unicodedata.unidata_version, and the interpreter. A
#      regeneration that shifts behaviour now leaves a trace in the diff.
#   2. The fallback no longer happens by accident. Without `regex` this script
#      REFUSES to write, because a silent fallback is exactly how a wrong
#      table ships. Set MLRIFT_ALLOW_UNICODEDATA_FALLBACK=1 to override it
#      deliberately — and expect the counts above to change if you do.
import os, sys, unicodedata
try:
    import regex as _re
    BACKEND = f"regex {getattr(_re, '__version__', 'unknown')}"
    isL = lambda cp: bool(_re.match(r"\p{L}", chr(cp)))
    isN = lambda cp: bool(_re.match(r"\p{N}", chr(cp)))
except ImportError:
    print("=" * 74, file=sys.stderr)
    print("WARNING: the `regex` module is NOT installed.", file=sys.stderr)
    print("The committed std/bpe_unicode.mlr was generated WITH it (684 L /", file=sys.stderr)
    print("146 N ranges). The unicodedata fallback was measured on 2026-08-24", file=sys.stderr)
    print("to produce DIFFERENT tables (659 L / 137 N) — a newer bundled UCD in", file=sys.stderr)
    print("the `regex` wheel vs CPython's pinned unicodedata. Regenerating with", file=sys.stderr)
    print("the fallback would silently change where the pre-tokenizer splits.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix: `pip install regex` (or use a venv that has it) and re-run.", file=sys.stderr)
    print("Override, only if you mean it: MLRIFT_ALLOW_UNICODEDATA_FALLBACK=1", file=sys.stderr)
    print("=" * 74, file=sys.stderr)
    if os.environ.get("MLRIFT_ALLOW_UNICODEDATA_FALLBACK") != "1":
        print("REFUSING to write a table from an unverified backend.", file=sys.stderr)
        sys.exit(2)
    BACKEND = "unicodedata (FALLBACK — `regex` not installed; NOT the committed backend)"
    ud = unicodedata
    isL = lambda cp: ud.category(chr(cp)).startswith("L")
    isN = lambda cp: ud.category(chr(cp)).startswith("N")
PROVENANCE = (f"backend={BACKEND}; "
              f"unicodedata.unidata_version={unicodedata.unidata_version}; "
              f"python={sys.version.split()[0]}")
def ranges(pred, hi=0x110000):
    out, s = [], None
    for cp in range(hi):
        if pred(cp):
            if s is None: s = cp
        elif s is not None:
            out.append((s, cp - 1)); s = None
    if s is not None: out.append((s, hi - 1))
    return out
L = ranges(isL); N = ranges(isN)
OUT = sys.argv[1] if len(sys.argv) > 1 else "std/bpe_unicode.mlr"
with open(OUT, "w") as f:
    f.write("// GENERATED — \\p{L}/\\p{N} range tables for the BPE pre-tokenizer.\n")
    f.write("// Regenerate with: python3 scripts/gen_bpe_unicode_tables.py std/bpe_unicode.mlr\n")
    f.write(f"// PROVENANCE: {PROVENANCE}\n")
    f.write(f"static u64 uni_L_count = {len(L)}\n")
    f.write(f"static u64 uni_N_count = {len(N)}\n")
    f.write("static u64 uni_L_lo = 0\nstatic u64 uni_L_hi = 0\n")
    f.write("static u64 uni_N_lo = 0\nstatic u64 uni_N_hi = 0\n")
    f.write("static u64 uni_tables_init = 0\n\n")
    # store as init function filling alloc'd arrays (MLRift has no array literals)
    f.write("fn uni_tables_ensure() {\n    if uni_tables_init == 1 { return }\n")
    f.write(f"    uni_L_lo = alloc({len(L)}*8); uni_L_hi = alloc({len(L)}*8)\n")
    f.write(f"    uni_N_lo = alloc({len(N)}*8); uni_N_hi = alloc({len(N)}*8)\n")
    for i,(lo,hi) in enumerate(L):
        f.write(f"    store64(uni_L_lo+{i}*8, 0x{lo:x}); store64(uni_L_hi+{i}*8, 0x{hi:x})\n")
    for i,(lo,hi) in enumerate(N):
        f.write(f"    store64(uni_N_lo+{i}*8, 0x{lo:x}); store64(uni_N_hi+{i}*8, 0x{hi:x})\n")
    f.write("    uni_tables_init = 1\n}\n\n")
    f.write("""fn uni_bsearch(u64 lo_arr, u64 hi_arr, u64 n, u64 cp) -> u64 {
    u64 lo = 0; u64 hi = n
    while lo < hi {
        u64 mid = (lo + hi) / 2
        u64 rlo = 0; u64 rhi = 0
        unsafe { *((lo_arr + mid*8) as u64) -> rlo }
        unsafe { *((hi_arr + mid*8) as u64) -> rhi }
        if cp < rlo { hi = mid } else { if cp > rhi { lo = mid + 1 } else { return 1 } }
    }
    return 0
}
fn uni_is_letter(u64 cp) -> u64 { uni_tables_ensure(); return uni_bsearch(uni_L_lo, uni_L_hi, uni_L_count, cp) }
fn uni_is_number(u64 cp) -> u64 { uni_tables_ensure(); return uni_bsearch(uni_N_lo, uni_N_hi, uni_N_count, cp) }
""")
# Name the file actually written, not a hardcoded default: this used to print
# "wrote std/bpe_unicode.mlr" even when argv[1] pointed somewhere else.
print(f"wrote {OUT}  ({PROVENANCE}; uni_L_count={len(L)} uni_N_count={len(N)})")
