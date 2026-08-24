#!/bin/bash
# rebuild_helper_cos.sh — rebuild every `/tmp/*.co` the GPU runtime expects.
#
# PROJECT RULE: no ROCm / HIP / hipcc in the build — every production `.co`
# must come out of MLRift's own AST-walker emitters. Everything below obeys
# that rule EXCEPT path 4 (mks8 / mks16), which is a documented, isolated
# exception tracked at the bottom of this file. Do not add new hipcc calls.
#
# Emit paths in mlrc, with different invocation conventions:
#
#   1. Single-flag emit (`--emit-amdgpu-<kernel>=<path>`):
#        Produces a `.co` directly from the AST-walker emitters.
#        Used for the ~25 kernels with a dedicated flag in src/main.mlr.
#
#   2. Native source-compile (`--target=amdgpu-native src.mlr -o stub`):
#        Writes <stub>.co by name-routing the @kernel in the source file
#        through the AST-walker recognizers in src/format_amdgpu.mlr.
#        Used where no dedicated `--emit-amdgpu-*` flag exists — either
#        because it was deleted in favour of the AST path (gemv_f32,
#        gemm_f32) or because format_hip could never lower the body
#        (silu_mul_f32's `silu_f32(g)` call).
#
#   These four (bf16_to_f32, gemv_f32, gemm_f32, residual_add_f32) used to
#   go through `--arch=x86_64 --target=hip-amd`, which forks hipcc. They were
#   moved to path 2 on 2026-08-24; the AST-walker `.co` are bare gfx1100 ELFs
#   rather than clang offload bundles, with identical explicit kernarg layouts
#   (24 / 40 / 48 / 32 bytes) and identical exported symbol names.
#
#   All four exit rc=0 and emit a correct `.co`, but "cleanly" would overstate
#   it: gemv_f32 and gemm_f32 emit recognizer-probe noise on stderr first.
#   See the note above _emit_run.
#
#   HARDWARE COVERAGE of these four, as of 2026-08-24 — do not assume an LLM
#   token-md5 run exercises them, it does not:
#     bf16_to_f32       LOAD-bearing only. inference_gpu.mlr:360 refuses GPU
#                       init if it is missing (verified: displacing it makes
#                       qwen3_generate exit rc=1), but a deliberately wrong
#                       build (bf16 value +1 ULP) changed NO tokens in either
#                       the mega-kernel or the per-op config. Not dispatched
#                       on the qwen3-0.6B path.
#     residual_add_f32  Same: a build computing `a+b+0.5` changed no tokens.
#                       qwen3.mlr:2510 routes hidden==1024 through the fused
#                       gpu_residual_rmsnorm_1024 instead.
#     gemv_f32          Covered by examples/llm/gemv_f32_launch.mlr vs its CPU
#                       reference: max_abs 7.6e-5, max_rel 3.1e-7.
#     gemm_f32          Covered by examples/llm/gemm_f32_launch.mlr vs its CPU
#                       reference: max_abs 4.6e-5, max_rel 3.7e-7.
#   Those two launchers are the ONLY correctness gate these kernels have.
#   Run them after touching the emitters; the LLM gate will not catch it.
#
# Usage:
#   scripts/rebuild_helper_cos.sh [path/to/mlrc]
#
# Defaults to ./build/mlrc if no path given.
set -e
MLRC="${1:-${MLRC:-./build/mlrc}}"
[ -x "$MLRC" ] || { echo "rebuild_helper_cos: mlrc not executable: $MLRC" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

mkdir -p /tmp
echo 'fn main() {}' > /tmp/empty.mlr

# stderr is captured, not discarded. A FAILING emit dumps it — previously
# `2>&1` swallowed the compiler's diagnostic along with the noise, so a real
# failure printed only "FAIL <src>" with no reason.
#
# On SUCCESS stderr is suppressed but is NOT empty for every kernel: the
# amdgpu-native recognizers are tried in sequence and each one that declines
# writes a line before the right one matches. Today `gemv_f32` prints 16 ×
# "kernel index prologue must be exactly `block_idx_x()`" and `gemm_f32` 2 ×,
# because those lowerers expect a `block_idx_x()` prologue while gemv/gemm use
# `tid_x()`. rc is 0 and the emitted `.co` is correct — this is probe noise
# from a first-match-wins design, not a diagnosis of the source. It is worth
# fixing in src/format_amdgpu.mlr (a rejection should be silent and only the
# final no-match should speak, so a genuine error is not lost in the noise),
# but that is a compiler change, not a script change.
_emit_run() {
    # $1 = human label, rest = argv for $MLRC
    local label="$1"; shift
    local errf; errf="$(mktemp)"
    if "$MLRC" "$@" > /dev/null 2>"$errf"; then
        echo "  ok  $label"
        rm -f "$errf"
    else
        echo "  FAIL $label" >&2
        sed 's/^/      /' "$errf" >&2
        rm -f "$errf"
        return 1
    fi
}

emit_flag() {
    # $1 = flag=/tmp/foo.co  (single-flag emit)
    _emit_run "$1" --arch=x86_64 "$1" /tmp/empty.mlr
}

emit_amdgpu_native() {
    # $1 = examples/llm/X.mlr → /tmp/X.co (via AST-walker)
    _emit_run "$1 → ${2}.co" --target=amdgpu-native "$1" -o "$2"
}

echo "=== path 1: single-flag AST-walker emits ==="
emit_flag '--emit-amdgpu-gemv-coop-f32=/tmp/gemv_coop_f32.co'
emit_flag '--emit-amdgpu-gemv-coop-bf16-f32=/tmp/gemv_coop_bf16_f32.co'
emit_flag '--emit-amdgpu-gemv-coop-f32-batched=/tmp/gemv_coop_f32_batched.co'
emit_flag '--emit-amdgpu-gemv-coop-bf16-f32-batched=/tmp/gemv_coop_bf16_f32_batched.co'
emit_flag '--emit-amdgpu-gemv-coop-q4-0-f32-batched=/tmp/gemv_coop_q4_0_f32_batched.co'
emit_flag '--emit-amdgpu-rope-qwen3=/tmp/rope_qwen3_f32.co'
emit_flag '--emit-amdgpu-qkv-split-f32=/tmp/qkv_split_f32.co'
emit_flag '--emit-amdgpu-qkv-split-f32-batched=/tmp/qkv_split_f32_batched.co'
emit_flag '--emit-amdgpu-qkv-split-f32-14b=/tmp/qkv_split_f32_14b.co'
emit_flag '--emit-amdgpu-qkv-split-f32-batched-14b=/tmp/qkv_split_f32_batched_14b.co'
emit_flag '--emit-amdgpu-qkv-split-f32-speck4=/tmp/qkv_split_f32_speck4.co'
emit_flag '--emit-amdgpu-attn-decode-f32=/tmp/attn_decode_f32.co'
emit_flag '--emit-amdgpu-attn-decode-f32-14b=/tmp/attn_decode_f32_14b.co'
emit_flag '--emit-amdgpu-attn-decode-f32-speck4=/tmp/attn_decode_f32_speck4.co'
emit_flag '--emit-amdgpu-extract-q-qwen3-f32=/tmp/extract_q_qwen3_f32.co'
emit_flag '--emit-amdgpu-extract-k-qwen3-f32=/tmp/extract_k_qwen3_f32.co'
emit_flag '--emit-amdgpu-extract-k-qwen3-f32-speck4=/tmp/extract_k_qwen3_f32_speck4.co'
emit_flag '--emit-amdgpu-insert-k-qwen3-f32=/tmp/insert_k_qwen3_f32.co'
emit_flag '--emit-amdgpu-insert-k-qwen3-f32-speck4=/tmp/insert_k_qwen3_f32_speck4.co'
emit_flag '--emit-amdgpu-head-extract-f32=/tmp/head_extract_f32.co'
emit_flag '--emit-amdgpu-head-insert-f32=/tmp/head_insert_f32.co'
emit_flag '--emit-amdgpu-transpose-f32=/tmp/transpose_f32.co'
emit_flag '--emit-amdgpu-kv-broadcast-f32=/tmp/kv_broadcast.co'
emit_flag '--emit-amdgpu-embedding-lookup-f32=/tmp/embedding_lookup_f32.co'
emit_flag '--emit-amdgpu-argmax-logits-f32=/tmp/argmax_logits_f32.co'
emit_flag '--emit-amdgpu-qknorm-f32=/tmp/qknorm_f32.co'
emit_flag '--emit-amdgpu-silu-mul-f32-batched=/tmp/silu_mul_f32_batched.co'
emit_flag '--emit-amdgpu-rmsnorm-f32-N=1024:/tmp/rmsnorm_f32_1024.co'
emit_flag '--emit-amdgpu-rmsnorm-f32-N=5120:/tmp/rmsnorm_f32_5120.co'

echo "=== path 2: AST-walker native source-compiles ==="
emit_amdgpu_native 'examples/llm/bf16_to_f32.mlr'      /tmp/bf16_to_f32
emit_amdgpu_native 'examples/llm/gemv_f32.mlr'         /tmp/gemv_f32
emit_amdgpu_native 'examples/llm/gemm_f32.mlr'         /tmp/gemm_f32
emit_amdgpu_native 'examples/llm/residual_add_f32.mlr' /tmp/residual_add_f32
emit_amdgpu_native 'examples/llm/silu_mul_f32.mlr'     /tmp/silu_mul_f32

echo "=== mega-kernel .cos ==="
"$MLRC" --emit-amdgpu-qwen3-megakernel-v2=/tmp/qwen3_layer_megakernel_v2.co examples/llm/qwen3_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  qwen3_layer_megakernel_v2.co"
"$MLRC" --emit-amdgpu-qwen3-megakernel-speck4-v2=/tmp/qwen3_layer_megakernel_speck4_v2.co examples/llm/qwen3_layer_megakernel_speck4.mlr > /dev/null 2>&1 \
    && echo "  ok  qwen3_layer_megakernel_speck4_v2.co"
"$MLRC" --emit-amdgpu-llama-megakernel-v2=/tmp/llama_layer_megakernel_v2.co examples/llm/llama_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  llama_layer_megakernel_v2.co"
"$MLRC" --emit-amdgpu-llama-3b-megakernel-v2=/tmp/llama_3b_layer_megakernel_v2.co examples/llm/llama_3b_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  llama_3b_layer_megakernel_v2.co"
"$MLRC" --emit-amdgpu-llama-megakernel-speck4-v2=/tmp/llama_layer_megakernel_speck4_v2.co examples/llm/llama_layer_megakernel_speck4.mlr > /dev/null 2>&1 \
    && echo "  ok  llama_layer_megakernel_speck4_v2.co"
"$MLRC" --emit-amdgpu-mistral-megakernel-v2=/tmp/mistral_layer_megakernel_v2.co examples/llm/mistral_layer_megakernel.mlr > /dev/null 2>&1 \
    && echo "  ok  mistral_layer_megakernel_v2.co"

# ── path 4: THE ONE REMAINING hipcc DEPENDENCY ───────────────────────────
# This VIOLATES the project's "no ROCm / HIP / hipcc in the build" rule and
# is kept only because there is no MLRift source to AST-walk. It is an
# acknowledged debt, not an accepted design.
#
#   Sources:  examples/llm/qwen3_layer_megakernel_speck8.hip.cpp
#             examples/llm/qwen3_layer_megakernel_speck16.hip.cpp
#   These are hand-written HIP C++ (not .mlr), so "porting" them means
#   writing MLRift `@kernel` implementations of the M=8 / M=16 speculative
#   mega-kernels from scratch, the way `--emit-amdgpu-qwen3-megakernel-v2`
#   and `-speck4-v2` already cover M=1 and M=4. That is a slice of work
#   (tracked as slices 4.21+), not a script fix.
#
#   Consumer: the qwen3-0.6B PLD speculative-decode path, opted into with
#             MLRIFT_NATIVE_MEGAKERNEL=2 + MLRIFT_QWEN3_MEGAKERNEL_SPECK8/16=1
#             + MLRIFT_SPEC_K=8/16. This is what produces the 200+ tok/s
#             headline rows.
#
#   Without them: nothing breaks. The driver degrades to the per-op
#             M_eff chain (~20 tok/s at spec_K=16). Every other kernel in
#             this script, and the M=1 / M=4 mega-kernel rows, are 100 %
#             AST-walker output and need no hipcc at all.
#
# DO NOT add further hipcc call sites here. If you touch this block, the
# only acceptable direction is deleting it in favour of an AST-walker emit.
echo "=== path 4: hipcc compile of mks8 / mks16 mega-kernels (KNOWN RULE VIOLATION) ==="
if command -v hipcc > /dev/null 2>&1; then
    echo "  NOTE: shelling out to hipcc — the only non-AST-walker .co in this script."
    hipcc --offload-arch=gfx1100 --genco -O3 examples/llm/qwen3_layer_megakernel_speck8.hip.cpp -o /tmp/qwen3_layer_megakernel_speck8.co > /dev/null 2>&1 \
        && echo "  ok  qwen3_layer_megakernel_speck8.co  (hipcc)" \
        || echo "  FAIL qwen3_layer_megakernel_speck8.co"
    hipcc --offload-arch=gfx1100 --genco -O3 examples/llm/qwen3_layer_megakernel_speck16.hip.cpp -o /tmp/qwen3_layer_megakernel_speck16.co > /dev/null 2>&1 \
        && echo "  ok  qwen3_layer_megakernel_speck16.co  (hipcc)" \
        || echo "  FAIL qwen3_layer_megakernel_speck16.co"
else
    echo "  skip mks8/mks16 — hipcc not on PATH. This is the EXPECTED state:"
    echo "       the .hip.cpp sources have no AST-walker port yet, so spec_K=8/16"
    echo "       falls back to the per-op chain. Every other .co above is"
    echo "       AST-walker output and is unaffected."
fi

echo
n=$(ls /tmp/*.co 2>/dev/null | wc -l)
echo "Done. /tmp/*.co count = $n"
