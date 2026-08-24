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

emit_flag() {
    # $1 = flag=/tmp/foo.co  (single-flag emit)
    local fp="$1"
    if "$MLRC" --arch=x86_64 "$fp" /tmp/empty.mlr > /dev/null 2>&1; then
        echo "  ok  $fp"
    else
        echo "  FAIL $fp" >&2
        return 1
    fi
}

emit_amdgpu_native() {
    # $1 = examples/llm/X.mlr → /tmp/X.co (via AST-walker)
    local src="$1"
    local stub_path="$2"
    if "$MLRC" --target=amdgpu-native "$src" -o "$stub_path" > /dev/null 2>&1; then
        echo "  ok  $src → ${stub_path}.co"
    else
        echo "  FAIL $src" >&2
        return 1
    fi
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
