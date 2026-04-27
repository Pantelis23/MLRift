# softmax_f32 native emit — in-progress (reverted in commit a347cc5+1)

Attempted in the same session as Phase 3a unary extension. Got far enough
to land a 229-dword kernel that assembles cleanly and dispatches, but
produces `+Inf` for every output cell. Reverted before commit.

## What works

- Assembly source at /tmp/softmax.s (still on disk after revert) — 229
  dwords of LDS-based pairwise max-reduce + sum-reduce, block=256,
  grid=M.
- llvm-mc accepts the .s; bytes round-trip via the dword extraction
  Python script.
- Splice into `src/format_amdgpu.mlr` as `_emit_softmax_f32_body` +
  `emit_amdgpu_softmax_f32_blob` + `amdgpu_lower_llm_softmax_f32`
  worked syntactically — bootstrap clean.
- Phase 2 routing (`tok_matches "softmax_f32"` → lowerer) wired in.
- Launcher at `examples/llm/softmax_f32_launch.mlr` (deleted on revert)
  fills 4×256 inputs, runs, checks per-row sum ≈ 1.0.

## What's broken

All output cells = `+Inf` (bit pattern 0x7F800000). Symptom is consistent
with `sum-reduce` producing 0 → `v_rcp_f32(0) = +Inf` → `e * Inf = Inf`.

Working theory: max-reduce is returning a junk value (not the actual
row max). Then `x - max = very_negative`, `exp(very_negative) = 0`,
sum-reduce of zeros = 0, rcp(0) = inf. But the reduction logic looks
correct on paper:

```
ds_load_b32 v7, v6 offset:STRIDE*4   # read LDS[tid+stride]
s_waitcnt lgkmcnt(0)
v_max_f32 v5, v5, v7
v_cmp_gt_u32 vcc_lo, STRIDE, v0      # only lanes < stride store back
s_and_saveexec_b32 s12, vcc_lo
ds_store_b32 v6, v5
s_mov_b32 exec_lo, s12
s_waitcnt lgkmcnt(0)
s_barrier
```

8 strides (128, 64, 32, 16, 8, 4, 2, 1).

## Things tried (all failed)

1. Found and fixed a kernarg-SGPR offset bug: m at s8 (was reading s10),
   n at s10 (was reading s12). Both addresses were correct, only the
   bounds-check + n-multiplier were wrong. After fix, ALL rows fail
   (was: row 0 only).

## Things NOT tried — start here next session

1. **Probe what max-reduce actually produces** — modify the kernel to
   write `max_value` to `out[row, 0]` (skip the exp/sum chain). If
   max is correct, bug is in exp or sum-reduce. If max is junk, bug
   is in max-reduce.
2. **Add `buffer_gl0_inv` after each `s_barrier`** — N3 LDS GEMM
   needed this; LDS-only ops technically shouldn't, but RDNA3 has
   surprising ordering quirks.
3. **Verify lane 0 actually reads LDS[0] correctly** — the broadcast
   read uses `v9 = 0` as the addr; check this isn't getting clobbered
   between the reduction and the broadcast read.
4. **Use `s_barrier_signal` + `s_barrier_wait` (gfx11 split form)**
   instead of plain `s_barrier`.
5. **Single-wave version (block=32, n=32)** — sidesteps cross-wave
   barriers entirely; if that works, the multi-wave LDS sync is the
   issue.

## Files (all reverted)

- `examples/llm/softmax_f32.mlr` — placeholder @kernel source
- `examples/llm/softmax_f32_launch.mlr` — host launcher with per-row
  sum verification and tolerance 0.01
- 229-line `_emit_softmax_f32_body` helper +
  `emit_amdgpu_softmax_f32_blob` (file write) +
  `amdgpu_lower_llm_softmax_f32` (kt_push for module emit) inside
  `src/format_amdgpu.mlr`
- routing case in `amdgpu_lower_kernel_to_table`
- `--emit-amdgpu-softmax-f32=` flag in `src/main.mlr`

## Estimate to finish

~1 focused session (2-3h). Probe step 1 above, fix, verify per-row sums
≈ 1.0 across M ∈ {1, 4, 64, 256}, then commit. Layernorm follows the
same shape (mean reduction, variance reduction, normalize, scale-shift)
with similar primitives — once softmax is solid, layernorm is mostly a
copy with `v_max_f32` → `v_add_f32` for mean, `v_mul_f32(d, d)` plus
sum for variance.
