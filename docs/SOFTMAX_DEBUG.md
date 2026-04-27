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

## Second debug pass (2026-04-27, also reverted)

Built five progressively simpler probes that replaced the kernel body
while keeping the same KD setup (group_segment_fixed_size=1024,
kt_push_lds, rsrc1=0xE0AF0041, rsrc2=0x009E):

- probe1: full max-reduce, no exp/sum — out = 0 everywhere
- probe2: pure passthrough (no LDS at all) — **works**, out[i]=in[i]
- probe3: lane k writes tid as f32 to LDS[tid], reads LDS[(255-tid)*4]
  — out = 0 everywhere
- probe4: only lane 0 writes 0xCAFEBABE to LDS[0] (saveexec dance),
  all lanes broadcast-read — out = 0 everywhere
- probe5: ALL lanes write 0xCAFEBABE to LDS[0] (no gating), broadcast
  read — out = 0 everywhere

**Bug found:** the AMDGPU metadata note (NT_AMDGPU_METADATA in .note)
hardcodes `.group_segment_fixed_size: 0` regardless of what KD says.
At line 436 of `amd_emit_metadata_body`. The runtime trusts this note
over KD for LDS allocation, so even though KD reserves 1024 bytes the
actual LDS region is 0.

**Fix:** read group_segment_size from kt entry (field index 13) and
emit it in metadata.

**Why N3 LDS still works without the fix:** unclear — N3 has an
identical layer-cake (kt_push_lds + group_segment=2048 in KD + 0 in
metadata) and produces correct results most of the time. Either the
runtime falls back to KD when metadata says 0, or the heisenbug pattern
that hit 256/320/768³ in the past is the SAME bug just less obvious.

**State at end of session 2:** applied the metadata fix and verified
the .co's note now says 1024. But probe5 still produced out=0 — and
N3 LDS GEMM also regressed (max_abs ≈ 0x7F7FFFFF). Then verified the
N3 regression existed on a clean tree too — sclk reads `0 MHz` from
sysfs, GPU is in deep-sleep state. Reverted all debug changes; the
softmax kernel + metadata fix are not committed.

## Things NOT tried — start here next session

The core finding from session 2 is that **LDS round-trip itself fails**
in the current setup even at the most reduced probe (probe5), despite
group_segment in KD = 1024. So before debugging softmax-specific logic,
fix the LDS infrastructure. Order of operations:

1. **Re-test the probes when GPU is in a healthy clock state** — at
   end of session 2 sclk read 0 MHz and N3 LDS GEMM also regressed.
   Force the GPU into known-good state first (`scripts/mlrift-gpu-perf-mode.sh
   high` with sudo, or run a long workload to wake it). Re-run probe5;
   if 0xCAFEBABE comes through, the metadata fix alone resolves it.
2. **Apply the metadata fix permanently** — `amd_emit_metadata_body` at
   line 436 reads `group_seg_sz` from kt entry index 13 and emits via
   `mp_uint`. The fix is one line. Once probe5 works, audit N3/N4 for
   regression (N3 had heisenbug history that may be the same bug).
3. **Then proceed to the original "things not tried" list:**
   probe-max-only (write LDS[0] to out, skip exp/sum), buffer_gl0_inv
   placement, lane-0 broadcast read with explicit zero VGPR, gfx11
   split barrier (s_barrier_signal/wait), single-wave version.

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
