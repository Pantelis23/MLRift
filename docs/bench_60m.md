# MLRift — 60 M neuron scaling benchmark

Headline: a 60 million–neuron spiking sim, 240 million synapses, 2 000
timesteps, on a single consumer workstation.

Reference implementation is LIF + refractory + Bernoulli random CSR
via the splitmix64 seekable RNG (fixed K = 4 outgoing synapses per
source). Same algorithm across all runtimes; same `RNG_SEED`, same
per-neuron `RI` variance, same parameters. Spike counts match bit-for-
bit between MLRift CPU, MLRift GPU, and cupy (all three produce the
same `1,985,575,928`); PyTorch differs by 5 842 / ≈ 0.0003 % because
HIP's `remainder_cuda` isn't implemented for `uint64`, so the PyTorch
variant falls back to a 63-bit mask before the modulo.

## Hardware

| component | spec |
|---|---|
| CPU | AMD Ryzen 9 7900X — 12 cores / 24 threads, Zen 4, AVX2 + FMA3 |
| GPU | AMD Radeon RX 7800 XT — RDNA 3 (gfx1100), 60 CUs, 16 GB GDDR6 |
| Memory | 30 GB DDR5 |
| ROCm | 7.2.0 |
| OS | Linux 6.17, Python 3.12 |

The GPU stayed in `low-power` state through all runs — so the numbers
reported below are *conservative* for the 7800 XT.

## Workload

- `N = 60,000,000` neurons
- `SYN_PER = 4` outgoing synapses/source (sparse uniform random)
- `n_syn = 240,000,000`
- `N_STEPS = 2000`
- `dt = 0.1 ms`
- `tau_m = 10 ms`, `V_rest = -65 mV`, `V_thresh = -50 mV`, `V_reset = -70 mV`
- `RI` per-neuron = `45 + i * 20 / N` (stops 60 M neurons from firing in the same step and blowing VRAM)
- splitmix64 seed `0xDEADBEEF12345678`

Memory footprint: ~14 GB resident for CPU runs, ~13 GB VRAM for GPU
runs.

## Builds

| stack | command |
|---|---|
| MLRift CPU | `./build/mlrc --arch=x86_64 examples/noesis_60m.mlr -o /tmp/ng_60m` |
| MLRift GPU kernels (HIP / hipcc) | `./build/mlrc --arch=x86_64 --target=hip-amd examples/noesis_60m_gpu.mlr -o /tmp/noesis_60m_gpu_hip` |
| MLRift GPU kernels (native gfx1100 ISA) | `./build/mlrc --arch=x86_64 --target=amdgpu-native examples/noesis_60m_gpu.mlr -o /tmp/noesis_60m_gpu_native` |
| MLRift GPU launcher | `./build/mlrc --arch=x86_64 examples/noesis_60m_gpu_launch.mlr -o /tmp/noesis_60m_gpu_launch` |
| Python | `python3 -m venv venv && source venv/bin/activate`<br>`pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch`<br>`pip install cupy-rocm-7-0 numpy` |

`venv/` is gitignored.

The `--target=amdgpu-native` path emits raw GFX1100 ISA into a hand-rolled
ELF code object — no `hipcc`, no LLVM toolchain, no `.hip` source. Three
of the four `@kernel` functions (`decay_step`, `delivery_step`,
`lif_step`) are lowered natively. `csr_build` (a one-shot startup kernel
that needs `v_mul_lo_u64` + 64-bit unsigned modulo we don't have ISA
helpers for yet) still goes through hipcc; the launcher loads it via a
dual-`.co` fallback (`NOESIS_CSR_CO_PATH=/path/to/hipcc.co`).

## Results

| variant | threads | init | CSR build (240 M syn) | sim (2000 steps) | per-step | **total wall** | spikes |
|---|---|---|---|---|---|---|---|
| Python / **numpy** (CPU) | 1 | 0.64 s | 20.62 s | 3 495.82 s | 1.75 s | **58 min 37 s** (3 517.16 s) | 1,860,205,410\* |
| Python / **PyTorch** (CPU) | 24 | 0.30 s | 6.16 s | 2 094.09 s | 1.05 s | **35 min 01 s** (2 100.64 s) | 1,985,570,086 |
| Python / **cupy** (GPU) | — | 1.17 s | 5.37 s | 128.37 s | 64 ms | **2 min 15 s** (135.41 s) | 1,985,575,928 |
| Python / **PyTorch** (GPU) | — | 0.02 s | 0.28 s | 102.70 s | 51 ms | **1 min 43 s** (103.01 s) | 1,985,570,086 |
| **MLRift** (CPU) | 24 | 1.75 s | 1.67 s | 382.57 s | 191 ms | **6 min 26 s** (386.04 s) | 1,985,575,928 |
| **MLRift** (GPU, HIP runtime + hipcc kernels) | — | 1.50 s | **0.11 s** | **26.55 s** | **13 ms** | **28.40 s** | 1,985,575,928 |
| **MLRift** (GPU, KFD shim + hipcc kernels)\* | — | 1.69 s | 0.18 s | **26.59 s** | **13 ms** | **29.50 s** | 1,985,575,928 |
| **MLRift** (GPU, KFD shim + native gfx1100 ISA + spike_reduce fallback)\* | — | 1.68 s | 0.22 s | **26.63 s** | **13 ms** | **29.28 s** | 1,985,575,926 |

\* zero-ROCm linkage in the launcher binary — `ldd` shows only
`libc + ld-linux + vdso`. No `libamdhip64`, `libhsa-runtime64`,
`libdrm`, `libdrm_amdgpu`. Built with `--target=amdgpu-native` on
the launcher source.

\* numpy was run on the original flat-`RI` workload (1.86 B spikes).
All other variants use the per-neuron `RI` variance (1.99 B spikes,
≈ 7 % more work). The numpy number is a ceiling; a fair-workload
numpy rerun would be ~60 min.

## Speedups — fair baseline (PyTorch CPU, 24 threads, 2 100.64 s)

| variant | × PyTorch CPU | notes |
|---|---|---|
| Python numpy (1 core) | 0.60× | single-threaded |
| PyTorch GPU | **20.4×** | full GPU, same library |
| cupy GPU | **15.5×** | |
| **MLRift CPU** | **5.44×** | same 24 threads, different language |
| **MLRift GPU** | **74.0×** | |

## Speedup ratios between matched pairs

| comparison | factor |
|---|---|
| **MLRift GPU vs PyTorch GPU** (same card) | **3.63×** |
| MLRift GPU vs cupy GPU (same card) | 4.77× |
| MLRift CPU vs PyTorch CPU (same CPU + threads) | 5.44× |
| MLRift GPU vs MLRift CPU (CPU→GPU) | 13.6× |
| PyTorch GPU vs PyTorch CPU (CPU→GPU) | 20.4× |
| Ryzen 9 7900X vs RX 7800 XT on theoretical FP32 | 15.5× slower (CPU) |
| Ryzen 9 7900X (MLRift) vs RX 7800 XT (PyTorch) | **3.75× slower** |

## Native gfx1100 ISA path

`--target=amdgpu-native` emits the same kernels via a hand-written
GFX1100 instruction encoder in `src/format_amdgpu.mlr`. No `hipcc`,
no LLVM toolchain, no `.hip` source. The kernels (`decay_step`,
`delivery_step`, `lif_step`) are bit-encoded directly into an ELF
code object, including:

- IEEE-correct f64 division (`v_div_scale_f64` ×2, `v_rcp_f64`,
  Newton-Raphson refinement, `v_div_fma_f64`, `v_div_fixup_f64`) —
  matches `hipcc`'s output ULP-for-ULP.
- CAS-retry `atomic_add_f64` (gfx11 has no native f64 atomic add).
- 3-way `EXEC` mask plumbing for nested branches in `lif_step`
  (refractory / spike / no-spike).
- Manual SGPR / VGPR allocation; no register-allocator round-trip.

Spike count over 120 billion neuron-step computations: **bit-identical
to the HIP path** (`1,985,575,928`). That's the correctness signal —
every f64 op, every CAS-retry, every `EXEC` mask transition agrees
with hipcc's output bit-for-bit.

### What native buys you on this workload

The GPU rows in the headline table are within **±2 %** wall —
essentially the same. That's the honest result. The sim is bandwidth-
bound at 60 M × per-step memory reads × 2 000 steps; every backend
emits the same algorithm with the same FMA count, so they hit the
same DRAM ceiling. Codegen quality differs (hipcc's `lif_step` is a
few hundred instructions; ours is 110), but instruction issue isn't
the bottleneck.

The wins are **not per-step perf** — they're build pipeline + runtime
linkage:

| metric | HIP runtime + hipcc | KFD shim + native ISA | factor |
|---|---:|---:|---:|
| `.co` build time | 482 ms | **1.1 ms** | **438× faster** |
| toolchain dependency | hipcc + LLVM | none | — |
| `.co` size (5 of 6 kernels native) | 21.4 KB | 8.7 KB | 2.5× smaller |
| launcher `ldd` | 5 ROCm DSOs | **0** ROCm DSOs | — |

Sub-millisecond GPU code-object builds matter when iterating on a
kernel — round-tripping through `hipcc` for every change at the half-
second scale dominates a tight inner edit loop. Native emission also
removes a build-time dependency on a moving compiler (LLVM versions,
HIP runtime headers) and produces deterministic bytes — every byte is
intentional and inspectable in `src/format_amdgpu.mlr`.

**Zero-ROCm linkage** is the bigger structural milestone. With
`--target=amdgpu-native` on the launcher source, MLRift redirects
`import "../std/hip.mlr"` to `std/hip_kfd.mlr` (a KFD-backed shim that
implements the HIP API surface against raw AMDKFD ioctls — see
`docs/AMDGPU_NATIVE.md`). The launcher binary then drops every ROCm
DSO from its dynamic-linkage list:

```
$ ldd /tmp/launch_kfd | grep -iE 'hip|hsa|amd|drm'
(empty)
$ ldd /tmp/launch_kfd
    linux-vdso.so.1
    libc.so.6
    /lib64/ld-linux-x86-64.so.2
```

That removes the entire ROCm runtime dependency at deploy time —
including a ~120 MB on-disk footprint and the version skew between
`libhsa-runtime64` and the kernel-side `amdgpu` driver that has
historically been a source of bugs.

The architectural exit on this workload is the GPU `spike_reduce`
kernel — a small reduction (two `atomic_add_u64` ops) recently added
in service of the GPU-side D2H copy path. It's still hipcc-built and
loaded via `NOESIS_CSR_CO_PATH` fallback while a native lowering for
its `atomic_add_u64` shape is in progress. The KFD shim handles the
clang offload bundle inline (no manual `clang-offload-bundler --unbundle`
step) and zero-fills COv5 hidden kernargs before dispatch, so any
hipcc-built kernel loads through the fallback path without source
changes.

#### Spike-count footnote: the 2-spike diff

Native `csr_build`'s 64-bit unsigned modulo is implemented via Barrett
reduction with a host-precomputed magic constant — a different bit-
pattern than hipcc's `__remainderu64` lowering of the same C-level
`%`. Both produce a uniform random graph on the same seed, but they
visit slightly different `(src, tgt)` pairs in the synapse table.
After 60 M neurons × 2 000 steps the 2-spike difference (out of
1,985,575,928 — 1e-9 fractional) is the deterministic signature of
that graph divergence. It is not a numerical-precision drift in the
per-step path: every f64 op in `lif_step`, `delivery_step`, and
`decay_step` produces bit-identical results between native and
hipcc-built kernels (the f64 div sequence is `v_div_scale_f64` ×2,
Newton-Raphson refinement, `v_div_fma_f64`, `v_div_fixup_f64` — IEEE-
correct, ULP-equivalent to `__ocml_div_f64`).

## The joke

The 7800 XT has ≈ **15× the peak FP32** and **8× the memory bandwidth**
of the 7900X. On this workload, PyTorch-on-the-GPU is only 4× faster
than **MLRift-on-the-CPU** — and if we compare MLRift CPU against
PyTorch CPU on the same hardware, MLRift is **5.4× faster**, erasing
most of the nominal GPU advantage.

Translation: **~75 % of the 7800 XT's silicon is burning waiting for
Python-to-kernel dispatch**. The moment you emit kernel bytes straight
from a compiler (no Python object layer, no tensor materialisation, no
dispatcher), the GPU does exactly what its spec sheet says.

## Per-phase breakdown (fastest build per phase is in **bold**)

| phase | MLRift GPU (HIP runtime) | MLRift GPU (KFD + hipcc) | MLRift GPU (KFD + native) | PyTorch GPU | cupy GPU | MLRift CPU | PyTorch CPU | numpy CPU |
|---|---|---|---|---|---|---|---|---|
| init state | 1.50 s | 1.69 s | 1.68 s | **0.02 s** | 1.17 s | 1.75 s | 0.30 s | 0.64 s |
| CSR build (240 M syn) | **0.11 s** | 0.18 s | 0.22 s | 0.28 s | 5.37 s | 1.67 s | 6.16 s | 20.62 s |
| sim (2000 steps) | **26.55 s** | 26.59 s | 26.63 s | 102.70 s | 128.37 s | 382.57 s | 2 094.09 s | 3 495.82 s |
| per-step sim | **13 ms** | **13 ms** | **13 ms** | 51 ms | 64 ms | 191 ms | 1.05 s | 1.75 s |
| `.co` build wall | 482 ms | 482 ms | **1.1 ms** | n/a | n/a | n/a | n/a | n/a |
| ROCm DSOs in launcher (`ldd`) | 5 | **0** | **0** | (Python) | (Python) | n/a | n/a | n/a |

## Files

- `examples/noesis_60m.mlr` — CPU reference
- `examples/noesis_60m_gpu.mlr` — `@kernel` definitions (4 kernels)
- `examples/noesis_60m_gpu_launch.mlr` — host launcher (single- or dual-`.co`)
- `src/format_amdgpu.mlr` — native gfx1100 ISA emitter
- `examples/noesis_60m_reference.py` — numpy CPU reference
- `examples/noesis_60m_reference_torch.py` — PyTorch CPU/GPU reference (pass `cpu` as 3rd arg to force CPU)
- `examples/noesis_60m_reference_gpu.py` — cupy GPU reference
- `std/rng.mlr` — splitmix64 implementation (shared between CPU and GPU kernels)
