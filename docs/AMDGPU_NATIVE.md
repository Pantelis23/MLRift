# `--target=amdgpu-native` — ROCm-free GPU launchers via raw AMDKFD

## What it does

`--target=amdgpu-native` makes MLRift compile a launcher that dispatches
GPU kernels through **raw AMDKFD ioctls** instead of HIP. The launcher
source itself is unchanged: same `import "../std/hip.mlr"`, same
`hipMalloc` / `hipMemcpy` / `hipModuleLoadData` / `hipModuleGetFunction`
/ `hipModuleLaunchKernel` / `hipDeviceSynchronize` / `hipFree` calls.
Under this flag the binary's `ldd` shows only `libc + ld-linux + vdso`
— no `libamdhip64`, `libhsa-runtime64`, `libdrm`, `libdrm_amdgpu`.

Default behavior (no flag) is unchanged: the launcher links the ROCm
HIP runtime as before.

```
# Default — uses HIP, links libamdhip64.
./build/mlrc --arch=x86_64 examples/m1o_atomic_load.mlr -o /tmp/m1o_hip
ldd /tmp/m1o_hip | grep hip          # → libamdhip64.so.7

# amdgpu-native — uses KFD, no ROCm linkage.
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/m1o_atomic_load.mlr -o /tmp/m1o_kfd
ldd /tmp/m1o_kfd | grep -iE 'hip|hsa|amd|drm'   # → (empty)
/tmp/m1o_kfd                                     # → "M1o gate: GREEN"
```

## How it works

Three pieces:

### 1. Import redirect (`src/main.mlr`)

`import_resolve` and the stdlib lookup branch run every resolved path
through `_maybe_redirect_hip_to_kfd`. When `target_amdgpu_native != 0`
and the path ends in `hip.mlr` (and not already `hip_kfd.mlr`), the
suffix is rewritten in place. The launcher `import "../std/hip.mlr"`
ends up loading `std/hip_kfd.mlr` instead.

### 2. KFD-backed shim (`std/hip_kfd.mlr`)

The shim provides concrete MLR `fn` implementations for the HIP API
surface used by Stage-1 launchers. They delegate to `std/kfd.mlr`
primitives:

| HIP call                      | KFD shim implementation                     |
|-------------------------------|---------------------------------------------|
| `hipSetDevice(0)`             | no-op (returns 0)                           |
| `hipGetDeviceCount(*ct)`      | writes 1                                    |
| `hipMalloc(*slot, n)`         | `dev_alloc_host_visible(n)` → `*slot`       |
| `hipFree(p)`                  | `dev_free(p)`                               |
| `hipMemcpy(d, s, n, H2D=1)`   | `dev_copy_h2d`                              |
| `hipMemcpy(d, s, n, D2H=2)`   | `dev_copy_d2h`                              |
| `hipMemcpy(d, s, n, D2D=3)`   | host-view-to-host-view memcpy               |
| `hipMemcpy(d, s, n, H2H/4)`   | plain `memcpy`                              |
| `hipModuleLoadData(*m, blob)` | `kernel_load_from_blob(blob, 0)` → `*m`     |
| `hipModuleGetFunction(*f, m, name)` | wrapper `{kd_va, kernarg_size}` → `*f`|
| `hipModuleLaunchKernel(...)`  | unmarshal ptr_table → kernarg blob, dispatch + busy-poll wait |
| `hipDeviceSynchronize()`      | no-op (`hipModuleLaunchKernel` is synchronous) |

A single AQL queue + signal + 4 KiB scratch kernarg buffer are
lazy-initialised on first call and reused across every launch.

`hipModuleLaunchKernel` is the only non-trivial translation. HIP's
`kernel_params` is `void**` — an array of pointers, each pointing to
an 8-byte slot holding one argument value. The shim:

1. Reads `kernarg_size` from the AMDHSA kernel descriptor at offset
   `0x08` (set by `amd_emit_kernel_descriptor` in
   `src/format_amdgpu.mlr`). `kernel_kernarg_size` in `std/kfd.mlr`
   does the lookup.
2. Iterates `kernarg_size / 8` argument slots, dereferences each
   ptr_table entry, and concatenates the values into a contiguous KFD
   kernarg blob — the layout the GPU's `s_load_dword` instructions
   actually expect.
3. Calls `kfd_dispatch` and `kfd_wait` synchronously.

Step 2 assumes every kernarg is 8 bytes. Every MLRift `@kernel` emitter
satisfies this (`args_desc` entries declare `size=8` for both
`GLOBAL_BUFFER` pointer args and `BY_VALUE` `uint64` / `f64` args).

### 3. Duplicate `@dynamic extern` deduplication (`src/analysis.mlr`)

Stage-1 HIP launchers locally declare `@dynamic extern fn puts(...)`
and `@dynamic extern fn exit(...)`. `std/kfd.mlr` (transitively imported
through the shim) declares the same. With strict redefinition checking
this collides. The semantic analyser now silently dedupes when **both**
declarations have flag bits 3 + 5 set (`extern` + `@dynamic`) — they
name the same `dlopen` symbol, so dropping one is correct.

## Supported launchers

Verified GREEN under `--target=amdgpu-native` with zero ROCm linkage:

| Launcher                                | Kernel emit flag                          |
|-----------------------------------------|-------------------------------------------|
| `m1o_atomic_load.mlr`                   | `--emit-amdgpu-atomic-demo=`              |
| `m1p_f64atomic_load.mlr`                | `--emit-amdgpu-f64-atomic-add=`           |
| `m1q_csr_scatter_load.mlr`              | `--emit-amdgpu-csr-scatter-demo=`         |
| `m1r_csr_i_deliver_load.mlr`            | `--emit-amdgpu-csr-i-deliver=`            |
| `m1s_csr_e_deliver_load.mlr`            | `--emit-amdgpu-csr-e-deliver=`            |
| `m1t_lif_integrate_load.mlr`            | `--emit-amdgpu-lif-integrate=`            |
| `m1t2_lif_spike_load.mlr`               | `--emit-amdgpu-lif-spike=`                |
| `m1t3_lif_stp_load.mlr`                 | `--emit-amdgpu-lif-stp=`                  |
| `m1t4_lif_full_load.mlr` / `b1_lif_full_loop.mlr` | `--emit-amdgpu-lif-full=`       |
| `m1u_mt_lif_integrate_load.mlr`         | `--emit-amdgpu-mt-lif-integrate=`         |
| `m1u2_mt_lif_full_load.mlr`             | `--emit-amdgpu-mt-lif-full=`              |
| `m1v_mt_csr_delivery_load.mlr`          | `--emit-amdgpu-mt-csr-i-deliver=`, `--emit-amdgpu-mt-csr-e-delayed=` |
| `b4_decay_relax_fill_load.mlr`          | `--emit-amdgpu-decay-step=`, `--emit-amdgpu-relax-step=`, `--emit-amdgpu-fill-step=` |
| `b4_native_stage13_load.mlr`            | all six Stage-13 emit flags               |

Standard build cycle:

```
./build/mlrc --emit-amdgpu-<flag>=/tmp/<name>.co
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/<launcher>.mlr -o /tmp/<launcher>
/tmp/<launcher>
```

`--target=amdgpu-native` on a launcher source with no `@kernel`
functions does NOT error any more — for these the redirect at import
time is what makes the flag mean something; no `.co` is emitted from
the launcher itself (the user is expected to emit kernels separately
via `--emit-amdgpu-<flag>=`).

## Semantics differences

* **`hipModuleLaunchKernel` is synchronous** under the shim. HIP
  proper is async + you sync via `hipDeviceSynchronize`; here the
  busy-poll wait runs inside the launch call and `hipDeviceSynchronize`
  is a no-op. This matches the b5b "single queue, wait between every
  dispatch" pattern. Launchers that rely on enqueueing many launches
  before syncing will run at HIP-async cadence under HIP and at
  serialised cadence under amdgpu-native — same final result, slower.

* **Graph API not implemented** in the shim. `hipGraphCreate`,
  `hipGraphLaunch`, etc. would need a graph→AQL-batch translator.
  Three launchers use it — `gpu_graph_smoke.mlr`,
  `gpu_stage13_graph.mlr`, and the noesis 60M graph variants — they
  fall through to undefined-symbol errors at compile time. The non-graph
  Stage-13 launcher (`b4_native_stage13_load.mlr`) works.

* **Buffer alignment is strict**. HIP allocates from pooled memory and
  silently absorbs out-of-bounds writes within the pool. KFD's
  `dev_alloc_host_visible` uses one mmap per allocation, so OOB writes
  page-fault. This caught a real bug in
  `b4_native_stage13_load.mlr` — see "Stage-13 kernel ABI gotcha"
  below.

## Gotchas

### #9: AQL `grid_size_*` is total work items, not num_workgroups

HIP/CUDA `gridDim` means num_workgroups; HSA AQL `grid_size` means
num_workgroups × workgroup_size. With `grid=1, block=1` the bug is
invisible (1×1 == 1), but with `block_x = N` the CP launches at most 1
lane per dispatch and the kernel does nothing useful (no fault, just
zero output). `kfd_dispatch` in `std/kfd.mlr` takes HIP-style params
(`grid = num_workgroups`) and multiplies internally for the AQL field —
the shim inherits this so HIP launchers work unchanged.

### #10: MLRift Stage-13 kernel ABI — `w_base` is slot, `row_base` is entry

`mt_lif_full` does `bd + (w_base * N + tid) * 8` (`s_lshl_b64` by 7
for `N=16`), so `w_base` must be a slot index `0..BUF_SIZE-1`.
`mt_csr_e_delayed` does `bd + (row_base + src) * 8` (`s_lshl_b64` by
3), so `row_base` must be `read_slot * N` (an entry index).

The original `b4_native_stage13_load.mlr` passed `write_slot * N` to
**both** kernels and was "accidentally GREEN" under HIP because:
- step 0's writes happened to be correct (`w_base = 0` makes both
  formulas coincide), and
- HIP's pooled allocations silently absorbed the out-of-bounds writes
  from steps 1..3.

KFD's per-buffer page reservations don't, so under
`--target=amdgpu-native` the launcher page-faulted on step 1's LIF
dispatch. The current `b4_native_stage13_load.mlr` passes
`write_slot` (slot index) to LIF — works under both HIP and KFD.
`b5b_kfd_stage13_load.mlr` documents the same fix.

For the full set of KFD-runtime gotchas (queue MQD layout, doorbell
off-by-one, EOP / ctx-save sizes, etc.) see `docs/KFD_GOTCHAS.md`.

## File map

* `std/hip_kfd.mlr` — the shim (~150 lines).
* `std/kfd.mlr` — KFD library: `kfd_init`, `dev_alloc_*`,
  `kernel_load` / `kernel_load_from_blob` / `kernel_kernarg_size`,
  `kfd_create_queue`, `kfd_dispatch`, `kfd_wait`,
  `kfd_signal_alloc`/`reset`/`value`, `dev_copy_h2d`/`dev_copy_d2h`/
  `dev_free`, `dev_host_ptr`.
* `std/kfd_raw.mlr` — bare AMDKFD ioctl wrappers.
* `src/main.mlr` — `_maybe_redirect_hip_to_kfd`, the
  `--target=amdgpu-native` flag handler.
* `src/analysis.mlr` — duplicate `@dynamic extern` dedup.
* `examples/b5a_*.mlr`, `examples/b5b_*.mlr`, `examples/b5c_*.mlr` —
  pre-shim launchers that import `std/kfd.mlr` directly. Useful as
  references for the KFD library API.

## Testing

```
make build && make bootstrap        # rebuild compiler + verify fixed point
make test                            # 439/439

# Per-launcher gate (under shim):
./build/mlrc --emit-amdgpu-atomic-demo=/tmp/m1o.co
./build/mlrc --target=amdgpu-native --arch=x86_64 \
    examples/m1o_atomic_load.mlr -o /tmp/m1o
/tmp/m1o
ldd /tmp/m1o | grep -iE 'hsa|hip|amd|drm'   # → (empty)
```
