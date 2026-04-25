# AMDKFD Direct-Dispatch Gotchas

Notes captured during Stage 2a (first MLRift kernel dispatched on gfx1100 via
raw AMDKFD ioctls, zero ROCm userspace). Each item below was a hard-won
discovery — none of them appear in AMD's user-facing documentation or the
`kfd_ioctl.h` comments. They came from reading the kernel driver source,
LD_PRELOAD-tracing `libhsa-runtime64`, and reading `journalctl -k` GPU-fault
logs after each hang.

If you're implementing your own KFD-direct path or porting to a new GPU
generation, start here.

**Tested target:** gfx1100 (RDNA3, Radeon 7900 XTX-class) on Linux 6.17 with
KFD v1.18. Items 1-3 and 6 may apply to other generations; items 4-5 are
gfx11-specific.

---

## 1. `ALLOC_MEMORY_OF_GPU` requires a pre-reserved host `va_addr`

The `va_addr` field is documented as "to KFD" — i.e. the user-supplied virtual
address where the allocation lives. Passing `va_addr=0` lets the kernel pick.

**Reality:** passing `va_addr=0` is silently accepted and `ALLOC` returns a
handle, but the subsequent `MAP_MEMORY_TO_GPU` then fails with `EINVAL`.

**Fix:** reserve a host-side VA range before each ALLOC, pass the returned
address as `va_addr`:

```c
void *va = mmap(NULL, size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
// Pass (uint64)va as the va_addr field of kfd_ioctl_alloc_memory_of_gpu_args.
```

The `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE` combination pins the address
range so unrelated allocations don't grab it; KFD then claims the same VA on
the GPU side via `MAP_MEMORY_TO_GPU`.

**Where:** `_kfd_reserve_va()` in `std/kfd.mlr`.

---

## 2. Host-visible `mmap` uses the DRM render-node fd, not `/dev/kfd`

The KFD docs imply that the `mmap_offset` returned by `ALLOC_MEMORY_OF_GPU`
should be passed to `mmap(/dev/kfd, ..., offset=mmap_offset)` to get a
host pointer.

**Reality:** mmapping with the KFD fd returns `ENODEV`. The mmap target is
the DRM render-node fd (`/dev/dri/renderD128`), not `/dev/kfd`.

```c
void *host = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                  drm_render_fd,        // <-- NOT kfd_fd!
                  mmap_offset);
```

This is confirmed by reading ROCT-Thunk's `queues.c` — but the kfd_ioctl.h
docstring just says "mmap a render node" without making clear which fd.

**Where:** `dev_alloc_host_visible()` in `std/kfd.mlr`.

---

## 3. AQL queue write/read pointers must be SEPARATE 4 KiB BOs

The `kfd_ioctl_create_queue_args` documentation says the kernel returns
`write_pointer_address` and `read_pointer_address` as "from KFD" out-fields.
The natural interpretation is that these are pointers into the AQL ring
allocation.

**Reality:** the kernel runs `access_ok()` on the values you pass IN as
`write_pointer_address` and `read_pointer_address`, AND requires each to be
the start of a separately KFD-allocated BO of exactly `PAGE_SIZE` (4 KiB).

You must allocate two separate 4 KiB host-visible BOs (via
`ALLOC_MEMORY_OF_GPU` + `MAP_MEMORY_TO_GPU` + mmap), pass their VAs as the
in-args, and use the same VAs at GPU-side and host-side for the wp/rp.

```c
uint64_t wp_va = dev_alloc_host_visible(4096);  // page-aligned BO
uint64_t rp_va = dev_alloc_host_visible(4096);
// Pass wp_va, rp_va as write_pointer_address, read_pointer_address.
// Host writes new write index by writing to *wp_host (= mmap'd host ptr to wp BO).
```

**Where:** queue handle layout in `std/kfd.mlr` — note offsets `[0x30] write_ptr_va`
and `[0x40] write_ptr_host` are different fields tracking the same BO.

---

## 4. EOP / context-save / control-stack sizes are EXACT, topology-derived

The kernel docs imply you can pass any reasonable size for `eop_buffer_size`,
`ctx_save_restore_size`, and `ctl_stack_size`. ROCm headers show "typical"
values like 16 KiB EOP and 96 KiB ctx-save.

**Reality:** for gfx1100 (`gfx_target_version=110001`), the kernel computes
expected sizes from the GPU topology and rejects anything else with `EINVAL`:

| Field | gfx1100 wave32 value |
|---|---|
| `eop_buffer_size` | 4096 (4 KiB) |
| `ctl_stack_size` | 0x6000 (24 KiB) |
| `cwsr_size` (ctx-save data part) | 0x1B72000 (28,778,496 ≈ 27.4 MiB) |
| Underlying ctx-save BO size | `ALIGN((cwsr_size + debug) * num_xcc, PAGE_SIZE) = 28,839,936` |

Get any of these wrong → `CREATE_QUEUE` returns `EINVAL` with no further
indication of which size was wrong.

**Where:** `KFD_EOP_BYTES`, `KFD_CTX_SAVE_BYTES`, `KFD_CTL_STACK_BYTES`
constants in `std/kfd.mlr`.

These will differ on other GPU generations; for gfx10/gfx9 read the kernel's
`kfd_topology.c` and `kfd_chardev.c` (look for the queue-size validation).

---

## 5. Ring + EOP + kernel image ALL need `EXECUTABLE` flag

The `KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE` bit is documented for the kernel
code segment. Naturally, you'd think only the `.text` allocation needs it.

**Reality:** GPU page-faults at runtime on three separate allocations if any
of them lacks the EXECUTABLE bit:

| Allocation | Why it needs EXECUTABLE |
|---|---|
| AQL ring | Command Processor Fetcher reads packets with execute permission |
| Kernel code (`.text` + `.kd`) | Shader instruction cache (SQC) requires it |
| EOP buffer | CP firmware writes end-of-pipe events with execute permission |

The faults are silent at the ioctl level — `CREATE_QUEUE` and dispatch both
return success. The only signal is in `journalctl -k`:

```
amdgpu: VM_L2_PROTECTION_FAULTS: VMID=11 PERMISSION_FAULTS=0x8 ... CLIENT=CPF
amdgpu: VM_L2_PROTECTION_FAULTS: ... CLIENT=SQC (inst)
```

The host-side observation is just a hung dispatch — the busy-poll never sees
the signal change.

**Canonical flag combos** (LD_PRELOAD-traced from `libhsa-runtime64`):

| Allocation | Flags |
|---|---|
| AQL ring | `0xD6000004` = `USERPTR \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| COHERENT \| UNCACHED` |
| EOP / kernel image | `0xD0000001` = `VRAM \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE` |

**Where:** `dev_alloc_aql_ring`, `dev_alloc_kernel_image`, `dev_alloc_vram_exec`
in `std/kfd.mlr`.

---

## 6. `KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM` is gfx7/8-only

The flag exists in `kfd_ioctl.h` and looks like the obvious choice for AQL
ring allocations. Documentation suggests it.

**Reality:** on gfx11 setting this flag halves the BO size and double-maps
the VA range to itself, creating a wraparound buffer for legacy hardware.
This causes weird "buffer is half the size you asked for" confusion.

`libhsa-runtime64` does NOT set this flag on gfx10+ — it allocates the AQL
ring as a normal USERPTR or GTT buffer with `EXECUTABLE` set (see #5).

**Fix:** drop `AQL_QUEUE_MEM` from the ring allocation flags on gfx10+.

---

## 7. `__sync_synchronize` is a GCC builtin, not a libc symbol

The HSA AQL packet release pattern says: write the body of the packet, then
do an `atomic_release` write of the header word so the GPU scheduler sees a
fully-formed packet.

The natural way to do this in plain C is `__sync_synchronize()` as a fence,
or `__atomic_store_n(..., __ATOMIC_RELEASE)`.

**Reality:** `__sync_synchronize` is a GCC compiler builtin that gets compiled
inline as an `mfence` instruction — there is no actual symbol named
`__sync_synchronize` in `libc.so.6` or anywhere else linkable. Trying to
declare `extern void __sync_synchronize(void);` and link it gives
`undefined symbol: __sync_synchronize` at runtime.

**On x86 you can skip the fence entirely.** x86 has strong store ordering:
single-core stores retire in-order, so the body stores naturally precede the
header store. Just write the body, then write the header, then write the
doorbell.

```c
// All packet body fields...
*(uint32_t *)(packet + 0x00) = (setup << 16) | header;  // header release
// Doorbell...
*(uint64_t *)(doorbell_host) = new_wp;
```

**On ARM or any weakly-ordered architecture,** you'd need a real `dmb ish`
inline asm before the header write. None of our targets need this today.

**Where:** `_kfd_fence()` in `std/kfd.mlr` is a no-op.

---

## Debug techniques that actually worked

1. **`journalctl -k | tail -30` after a hang.** The single most valuable tool
   for KFD debugging. GPU-internal page faults appear here with VMID,
   offending VA, and the client unit (CPF / SQC / TC). Neither dmesg, strace,
   nor ltrace surface KFD's complaints.

2. **LD_PRELOAD-trace libhsa-runtime64.** Build a small `.so` that intercepts
   `ioctl()` via `dlsym(RTLD_NEXT, "ioctl")`, log every KFD ioctl call with
   its args struct dumped via `xxd`, then run any HSA-using program (e.g.
   `rocm-smi`). You get the canonical ioctl sequence + flag combinations for
   free. This is how we discovered the `0xD6000004` / `0xD0000001` flag
   patterns in #5.

3. **Compare with a working reference C program at each step.** If the KFD
   layer fails, write the equivalent in raw C (with kfd_ioctl.h included),
   run that. If the C version also fails, the bug is in your understanding
   of the API. If it works, the bug is in the MLRift port.

4. **Diff the kernel source.** `linux/drivers/gpu/drm/amd/amdkfd/`. The
   `kfd_chardev.c` is the ioctl entry point; follow the `validate_*`
   functions for size/alignment checks; follow `kfd_queue_buffer_get` for
   wp/rp BO requirements (#3 came from there).
