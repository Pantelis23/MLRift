"""cupy/ROCm-port of examples/noesis_60m_reference.py.

Same algorithm (fixed-K sparse CSR, splitmix64 seekable rng, LIF
integrate + atomic scatter delivery); runs on the AMD GPU via HIP.
RX 7800 XT has 16 GB VRAM which fits the 11-ish GB working set for
60 M × 4 syn comfortably.

Timings include a warm-up kernel-launch + first JIT compile which is
unavoidable on first cupy call; the reported numbers print after
cp.cuda.Stream.null.synchronize() so they are wall-accurate.
"""
import cupy as cp
import numpy as np
import time, sys, os

N_NEURONS = 60_000_000
N_E       = 48_000_000
SYN_PER   = 4
N_STEPS   = 2000

dt        = 0.1
V_rest    = -65.0
V_reset   = -70.0
V_thresh  = -50.0
tau_m     = 10.0
tau_e     = 10.0
tau_i     = 5.0
J_E       = 1.0
J_I       = 1.0
ref_steps = 20
w_ee      = 0.20
w_ei      = 0.30
w_ie      = 0.30

RNG_SEED  = cp.uint64(0xDEADBEEF12345678)
GAMMA     = cp.uint64(0x9E3779B97F4A7C15)
M1        = cp.uint64(0xBF58476D1CE4E5B9)
M2        = cp.uint64(0x94D049BB133111EB)

if len(sys.argv) > 1:
    N_NEURONS = int(sys.argv[1])
    N_E       = int(N_NEURONS * 0.8)
if len(sys.argv) > 2:
    N_STEPS   = int(sys.argv[2])

print(f"# N_NEURONS={N_NEURONS}", flush=True)
print(f"# N_STEPS={N_STEPS}", flush=True)
print(f"# device={cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)

# Warm the JIT so the first-call compile cost doesn't land in init_s.
_warm = cp.arange(16, dtype=cp.float64) + 1.0
cp.cuda.Stream.null.synchronize()
del _warm

t0 = time.time()

# ── State ─────────────────────────────────────────────────────────────
V         = cp.full(N_NEURONS, V_rest, dtype=cp.float64)
ref_count = cp.zeros(N_NEURONS, dtype=cp.int64)
# Slight per-neuron variance desynchronizes firings across steps —
# otherwise every neuron crosses threshold on the same step and the
# 60M-element fired-index temporary blows 16 GB VRAM.
RI        = 45.0 + cp.arange(N_NEURONS, dtype=cp.float64) * (20.0 / N_NEURONS)
s_exc     = cp.zeros(N_NEURONS, dtype=cp.float64)
s_inh     = cp.zeros(N_NEURONS, dtype=cp.float64)
spike_cnt = cp.zeros(N_NEURONS, dtype=cp.int64)
cell_type = cp.zeros(N_NEURONS, dtype=cp.int64)
cell_type[N_E:] = 1

cp.cuda.Stream.null.synchronize()
t_init = time.time()
print(f"# init_s={t_init - t0:.2f}", flush=True)

# ── CSR build (fixed K, splitmix64 seekable) ──────────────────────────
# VRAM-tight path: fuse ctr generation + splitmix64 + modulo into one
# kernel, skip the explicit ctrs/src_ids buffers (saves ~3.8 GB).
n_syn = N_NEURONS * SYN_PER

csr_tgt_kernel = cp.ElementwiseKernel(
    'uint64 ctr, uint64 seed, uint64 gamma, uint64 m1, uint64 m2, int64 N, int64 syn_per',
    'int64 tgt',
    '''
    unsigned long long z = seed + ctr * gamma;
    z = (z ^ (z >> 30)) * m1;
    z = (z ^ (z >> 27)) * m2;
    z = z ^ (z >> 31);
    long long src = (long long)(ctr / syn_per);
    long long t   = (long long)(z % (unsigned long long)N);
    if (t == src) { t = t + 1; if (t >= N) t = 0; }
    tgt = t;
    ''',
    'csr_tgt')

# Sequence counter is the array index; cupy supports indexing in-kernel
# via 'i' but we pass an explicit arange here for clarity. Free the
# arange buffer right after to reclaim VRAM.
ctrs = cp.arange(n_syn, dtype=cp.uint64)
col_tgt = csr_tgt_kernel(ctrs, RNG_SEED, GAMMA, M1, M2,
                          cp.int64(N_NEURONS), cp.int64(SYN_PER))
del ctrs
cp.get_default_memory_pool().free_all_blocks()

# Weight + inhibition flags. src_is_e depends on source index; we
# recompute it from (syn_index // SYN_PER) > N_E rather than keeping
# a separate src_ids buffer.
# Chunk in 40M-element slices so intermediate masks stay small.
col_w = cp.empty(n_syn, dtype=cp.float64)
col_is_inh = cp.empty(n_syn, dtype=cp.bool_)
CHUNK = 40_000_000
for off in range(0, n_syn, CHUNK):
    end = min(off + CHUNK, n_syn)
    s = cp.arange(off, end, dtype=cp.int64) // SYN_PER
    t = col_tgt[off:end]
    src_e = cell_type[s] == 0
    tgt_e = cell_type[t] == 0
    wchunk = cp.empty(end - off, dtype=cp.float64)
    wchunk[ src_e &  tgt_e] = w_ee
    wchunk[ src_e & ~tgt_e] = w_ei
    wchunk[~src_e]          = w_ie
    col_w[off:end]      = wchunk
    col_is_inh[off:end] = ~src_e
    del s, t, src_e, tgt_e, wchunk
cp.get_default_memory_pool().free_all_blocks()

cp.cuda.Stream.null.synchronize()
t_csr = time.time()
print(f"# csr_build_s={t_csr - t_init:.2f}", flush=True)

# ── Simulation loop ───────────────────────────────────────────────────
dec_e     = cp.float64(dt / tau_e)
dec_i     = cp.float64(dt / tau_i)
inv_tau_m = cp.float64(dt / tau_m)

# Index range per source (for fanning out fired neurons to their synapses).
arange_K = cp.arange(SYN_PER, dtype=cp.int64)

for step in range(N_STEPS):
    s_exc *= (1.0 - dec_e)
    s_inh *= (1.0 - dec_i)

    in_ref = ref_count > 0
    drive  = RI + s_exc * J_E - s_inh * J_I
    v_new  = V + inv_tau_m * (V_rest - V + drive)
    fired  = (~in_ref) & (v_new >= V_thresh)

    # Refractory tick / non-fire / fire state machine
    ref_count[in_ref] -= 1
    V[in_ref] = V_reset
    non_fire = (~in_ref) & (~fired)
    V[non_fire] = v_new[non_fire]
    V[fired] = V_reset
    ref_count[fired] = ref_steps
    spike_cnt[fired] += 1

    # Delivery: scatter-add weights for fired neurons into s_exc/s_inh
    fired_idx = cp.nonzero(fired)[0]
    if fired_idx.size > 0:
        syn_start = fired_idx * SYN_PER
        syn_idx = (syn_start[:, None] + arange_K[None, :]).ravel()
        tgts = col_tgt[syn_idx]
        ws   = col_w[syn_idx]
        inh  = col_is_inh[syn_idx]
        # cupyx.scatter_add for atomic-correct scatter on GPU
        e_mask = ~inh
        i_mask = inh
        cp.add.at(s_exc, tgts[e_mask], ws[e_mask])
        cp.add.at(s_inh, tgts[i_mask], ws[i_mask])

cp.cuda.Stream.null.synchronize()
t_sim = time.time()
print(f"# sim_s={t_sim - t_csr:.2f}", flush=True)
print(f"# sim_s_per_step={(t_sim - t_csr) / N_STEPS:.4f}", flush=True)

total_spikes = int(spike_cnt.sum().get())
active = int((spike_cnt > 0).sum().get())

t_end = time.time()
print(f"# N={N_NEURONS}")
print(f"# n_syn={n_syn}")
print(f"# n_steps={N_STEPS}")
print(f"# total_spikes={total_spikes}")
print(f"# active_neurons={active}")
print(f"# total_wall_s={t_end - t0:.2f}")
