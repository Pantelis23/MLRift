"""Python/numpy equivalent of examples/noesis_60m.mlr.

Pure-Python loops at 60M scale would take days; this uses numpy to
get a fair throughput comparison against the MLRift version.  The
algorithm and parameters are identical — sparse CSR (fixed K per
source, splitmix64 seekable rng), LIF integrate, scatter-add delivery.
No Python reference at this scale exists elsewhere because pure-Python
can't reach it.
"""
import numpy as np
import time
import sys

# ── Scale ─────────────────────────────────────────────────────────────
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

RNG_SEED  = 0xDEADBEEF12345678
GAMMA     = 0x9E3779B97F4A7C15
MASK64    = (1 << 64) - 1

# Optional command-line override for smaller test runs.
if len(sys.argv) > 1:
    N_NEURONS = int(sys.argv[1])
    N_E       = int(N_NEURONS * 0.8)
if len(sys.argv) > 2:
    N_STEPS   = int(sys.argv[2])

print(f"# N_NEURONS={N_NEURONS}", flush=True)
print(f"# N_STEPS={N_STEPS}", flush=True)

t0 = time.time()

# ── State ─────────────────────────────────────────────────────────────
V         = np.full(N_NEURONS, V_rest, dtype=np.float64)
ref_count = np.zeros(N_NEURONS, dtype=np.int64)
RI        = np.full(N_NEURONS, 50.0, dtype=np.float64)       # matches mlrc side
s_exc     = np.zeros(N_NEURONS, dtype=np.float64)
s_inh     = np.zeros(N_NEURONS, dtype=np.float64)
spike_cnt = np.zeros(N_NEURONS, dtype=np.int64)
cell_type = np.zeros(N_NEURONS, dtype=np.int64)
cell_type[N_E:] = 1

t_init = time.time()
print(f"# init_s={t_init - t0:.2f}", flush=True)

# ── CSR build (fixed K per source, splitmix64 seekable) ───────────────
# For source i, its K draws live at counters [i*K, (i+1)*K). We batch
# all N*K draws into one numpy vector.
n_syn = N_NEURONS * SYN_PER

# splitmix64 vectorized
def splitmix64_vec(ctrs):
    z = (RNG_SEED + ctrs * GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    return z

ctrs = np.arange(n_syn, dtype=np.uint64)
r = splitmix64_vec(ctrs)
col_tgt = r % N_NEURONS
# Avoid self-loops by bumping tgt if it equals source.
src_ids = np.repeat(np.arange(N_NEURONS, dtype=np.int64), SYN_PER)
self_mask = (col_tgt == src_ids)
col_tgt[self_mask] = (col_tgt[self_mask] + 1) % N_NEURONS
col_tgt = col_tgt.astype(np.int64)

# Weights: pick by (src type, tgt type).
src_is_e = (cell_type[src_ids] == 0)
tgt_is_e = (cell_type[col_tgt]  == 0)
col_w = np.empty(n_syn, dtype=np.float64)
col_w[ src_is_e &  tgt_is_e] = w_ee
col_w[ src_is_e & ~tgt_is_e] = w_ei
col_w[~src_is_e]             = w_ie
col_is_inh = (~src_is_e).astype(np.bool_)   # true for I sources

t_csr = time.time()
print(f"# csr_build_s={t_csr - t_init:.2f}", flush=True)

# Precompute decay factors
dec_e = np.float64(dt / tau_e)
dec_i = np.float64(dt / tau_i)
inv_tau_m = np.float64(dt / tau_m)

# ── Simulation loop ───────────────────────────────────────────────────
# s_exc mask for delivery-target filtering
is_inh_src = col_is_inh

for step in range(N_STEPS):
    # Decay s_exc, s_inh.
    s_exc *= (1.0 - dec_e)
    s_inh *= (1.0 - dec_i)

    # Integrate.  Three per-neuron cases:
    #   ref_count > 0  : tick down, V := V_reset
    #   else           : compute v_new; if v_new >= V_thresh → spike
    in_ref   = ref_count > 0
    drive    = RI + s_exc * J_E - s_inh * J_I
    v_new    = V + inv_tau_m * (V_rest - V + drive)
    fired    = (~in_ref) & (v_new >= V_thresh)
    # Refractory bookkeeping
    ref_count[in_ref] -= 1
    V[in_ref] = V_reset
    # Non-refractory, non-firing: V := v_new
    non_fire = (~in_ref) & (~fired)
    V[non_fire] = v_new[non_fire]
    # Fired: V := V_reset, ref_count := ref_steps, spike_cnt++
    V[fired] = V_reset
    ref_count[fired] = ref_steps
    spike_cnt[fired] += 1

    # Delivery: for each fired neuron, find its SYN_PER synapses and
    # scatter-add weight into s_exc / s_inh.
    fired_idx = np.nonzero(fired)[0]
    if fired_idx.size > 0:
        # Build per-synapse scatter list: for each fired src, K synapses.
        syn_start = fired_idx * SYN_PER
        syn_idx = (syn_start[:, None] + np.arange(SYN_PER, dtype=np.int64)[None, :]).ravel()
        tgts = col_tgt[syn_idx]
        ws   = col_w[syn_idx]
        inh  = is_inh_src[syn_idx]
        # Scatter (np.add.at is atomic-correct; np.ufunc.at).
        np.add.at(s_exc, tgts[~inh], ws[~inh])
        np.add.at(s_inh, tgts[ inh], ws[ inh])

t_sim = time.time()
print(f"# sim_s={t_sim - t_csr:.2f}", flush=True)
print(f"# sim_s_per_step={(t_sim - t_csr) / N_STEPS:.4f}", flush=True)

# ── Summary ──────────────────────────────────────────────────────────
total_spikes = int(spike_cnt.sum())
active = int((spike_cnt > 0).sum())

t_end = time.time()
print(f"# N={N_NEURONS}")
print(f"# n_syn={n_syn}")
print(f"# n_steps={N_STEPS}")
print(f"# total_spikes={total_spikes}")
print(f"# active_neurons={active}")
print(f"# total_wall_s={t_end - t0:.2f}")
