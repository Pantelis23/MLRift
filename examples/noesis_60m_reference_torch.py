"""PyTorch/ROCm port of noesis_60m_reference.py.

Same math as the numpy and cupy references (splitmix64 seekable rng,
fixed K per source, LIF integrate, atomic scatter delivery). Runs on
the AMD GPU via PyTorch's HIP backend. Requires torch-rocm installed
in a venv (see venv/).
"""
import torch
import time
import sys

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
M1        = 0xBF58476D1CE4E5B9
M2        = 0x94D049BB133111EB
MASK      = 0xFFFFFFFFFFFFFFFF

if len(sys.argv) > 1:
    N_NEURONS = int(sys.argv[1])
    N_E       = int(N_NEURONS * 0.8)
if len(sys.argv) > 2:
    N_STEPS   = int(sys.argv[2])

# Device selection. Pass "cpu" as 3rd arg to force the CPU backend
# (multi-threaded via intraop OMP pool — the fair Python-CPU baseline
# vs numpy's single-core loop).
force_cpu = len(sys.argv) > 3 and sys.argv[3] == "cpu"
if (not force_cpu) and torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print(f"# device={torch.cuda.get_device_name(0)}", flush=True)
else:
    dev = torch.device("cpu")
    # Let PyTorch use every available hardware thread.
    import os
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    torch.set_num_interop_threads(4)
    print(f"# device=cpu (torch intraop threads={torch.get_num_threads()})", flush=True)

print(f"# N_NEURONS={N_NEURONS}", flush=True)
print(f"# N_STEPS={N_STEPS}", flush=True)

# Warm so first-call compile cost doesn't land in init.
_w = torch.arange(16, dtype=torch.float64, device=dev) + 1.0
torch.cuda.synchronize() if dev.type == "cuda" else None
del _w

t0 = time.time()

# ── State ─────────────────────────────────────────────────────────────
V         = torch.full((N_NEURONS,), V_rest, dtype=torch.float64, device=dev)
ref_count = torch.zeros(N_NEURONS, dtype=torch.int64, device=dev)
RI        = 45.0 + torch.arange(N_NEURONS, dtype=torch.float64, device=dev) * (20.0 / N_NEURONS)
s_exc     = torch.zeros(N_NEURONS, dtype=torch.float64, device=dev)
s_inh     = torch.zeros(N_NEURONS, dtype=torch.float64, device=dev)
spike_cnt = torch.zeros(N_NEURONS, dtype=torch.int64, device=dev)
cell_type = torch.zeros(N_NEURONS, dtype=torch.int64, device=dev)
cell_type[N_E:] = 1

if dev.type == "cuda":
    torch.cuda.synchronize()
t_init = time.time()
print(f"# init_s={t_init - t0:.2f}", flush=True)

# ── CSR build (splitmix64 vectorized, VRAM-tight) ─────────────────────
n_syn = N_NEURONS * SYN_PER
# Torch's uint64 supports multiplication but the ops are limited. Do
# the math in int64 (2's-complement wraps modulo 2^64 naturally).
# Direct fused kernel: one big vector op per stage, intermediates freed.
ctrs = torch.arange(n_syn, dtype=torch.int64, device=dev)  # used as uint64 by wrap
seed = torch.tensor(RNG_SEED - (1 << 63), dtype=torch.int64, device=dev)  # wrap to signed
GAMMA_t = torch.tensor(GAMMA - (1 << 63), dtype=torch.int64, device=dev)
M1_t    = torch.tensor(M1    - (1 << 63), dtype=torch.int64, device=dev)
M2_t    = torch.tensor(M2    - (1 << 63), dtype=torch.int64, device=dev)

# Compute splitmix64 on uint64 via bitwise_* ops on int64.
z = (seed + ctrs * GAMMA_t)           # mod 2^64 by int64 wrap
z = torch.bitwise_xor(z, torch.bitwise_right_shift(z.to(torch.uint64).to(torch.int64), 30)) * M1_t
z = torch.bitwise_xor(z, torch.bitwise_right_shift(z.to(torch.uint64).to(torch.int64), 27)) * M2_t
z = torch.bitwise_xor(z, torch.bitwise_right_shift(z.to(torch.uint64).to(torch.int64), 31))

# tgt = z mod N.  PyTorch's uint64 remainder isn't implemented on HIP,
# and int64 `%` with negative z gives negative output. N_NEURONS < 2^32,
# so taking the low 63 bits (bitmask ensures non-negative int64) and
# modding by N_NEURONS is both fast and uniform enough.
LOW63 = (1 << 63) - 1
col_tgt = (z & LOW63) % N_NEURONS
del z, ctrs

# Avoid self-loops
src_ids = torch.arange(N_NEURONS, dtype=torch.int64, device=dev).repeat_interleave(SYN_PER)
self_mask = col_tgt == src_ids
col_tgt = torch.where(self_mask, (col_tgt + 1) % N_NEURONS, col_tgt)

# Weights
src_is_e = cell_type[src_ids] == 0
tgt_is_e = cell_type[col_tgt] == 0
col_w = torch.empty(n_syn, dtype=torch.float64, device=dev)
col_w[ src_is_e &  tgt_is_e] = w_ee
col_w[ src_is_e & ~tgt_is_e] = w_ei
col_w[~src_is_e]             = w_ie
col_is_inh = ~src_is_e
del src_is_e, tgt_is_e, self_mask, src_ids

if dev.type == "cuda":
    torch.cuda.synchronize()
t_csr = time.time()
print(f"# csr_build_s={t_csr - t_init:.2f}", flush=True)

# ── Sim loop ──────────────────────────────────────────────────────────
dec_e     = dt / tau_e
dec_i     = dt / tau_i
inv_tau_m = dt / tau_m

arange_K = torch.arange(SYN_PER, dtype=torch.int64, device=dev)

for step in range(N_STEPS):
    s_exc.mul_(1.0 - dec_e)
    s_inh.mul_(1.0 - dec_i)

    in_ref = ref_count > 0
    drive  = RI + s_exc * J_E - s_inh * J_I
    v_new  = V + inv_tau_m * (V_rest - V + drive)
    fired  = (~in_ref) & (v_new >= V_thresh)

    ref_count[in_ref] -= 1
    V[in_ref] = V_reset
    non_fire = (~in_ref) & (~fired)
    V[non_fire] = v_new[non_fire]
    V[fired] = V_reset
    ref_count[fired] = ref_steps
    spike_cnt[fired] += 1

    fired_idx = fired.nonzero(as_tuple=True)[0]
    if fired_idx.numel() > 0:
        syn_start = fired_idx * SYN_PER
        syn_idx = (syn_start.unsqueeze(1) + arange_K.unsqueeze(0)).reshape(-1)
        tgts = col_tgt[syn_idx]
        ws   = col_w[syn_idx]
        inh  = col_is_inh[syn_idx]
        e_mask = ~inh
        i_mask =  inh
        # index_add_ is atomic on GPU and supports accumulation.
        s_exc.index_add_(0, tgts[e_mask], ws[e_mask])
        s_inh.index_add_(0, tgts[i_mask], ws[i_mask])

if dev.type == "cuda":
    torch.cuda.synchronize()
t_sim = time.time()
print(f"# sim_s={t_sim - t_csr:.2f}", flush=True)
print(f"# sim_s_per_step={(t_sim - t_csr) / N_STEPS:.4f}", flush=True)

total_spikes = int(spike_cnt.sum().item())
active = int((spike_cnt > 0).sum().item())
t_end = time.time()

print(f"# N={N_NEURONS}")
print(f"# n_syn={n_syn}")
print(f"# n_steps={N_STEPS}")
print(f"# total_spikes={total_spikes}")
print(f"# active_neurons={active}")
print(f"# total_wall_s={t_end - t0:.2f}")
