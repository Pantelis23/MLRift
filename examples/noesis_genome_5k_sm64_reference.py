#!/usr/bin/env python3
"""Reference for examples/noesis_genome_5k.mlr — pure-Python version of
the stage-14 kernel (LIF + STP + bidir STDP) at 5,000 neurons using
Noesis FormaGenome default parameters. Matches MLRift byte-identical.

Note: this does NOT match Noesis's actual PyTorch PlasticPartitionedSim
bit-for-bit, because Noesis quantises weights to int8 and uses tensor
reduction orders that differ from scalar summation. Matching that is a
separate task. This reference's purpose is to validate that MLRift's
compiled version of the Noesis-shaped equations is correct."""

dt = 0.1
V_rest = -65.0
V_reset = -70.0
V_thresh = -50.0
tau_m = 10.0
tau_e = 10.0
tau_i = 5.0
tau_pre = 20.0
tau_post = 20.0
tau_f = 1500.0
tau_d = 200.0
U_baseline = 0.2
w_ee_scale = 0.20
w_ei_scale = 0.30
w_ie_scale = 0.30
J_E = 1.0
J_I = 1.0
A_plus = 0.003
A_minus = -0.003
w_max = 0.5

N = 5000
n_E = 4000
n_steps = 2000
ref_steps = 20
p_ee_denom = 20
p_ei_denom = 10
p_ie_denom = 10

# splitmix64 — seekable 64-bit RNG. rng(n) depends only on n, not on
# prior draws, so parallel workers can each compute their own range in
# O(1). Matches std/rng.kr.
RNG_SEED = 0x12345678
GAMMA = 0x9E3779B97F4A7C15
MASK  = (1 << 64) - 1

def rng_at(n):
    z = (RNG_SEED + n * GAMMA) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z = (z ^ (z >> 31)) & MASK
    return z

# Monotonic draw counter — pre-CSR-build we call rng_at(draw_ctr++)
# in the same (i, j) order as the MLRift side, so the two sequences
# stay in lock-step.
_draw_ctr = 0
def rng_next():
    global _draw_ctr
    r = rng_at(_draw_ctr)
    _draw_ctr += 1
    return r

cell_type = [0 if i < n_E else 1 for i in range(N)]
V = [V_rest] * N
ref_count = [0] * N
s_exc = [0.0] * N
s_inh = [0.0] * N
x_pre = [0.0] * N
x_post = [0.0] * N
u_stp = [U_baseline] * N
x_stp = [1.0] * N
release_factor = [0.0] * N
spike_count = [0] * N
RI = [15.0 + float(i) * 0.002 for i in range(N)]

row_ptr = [0]
col_tgt = []
col_w = []
col_is_inh = []
syn_src = []

for i in range(N):
    src_e = (cell_type[i] == 0)
    for j in range(N):
        if i != j:
            tgt_e = (cell_type[j] == 0)
            denom = 0
            w = 0.0
            if src_e:
                if tgt_e:
                    denom = p_ee_denom; w = w_ee_scale
                else:
                    denom = p_ei_denom; w = w_ei_scale
            else:
                if tgt_e:
                    denom = p_ie_denom; w = w_ie_scale
            if denom > 0:
                r = rng_next()
                if (r % denom) == 0:
                    col_tgt.append(j)
                    col_w.append(w)
                    col_is_inh.append(1 if cell_type[i] == 1 else 0)
                    syn_src.append(i)
    row_ptr.append(len(col_tgt))

n_syn = len(col_tgt)

csc_ptr = [0] * (N + 1)
for p1 in range(n_syn):
    if not col_is_inh[p1]:
        tgt = col_tgt[p1]
        if cell_type[tgt] == 0:
            csc_ptr[tgt + 1] += 1
acc = 0
for pi in range(N + 1):
    c = csc_ptr[pi]
    csc_ptr[pi] = acc
    acc += c
n_ee = acc

csc_idx = [0] * n_ee
csc_fill = [0] * N
for p2 in range(n_syn):
    if not col_is_inh[p2]:
        tgt = col_tgt[p2]
        if cell_type[tgt] == 0:
            base = csc_ptr[tgt]
            off = csc_fill[tgt]
            csc_idx[base + off] = p2
            csc_fill[tgt] = off + 1

print(f"# n_syn={n_syn}")
print(f"# n_ee={n_ee}")

f_e = dt / tau_e
f_i = dt / tau_i
f_pre = dt / tau_pre
f_post = dt / tau_post
f_u = dt / tau_f
f_x = dt / tau_d

def vec_decay(buf, n, f):
    i = 0; n4 = n & ~3
    while i < n4:
        a = buf[i]; b = buf[i+1]; c = buf[i+2]; d = buf[i+3]
        a = a - a*f; b = b - b*f; c = c - c*f; d = d - d*f
        buf[i] = a; buf[i+1] = b; buf[i+2] = c; buf[i+3] = d
        i += 4
    while i < n:
        v = buf[i]; buf[i] = v - v*f; i += 1

def vec_relax(buf, n, t, f):
    i = 0; n4 = n & ~3
    while i < n4:
        a = buf[i]; b = buf[i+1]; c = buf[i+2]; d = buf[i+3]
        a = a + (t-a)*f; b = b + (t-b)*f; c = c + (t-c)*f; d = d + (t-d)*f
        buf[i] = a; buf[i+1] = b; buf[i+2] = c; buf[i+3] = d
        i += 4
    while i < n:
        v = buf[i]; buf[i] = v + (t-v)*f; i += 1

def vec_fill(buf, n, value):
    for i in range(n):
        buf[i] = value

ltp_events = 0
ltd_events = 0

for step in range(n_steps):
    vec_decay(s_exc, N, f_e)
    vec_decay(s_inh, N, f_i)
    vec_decay(x_pre, N, f_pre)
    vec_decay(x_post, N, f_post)
    vec_relax(u_stp, N, U_baseline, f_u)
    vec_relax(x_stp, N, 1.0, f_x)
    vec_fill(release_factor, N, 0.0)
    # The following three lines keep the old structure but are now no-ops
    # (decay was handled above by vec_*). Python still walks neurons for
    # voltage integration below.
    if False:
        for k in range(N):
            pass
        release_factor[k] = 0.0

    fired_list = []
    for i in range(N):
        if ref_count[i] > 0:
            ref_count[i] -= 1
            V[i] = V_reset
        else:
            v = V[i]
            ri = RI[i]
            se = s_exc[i]
            si = s_inh[i]
            drive = ri + se * J_E - si * J_I
            dv = dt * (V_rest - v + drive) / tau_m
            v_new = v + dv
            V[i] = v_new
            if v_new >= V_thresh:
                V[i] = V_reset
                ref_count[i] = ref_steps
                spike_count[i] += 1
                fired_list.append(i)
                x_pre[i] = x_pre[i] + 1.0
                x_post[i] = x_post[i] + 1.0
                uv = u_stp[i]
                xv = x_stp[i]
                rel = uv * xv
                release_factor[i] = rel
                x_stp[i] = xv - rel
                u_stp[i] = uv + U_baseline * (1.0 - uv)

    for src in fired_list:
        rel = release_factor[src]
        for syn in range(row_ptr[src], row_ptr[src + 1]):
            tgt = col_tgt[syn]
            w = col_w[syn]
            is_inh = col_is_inh[syn]
            effective = w * rel
            if is_inh:
                s_inh[tgt] = s_inh[tgt] + effective
            else:
                s_exc[tgt] = s_exc[tgt] + effective
                if cell_type[tgt] == 0:
                    xq = x_post[tgt]
                    w_new = w + A_minus * xq
                    if w_new < 0.0:
                        w_new = 0.0
                    if w_new > w_max:
                        w_new = w_max
                    col_w[syn] = w_new
                    ltd_events += 1

    for tgt in fired_list:
        if cell_type[tgt] == 0:
            for ck in range(csc_ptr[tgt], csc_ptr[tgt + 1]):
                syn_i = csc_idx[ck]
                src = syn_src[syn_i]
                w = col_w[syn_i]
                xp = x_pre[src]
                w_new = w + A_plus * xp
                if w_new < 0.0:
                    w_new = 0.0
                if w_new > w_max:
                    w_new = w_max
                col_w[syn_i] = w_new
                ltp_events += 1

spikes_E = 0
spikes_I = 0
active_E = 0
active_I = 0
for oi in range(N):
    c = spike_count[oi]
    if cell_type[oi] == 0:
        spikes_E += c
        if c > 0:
            active_E += 1
    else:
        spikes_I += c
        if c > 0:
            active_I += 1

for pi2 in range(N):
    print(f"{pi2} {spike_count[pi2]}")

def trunc(f):
    return int(f) if f >= 0 else -int(-f)

sum_w_ee = 0.0
for sc in range(n_syn):
    if not col_is_inh[sc]:
        tgt = col_tgt[sc]
        if cell_type[tgt] == 0:
            sum_w_ee += col_w[sc]
sum_u = sum(u_stp)
sum_x = sum(x_stp)

print(f"# spikes_E={spikes_E}")
print(f"# spikes_I={spikes_I}")
print(f"# active_E={active_E}")
print(f"# active_I={active_I}")
print(f"# ltp_events={ltp_events}")
print(f"# ltd_events={ltd_events}")
print(f"# sum_w_ee_ppm={trunc(sum_w_ee * 1000000.0)}")
print(f"# sum_u_stp_milli={trunc(sum_u * 1000.0)}")
print(f"# sum_x_stp_milli={trunc(sum_x * 1000.0)}")
