#!/usr/bin/env python3
"""Reference for examples/scale_10k.mlr — 10k neurons, ~1M synapses,
CSR delivery, stage-9 kernel with LTD-only STDP (LTP-on-post omitted
at scale in both versions; noted in commit). IEEE-754 f64."""

dt = 0.1
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
tau_m = 20.0
tau_e = 10.0
tau_i = 5.0
tau_pre = 20.0
tau_post = 20.0
tau_f = 1000.0
tau_d = 200.0
U_baseline = 0.2
J_E = 2.0
J_I = 8.0
syn_weight_init = 0.05
A_plus = 0.005
A_minus = -0.006
w_max = 0.2

N = 10000
n_E = 8000
n_steps = 1000
ref_steps = 20
connect_prob_denom = 100

drive_span = 20.0
drive_base = 10.0

rng_state = 0x12345678

def rng_next():
    global rng_state
    s = rng_state
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= (s >> 17)
    s ^= (s << 5) & 0xFFFFFFFF
    s &= 0xFFFFFFFF
    rng_state = s
    return s

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
RI = [drive_base + drive_span * float(i) / float(N - 1) for i in range(N)]

row_ptr = [0]
col_tgt = []
col_w = []
col_is_inh = []

for i in range(N):
    for j in range(N):
        if i != j:
            r = rng_next()
            if (r % connect_prob_denom) == 0:
                col_tgt.append(j)
                col_w.append(syn_weight_init)
                col_is_inh.append(cell_type[i])
    row_ptr.append(len(col_tgt))

n_syn = len(col_tgt)
print(f"# n_syn={n_syn}")

ltp_events = 0
ltd_events = 0

for step in range(n_steps):
    for k in range(N):
        se = s_exc[k]
        si = s_inh[k]
        xp = x_pre[k]
        xq = x_post[k]
        uv = u_stp[k]
        xv = x_stp[k]
        s_exc[k] = se - se * dt / tau_e
        s_inh[k] = si - si * dt / tau_i
        x_pre[k] = xp - xp * dt / tau_pre
        x_post[k] = xq - xq * dt / tau_post
        u_stp[k] = uv + (U_baseline - uv) * dt / tau_f
        x_stp[k] = xv + (1.0 - xv) * dt / tau_d
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
        start = row_ptr[src]
        end = row_ptr[src + 1]
        for syn in range(start, end):
            tgt = col_tgt[syn]
            w = col_w[syn]
            is_inh = col_is_inh[syn]
            effective = w * rel
            if is_inh:
                s_inh[tgt] = s_inh[tgt] + effective
            else:
                s_exc[tgt] = s_exc[tgt] + effective
                xq = x_post[tgt]
                w_new = w + A_minus * xq
                if w_new < 0.0:
                    w_new = 0.0
                if w_new > w_max:
                    w_new = w_max
                col_w[syn] = w_new
                ltd_events += 1

    # LTP path intentionally omitted at this stage (reverse-index
    # requirement to dispatch by post-fire). See commit.

for oi in range(N):
    print(f"{oi} {spike_count[oi]}")

def trunc(f):
    return int(f) if f >= 0 else -int(-f)

sum_u = sum(u_stp)
sum_x = sum(x_stp)
print(f"# ltp_events={ltp_events}")
print(f"# ltd_events={ltd_events}")
print(f"# sum_u_stp_milli={trunc(sum_u * 1000.0)}")
print(f"# sum_x_stp_milli={trunc(sum_x * 1000.0)}")
