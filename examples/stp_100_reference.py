#!/usr/bin/env python3
"""Reference for examples/stp_100.mlr — stage 8 + Tsodyks-Markram STP."""

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
syn_weight_init = 0.1
A_plus = 0.01
A_minus = -0.012
w_max = 0.3

N = 100
n_E = 80
n_steps = 10000
ref_steps = 20
connect_prob_denom = 10

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
RI = [10.0 + float(i) * 0.2 for i in range(N)]
fired_this_step = [0] * N

syn_src = []
syn_tgt = []
syn_w = []
syn_is_inh = []
for i in range(N):
    for j in range(N):
        if i != j:
            r = rng_next()
            if (r % connect_prob_denom) == 0:
                syn_src.append(i)
                syn_tgt.append(j)
                syn_w.append(syn_weight_init)
                syn_is_inh.append(cell_type[i])

n_syn = len(syn_src)
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
        fired_this_step[k] = 0
        release_factor[k] = 0.0

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
                print(f"{i} {step}")
                V[i] = V_reset
                ref_count[i] = ref_steps
                fired_this_step[i] = 1
                x_pre[i] = x_pre[i] + 1.0
                x_post[i] = x_post[i] + 1.0
                uv = u_stp[i]
                xv = x_stp[i]
                rel = uv * xv
                release_factor[i] = rel
                x_stp[i] = xv - rel
                u_stp[i] = uv + U_baseline * (1.0 - uv)

    for syn in range(n_syn):
        src = syn_src[syn]
        tgt = syn_tgt[syn]
        w = syn_w[syn]
        is_inh = syn_is_inh[syn]

        if fired_this_step[src]:
            rel = release_factor[src]
            effective = w * rel
            if is_inh:
                s_inh[tgt] = s_inh[tgt] + effective
            else:
                s_exc[tgt] = s_exc[tgt] + effective

        if not is_inh:
            w_new = w
            if fired_this_step[src]:
                xq = x_post[tgt]
                w_new = w_new + A_minus * xq
                ltd_events += 1
            if fired_this_step[tgt]:
                xp = x_pre[src]
                w_new = w_new + A_plus * xp
                ltp_events += 1
            if w_new < 0.0:
                w_new = 0.0
            if w_new > w_max:
                w_new = w_max
            syn_w[syn] = w_new

sum_u = sum(u_stp)
sum_x = sum(x_stp)
sum_w = sum(syn_w[s] for s in range(n_syn) if not syn_is_inh[s])
n_ee = sum(1 for s in range(n_syn) if not syn_is_inh[s])

def trunc_toward_zero(f):
    return int(f) if f >= 0 else -int(-f)

print(f"# ltp_events={ltp_events}")
print(f"# ltd_events={ltd_events}")
print(f"# n_ee_synapses={n_ee}")
print(f"# sum_ee_weight_ppm={trunc_toward_zero(sum_w * 1000000.0)}")
print(f"# sum_u_stp_ppm={trunc_toward_zero(sum_u * 1000000.0)}")
print(f"# sum_x_stp_ppm={trunc_toward_zero(sum_x * 1000000.0)}")
