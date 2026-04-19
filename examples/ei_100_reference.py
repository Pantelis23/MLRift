#!/usr/bin/env python3
"""Reference for examples/ei_100.mlr — 80 E / 20 I LIF network, random
p=0.1 connectivity via xorshift32, two conductances with distinct taus,
E/I-balanced drive scale. IEEE-754 f64."""

dt = 0.1
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
tau_m = 20.0
tau_e = 10.0
tau_i = 5.0
J_E = 2.0
J_I = 8.0
syn_weight = 0.1

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
                syn_w.append(syn_weight)
                syn_is_inh.append(cell_type[i])

n_syn = len(syn_src)
n_i_syn = sum(1 for f in syn_is_inh if f)
print(f"# n_syn={n_syn} n_i={n_i_syn}")

for step in range(n_steps):
    for k in range(N):
        se = s_exc[k]
        si = s_inh[k]
        s_exc[k] = se - se * dt / tau_e
        s_inh[k] = si - si * dt / tau_i
        fired_this_step[k] = 0

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

    for syn in range(n_syn):
        s_src = syn_src[syn]
        if fired_this_step[s_src]:
            s_tgt = syn_tgt[syn]
            w = syn_w[syn]
            if syn_is_inh[syn]:
                s_inh[s_tgt] = s_inh[s_tgt] + w
            else:
                s_exc[s_tgt] = s_exc[s_tgt] + w
