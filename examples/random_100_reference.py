#!/usr/bin/env python3
"""Reference for examples/random_100.mlr — 100 LIF neurons with random
E→E connectivity at p=0.1, driven by a bit-identical xorshift32 RNG.
Output begins with '# n_syn=<N>' then 'i step' per spike."""

dt = 0.1
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
tau_m = 20.0
tau_s = 10.0
J = 2.0
syn_weight = 1.0

N = 100
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

V = [V_rest] * N
ref_count = [0] * N
s_exc = [0.0] * N
RI = [10.0 + float(i) * 0.2 for i in range(N)]
fired_this_step = [0] * N

syn_src = []
syn_tgt = []
syn_w = []
for i in range(N):
    for j in range(N):
        if i != j:
            r = rng_next()
            if (r % connect_prob_denom) == 0:
                syn_src.append(i)
                syn_tgt.append(j)
                syn_w.append(syn_weight)

n_syn = len(syn_src)
print(f"# n_syn={n_syn}")

for step in range(n_steps):
    for k in range(N):
        s = s_exc[k]
        s_exc[k] = s - s * dt / tau_s
        fired_this_step[k] = 0

    for i in range(N):
        if ref_count[i] > 0:
            ref_count[i] -= 1
            V[i] = V_reset
        else:
            v = V[i]
            ri = RI[i]
            se = s_exc[i]
            drive = ri + se * J
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
            cur = s_exc[s_tgt]
            w = syn_w[syn]
            s_exc[s_tgt] = cur + w
