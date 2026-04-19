#!/usr/bin/env python3
"""Reference for examples/sparse_1000.mlr — 1000 LIF neurons, ring
lattice fan_out=10 = 10,000 synapses. IEEE-754 f64. Output: 'i step'."""

dt = 0.1
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
tau_m = 20.0
tau_s = 10.0
J = 2.0
syn_weight = 1.0

N = 1000
fan_out = 10
n_syn = N * fan_out
n_steps = 10000
ref_steps = 20

drive_span = 20.0
drive_base = 10.0

V = [V_rest] * N
ref_count = [0] * N
s_exc = [0.0] * N
RI = [drive_base + drive_span * float(i) / float(N - 1) for i in range(N)]
fired_this_step = [0] * N

syn_src = [0] * n_syn
syn_tgt = [0] * n_syn
syn_w = [0.0] * n_syn
sidx = 0
for src in range(N):
    for off in range(1, fan_out + 1):
        syn_src[sidx] = src
        syn_tgt[sidx] = (src + off) % N
        syn_w[sidx] = syn_weight
        sidx += 1

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
        if fired_this_step[s_src] > 0:
            s_tgt = syn_tgt[syn]
            cur = s_exc[s_tgt]
            w = syn_w[syn]
            s_exc[s_tgt] = cur + w
