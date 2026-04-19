#!/usr/bin/env python3
"""Reference for examples/population_100.kr — 100 uncoupled LIF neurons,
RI[i] = 10.0 + 0.2 * i, same update order (outer: step, inner: neuron),
IEEE-754 f64 throughout. Output format: 'i step\\n' per spike."""

dt = 0.1
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
tau_m = 20.0
ref_steps = 20
N = 100
n_steps = 10000

V = [V_rest] * N
ref_count = [0] * N
RI = [10.0 + float(i) * 0.2 for i in range(N)]

for step in range(n_steps):
    for i in range(N):
        if ref_count[i] > 0:
            ref_count[i] -= 1
            V[i] = V_reset
        else:
            v = V[i]
            ri = RI[i]
            dv = dt * (V_rest - v + ri) / tau_m
            v_new = v + dv
            V[i] = v_new
            if v_new >= V_thresh:
                print(f"{i} {step}")
                V[i] = V_reset
                ref_count[i] = ref_steps
