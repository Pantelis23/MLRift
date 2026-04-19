#!/usr/bin/env python3
"""Reference for examples/two_neuron_synapse.kr — same math, IEEE-754 f64.
Output format: 'pre <step>' / 'post <step>' lines, one per spike, in
the order the events actually happen in the integration loop."""

dt = 0.1
n_steps = 10000

V_pre_rest = -70.0
V_pre_reset = -75.0
V_pre_thresh = -55.0
tau_pre = 20.0
RI_pre = 20.0
ref_pre_steps = 20
V_pre = V_pre_rest
ref_pre = 0

V_post_rest = -70.0
V_post_reset = -75.0
V_post_thresh = -55.0
tau_post = 20.0
ref_post_steps = 20
V_post = V_post_rest
ref_post = 0

tau_s = 10.0
J = 100.0
s_exc = 0.0

for step in range(n_steps):
    s_exc = s_exc - s_exc * dt / tau_s

    pre_fired = 0
    if ref_pre > 0:
        ref_pre -= 1
        V_pre = V_pre_reset
    else:
        dv_pre = dt * (V_pre_rest - V_pre + RI_pre) / tau_pre
        V_pre = V_pre + dv_pre
        if V_pre >= V_pre_thresh:
            pre_fired = 1
            V_pre = V_pre_reset
            ref_pre = ref_pre_steps
            print(f"pre {step}")

    if pre_fired:
        s_exc = s_exc + 1.0

    if ref_post > 0:
        ref_post -= 1
        V_post = V_post_reset
    else:
        dv_post = dt * (V_post_rest - V_post + s_exc * J) / tau_post
        V_post = V_post + dv_post
        if V_post >= V_post_thresh:
            V_post = V_post_reset
            ref_post = ref_post_steps
            print(f"post {step}")
