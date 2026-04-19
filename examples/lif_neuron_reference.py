#!/usr/bin/env python3
"""
Reference implementation of examples/lif_neuron.kr — same parameters,
same update rule, IEEE-754 f64 (Python's native float). Prints the
integer step index at every spike. Compared byte-for-byte against the
mlrc-compiled binary's output by examples/lif_compare.sh.

If the two diverge, one of the following happened:
  - MLRift codegen lost float precision somewhere
  - the two implementations drifted (update rule, refractory semantics)
  - the integration order doesn't match (see note on `ref_count > 0`
    check being before the voltage update in both)
"""

dt = 0.1
tau_m = 20.0
V_rest = -70.0
V_reset = -75.0
V_thresh = -55.0
RI = 20.0
n_steps = 10000
ref_steps = 20

V = V_rest
ref_count = 0
for step in range(n_steps):
    if ref_count > 0:
        ref_count -= 1
        V = V_reset
    else:
        dv = dt * (V_rest - V + RI) / tau_m
        V = V + dv
        if V >= V_thresh:
            print(step)
            V = V_reset
            ref_count = ref_steps
