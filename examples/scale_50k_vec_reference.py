#!/usr/bin/env python3
"""Reference for examples/scale_50k.mlr — 50k neurons, 5M synapses,
CSR + STP + LTD-only STDP + ring-buffer delay on E→E."""

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
A_minus = -0.006
w_max = 0.2

delay_ee = 50

N = 50000
n_E = 40000
n_steps = 1000
ref_steps = 20
connect_prob_denom = 500
buf_size = delay_ee + 1

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

rpe_ptr = [0]
rpe_tgt = []
rpe_w = []
rpe_is_ee = []
rpi_ptr = [0]
rpi_tgt = []
rpi_w = []

for i in range(N):
    if cell_type[i] == 0:
        for j in range(N):
            if i != j:
                r = rng_next()
                if (r % connect_prob_denom) == 0:
                    rpe_tgt.append(j)
                    rpe_w.append(syn_weight_init)
                    rpe_is_ee.append(1 if cell_type[j] == 0 else 0)
    else:
        for j in range(N):
            if i != j:
                r = rng_next()
                if (r % connect_prob_denom) == 0:
                    rpi_tgt.append(j)
                    rpi_w.append(syn_weight_init)
    rpe_ptr.append(len(rpe_tgt))
    rpi_ptr.append(len(rpi_tgt))

n_syn_e = len(rpe_tgt)
n_syn_i = len(rpi_tgt)
print(f"# n_syn_e={n_syn_e}")
print(f"# n_syn_i={n_syn_i}")
print(f"# n_syn={n_syn_e + n_syn_i}")

buf_delay = [0.0] * (buf_size * N)

f_e = dt / tau_e
f_i = dt / tau_i
f_pre = dt / tau_pre
f_post = dt / tau_post
f_u = dt / tau_f
f_x = dt / tau_d

def vec_decay(buf, n, f):
    i = 0
    n4 = n & ~3
    while i < n4:
        a = buf[i]; b = buf[i+1]; c = buf[i+2]; d = buf[i+3]
        a = a - a * f; b = b - b * f; c = c - c * f; d = d - d * f
        buf[i] = a; buf[i+1] = b; buf[i+2] = c; buf[i+3] = d
        i += 4
    while i < n:
        v = buf[i]
        buf[i] = v - v * f
        i += 1

def vec_relax(buf, n, target, f):
    i = 0
    n4 = n & ~3
    while i < n4:
        a = buf[i]; b = buf[i+1]; c = buf[i+2]; d = buf[i+3]
        a = a + (target - a) * f
        b = b + (target - b) * f
        c = c + (target - c) * f
        d = d + (target - d) * f
        buf[i] = a; buf[i+1] = b; buf[i+2] = c; buf[i+3] = d
        i += 4
    while i < n:
        v = buf[i]
        buf[i] = v + (target - v) * f
        i += 1

def vec_fill(buf, n, value):
    for i in range(n):
        buf[i] = value

ltd_events = 0

for step in range(n_steps):
    vec_decay(s_exc, N, f_e)
    vec_decay(s_inh, N, f_i)
    vec_decay(x_pre, N, f_pre)
    vec_decay(x_post, N, f_post)
    vec_relax(u_stp, N, U_baseline, f_u)
    vec_relax(x_stp, N, 1.0, f_x)
    vec_fill(release_factor, N, 0.0)

    read_slot = (step + 1) % buf_size
    row_base = read_slot * N
    for ri2 in range(N):
        rel = buf_delay[row_base + ri2]
        if rel > 0.0:
            start = rpe_ptr[ri2]
            end = rpe_ptr[ri2 + 1]
            for syn in range(start, end):
                tgt = rpe_tgt[syn]
                w = rpe_w[syn]
                effective = w * rel
                s_exc[tgt] = s_exc[tgt] + effective
                if rpe_is_ee[syn]:
                    xq = x_post[tgt]
                    w_new = w + A_minus * xq
                    if w_new < 0.0:
                        w_new = 0.0
                    if w_new > w_max:
                        w_new = w_max
                    rpe_w[syn] = w_new
                    ltd_events += 1
            buf_delay[row_base + ri2] = 0.0

    write_slot = step % buf_size
    w_base = write_slot * N
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
                if cell_type[i] == 0:
                    buf_delay[w_base + i] = rel

    for src in fired_list:
        if cell_type[src] == 1:
            rel = release_factor[src]
            for syn in range(rpi_ptr[src], rpi_ptr[src + 1]):
                s_inh[rpi_tgt[syn]] = s_inh[rpi_tgt[syn]] + rpi_w[syn] * rel

for oi in range(N):
    print(f"{oi} {spike_count[oi]}")

def trunc(f):
    return int(f) if f >= 0 else -int(-f)

sum_u = sum(u_stp)
sum_x = sum(x_stp)
print(f"# ltd_events={ltd_events}")
print(f"# sum_u_stp_milli={trunc(sum_u * 1000.0)}")
print(f"# sum_x_stp_milli={trunc(sum_x * 1000.0)}")
