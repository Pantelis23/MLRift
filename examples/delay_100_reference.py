#!/usr/bin/env python3
"""Reference for examples/delay_100.mlr — two E→E pathways with ring-buffer
delays (2 ms and 10 ms), plus direct I→* CSR. Bit-identical xorshift32."""

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

delay_fast = 20
delay_slow = 100

N = 100
n_E = 80
n_steps = 10000
ref_steps = 20
buf_fast_size = delay_fast + 1
buf_slow_size = delay_slow + 1

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
buf_fast = [0.0] * (buf_fast_size * N)
buf_slow = [0.0] * (buf_slow_size * N)

rpf_ptr = [0]; rpf_tgt = []; rpf_w = []
rps_ptr = [0]; rps_tgt = []; rps_w = []
ri_ptr  = [0]; ri_tgt  = []; ri_w  = []

for i in range(N):
    if cell_type[i] == 0:
        for j in range(N):
            if i != j:
                rf = rng_next()
                if (rf % 10) == 0:
                    rpf_tgt.append(j); rpf_w.append(syn_weight)
                rs = rng_next()
                if (rs % 10) == 0:
                    rps_tgt.append(j); rps_w.append(syn_weight)
    else:
        for j in range(N):
            if i != j:
                r = rng_next()
                if (r % 10) == 0:
                    ri_tgt.append(j); ri_w.append(syn_weight)
    rpf_ptr.append(len(rpf_tgt))
    rps_ptr.append(len(rps_tgt))
    ri_ptr.append(len(ri_tgt))

print(f"# n_fast={len(rpf_tgt)}")
print(f"# n_slow={len(rps_tgt)}")
print(f"# n_inh={len(ri_tgt)}")

fast_deliveries = 0
slow_deliveries = 0

for step in range(n_steps):
    for k in range(N):
        se = s_exc[k]
        si = s_inh[k]
        s_exc[k] = se - se * dt / tau_e
        s_inh[k] = si - si * dt / tau_i

    read_fast = (step + 1) % buf_fast_size
    read_slow = (step + 1) % buf_slow_size
    row_base_f = read_fast * N
    row_base_s = read_slow * N

    for i in range(N):
        ef = buf_fast[row_base_f + i]
        if ef > 0.0:
            for syn in range(rpf_ptr[i], rpf_ptr[i + 1]):
                s_exc[rpf_tgt[syn]] += rpf_w[syn] * ef
                fast_deliveries += 1
            buf_fast[row_base_f + i] = 0.0
        es = buf_slow[row_base_s + i]
        if es > 0.0:
            for syn in range(rps_ptr[i], rps_ptr[i + 1]):
                s_exc[rps_tgt[syn]] += rps_w[syn] * es
                slow_deliveries += 1
            buf_slow[row_base_s + i] = 0.0

    fired_list = []
    for ii in range(N):
        if ref_count[ii] > 0:
            ref_count[ii] -= 1
            V[ii] = V_reset
        else:
            v = V[ii]
            ri = RI[ii]
            se = s_exc[ii]
            si = s_inh[ii]
            drive = ri + se * J_E - si * J_I
            dv = dt * (V_rest - v + drive) / tau_m
            v_new = v + dv
            V[ii] = v_new
            if v_new >= V_thresh:
                print(f"{ii} {step}")
                V[ii] = V_reset
                ref_count[ii] = ref_steps
                fired_list.append(ii)

    write_fast = step % buf_fast_size
    write_slow = step % buf_slow_size
    w_base_f = write_fast * N
    w_base_s = write_slow * N

    for src in fired_list:
        if cell_type[src] == 0:
            buf_fast[w_base_f + src] = 1.0
            buf_slow[w_base_s + src] = 1.0
        else:
            for syn in range(ri_ptr[src], ri_ptr[src + 1]):
                s_inh[ri_tgt[syn]] += ri_w[syn]

print(f"# fast_deliveries={fast_deliveries}")
print(f"# slow_deliveries={slow_deliveries}")
