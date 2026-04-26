#!/usr/bin/env bash
# Set the AMD GPU's DPM performance level — diagnostic helper.
#
# This is a thin wrapper around `echo MODE >
# /sys/class/drm/cardN/device/power_dpm_force_performance_level`. The
# sysfs node is root-only, so the script exists as a one-liner the
# user can invoke with sudo rather than escalating in-process.
#
# Honest status: on RDNA3 (gfx1100) neither `high` nor `profile_peak`
# closes the KFD-shim's sync-launch latency gap to HIP runtime. Both
# modes pin mclk + fclk to peak, but sclk DPM stays gated when the
# user queue is idle, so sync-launch sits at ~830-940 us regardless.
# HIP runtime hits ~157 us under profile_peak, so the path it uses
# to pin sclk is something we haven't reproduced from this side of
# the kernel boundary. See docs/AMDGPU_NATIVE.md for the full table.
#
# `profile_peak + MLRIFT_BOOST=0` deadlocks the user queue (timeout
# in hipDeviceSynchronize) — see docs/KFD_GOTCHAS.md gotcha #11.
#
# Real workloads with sustained dispatch (noesis_60m, batched gemv,
# LLM decode) already hit HIP-runtime parity without touching this
# knob. Leave the GPU in `auto` for normal use.
#
# Usage:
#   sudo scripts/mlrift-gpu-perf-mode.sh                 # → high
#   sudo scripts/mlrift-gpu-perf-mode.sh profile_peak    # explicit mode
#   sudo scripts/mlrift-gpu-perf-mode.sh auto            # restore default
#   sudo scripts/mlrift-gpu-perf-mode.sh --card=2 high   # pick a specific card
#
# Effect is non-persistent (resets to "auto" on reboot). For a permanent
# install, drop the systemd unit at the bottom of this file into
# /etc/systemd/system/mlrift-gpu-perf.service and `systemctl enable` it.

set -euo pipefail

MODE="high"
CARD=""

# Argument parse — split off any --card=N, take the first non-option as MODE.
for arg in "$@"; do
    case "$arg" in
        --card=*) CARD="${arg#--card=}" ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        auto|low|high|manual|profile_standard|profile_min_sclk|profile_min_mclk|profile_peak)
            MODE="$arg" ;;
        *)
            echo "unknown argument: $arg (try --help)" >&2
            exit 2 ;;
    esac
done

# Auto-detect the discrete AMD card if --card wasn't supplied. Picks the
# first AMD card whose hwmon directory has the dpm node (rules out APUs
# whose iGPU lives at a higher card index without DPM control).
if [[ -z "$CARD" ]]; then
    for path in /sys/class/drm/card*/device; do
        [[ -e "$path/vendor" ]] || continue
        vendor=$(cat "$path/vendor")
        [[ "$vendor" == "0x1002" ]] || continue          # AMD
        [[ -w "$path/power_dpm_force_performance_level" ]] || \
            [[ -e "$path/power_dpm_force_performance_level" ]] || continue
        # Strip /sys/class/drm/cardN/device → cardN → N
        cardname=$(basename "$(dirname "$path")")
        CARD="${cardname#card}"
        break
    done
fi

if [[ -z "$CARD" ]]; then
    echo "no AMD card with a DPM control node found under /sys/class/drm/" >&2
    exit 1
fi

NODE="/sys/class/drm/card${CARD}/device/power_dpm_force_performance_level"
SCLK="/sys/class/drm/card${CARD}/device/pp_dpm_sclk"
MCLK="/sys/class/drm/card${CARD}/device/pp_dpm_mclk"

if [[ ! -e "$NODE" ]]; then
    echo "$NODE does not exist — wrong card index?" >&2
    exit 1
fi

if [[ "$EUID" -ne 0 ]]; then
    echo "writing $NODE requires root; rerun with sudo" >&2
    exit 1
fi

prev=$(cat "$NODE")
echo "$MODE" > "$NODE"
now=$(cat "$NODE")
echo "card${CARD}: power_dpm_force_performance_level: $prev → $now"

# Surface the resulting clock states so the user can see the effect
# without having to remember the sysfs paths.
if [[ -e "$SCLK" ]]; then
    echo "  sclk:"
    sed 's/^/    /' "$SCLK"
fi
if [[ -e "$MCLK" ]]; then
    echo "  mclk:"
    sed 's/^/    /' "$MCLK"
fi

# ─────────────────────────────────────────────────────────────────────
# Optional systemd unit — paste into /etc/systemd/system/mlrift-gpu-perf.service
# and `sudo systemctl enable --now mlrift-gpu-perf.service` for a setting
# that survives reboot.
#
# [Unit]
# Description=Pin AMD GPU DPM performance level for MLRift KFD-shim
# After=multi-user.target
#
# [Service]
# Type=oneshot
# RemainAfterExit=yes
# ExecStart=/usr/local/bin/mlrift-gpu-perf-mode.sh high
# ExecStop=/usr/local/bin/mlrift-gpu-perf-mode.sh auto
#
# [Install]
# WantedBy=multi-user.target
