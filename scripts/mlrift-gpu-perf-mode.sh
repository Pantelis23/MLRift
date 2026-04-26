#!/usr/bin/env bash
# Set the AMD GPU's DPM performance level for the KFD-shim path.
#
# Why: the KFD-shim's two-tier boost (compute + SDMA, see std/hip_kfd.mlr)
# pins sclk and ramps mclk, but the firmware DPM controller still drops
# clocks during the brief idle window between a hipDeviceSynchronize and
# the next hipModuleLaunchKernel. For sync-launch micro-benchmarks (one
# launch, then sync, then repeat) that ramp-up cost dominates wall time.
# `power_dpm_force_performance_level=high` removes the ramp by pinning
# the GPU at its highest DPM state continuously. The sysfs node is root-
# only, so we ship this script instead of doing it in-process.
#
# Real workloads with sustained dispatch (noesis_60m, batched gemv, LLM
# decode) already hit HIP-runtime parity without this — the firmware
# never gets a chance to drop clocks while the queue stays non-empty.
# Run this only if you care about sync-launch latency.
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
