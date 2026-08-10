#!/usr/bin/env bash
# Give one torchrun worker exactly one physical GPU, exposed inside that
# process as logical cuda:0.  The production v3.3 TorchScript teacher embeds
# literal cuda:0 device constants in its RoPE graph, so map_location alone
# cannot make the trace safe on ordinary LOCAL_RANK=1..N workers.
set -euo pipefail

fail() {
    echo "[error] isolated CUDA rank setup failed: $*" >&2
    exit 69
}

original_local_rank="${LOCAL_RANK:-}"
[[ "$original_local_rank" =~ ^[0-9]+$ ]] \
    || fail "LOCAL_RANK must be a non-negative integer, got '${original_local_rank:-unset}'"

trim_space() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

selected_device=""
original_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$original_visible_devices" ]]; then
    IFS=',' read -r -a visible_devices <<<"$original_visible_devices"
    (( original_local_rank < ${#visible_devices[@]} )) \
        || fail "LOCAL_RANK=$original_local_rank exceeds CUDA_VISIBLE_DEVICES='$original_visible_devices'"

    declare -A observed_devices=()
    for index in "${!visible_devices[@]}"; do
        token="$(trim_space "${visible_devices[$index]}")"
        [[ -n "$token" && "$token" != "-1" ]] \
            || fail "CUDA_VISIBLE_DEVICES contains an empty/disabled entry: '$original_visible_devices'"
        [[ -z "${observed_devices[$token]+x}" ]] \
            || fail "CUDA_VISIBLE_DEVICES contains duplicate device '$token'"
        observed_devices[$token]=1
        visible_devices[$index]="$token"
    done
    selected_device="${visible_devices[$original_local_rank]}"
else
    # When the container runtime already limits the device files, CUDA's
    # visible indices are the dense range 0..N-1.
    selected_device="$original_local_rank"
fi

[[ $# -gt 0 ]] || fail "missing worker command"

export CUDA_VISIBLE_DEVICES="$selected_device"
export D2R_PHYSICAL_LOCAL_RANK="$original_local_rank"
export D2R_LAUNCH_CUDA_VISIBLE_DEVICES="$original_visible_devices"
export D2R_LAUNCH_LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-}"
# Preserve global RANK/WORLD_SIZE from torchrun, but make the only GPU in this
# worker the device selected by train.py and by the hard-coded teacher graph.
# LOCAL_WORLD_SIZE remains the number of workers on this node; narrowing each
# worker's CUDA mask does not turn an N-rank node into N one-rank nodes.
export LOCAL_RANK=0

echo "[info] global_rank=${RANK:-unknown} physical_local_rank=$D2R_PHYSICAL_LOCAL_RANK device='$selected_device' -> logical cuda:0" >&2
exec "$@"
