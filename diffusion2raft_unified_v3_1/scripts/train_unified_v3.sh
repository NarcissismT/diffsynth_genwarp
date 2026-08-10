#!/usr/bin/env bash
# =============================================================================
# Unified v3.1 safe/resumable training entrypoint.
#
# Default resume order: latest.pt, highest epoch_*.pt, best.pt, then v2 epoch 8.
# It validates the checkpoint and prevents concurrent checkpoint writers.
# Use run_unified_v31_background.sh for detached execution and persistent logs.
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-configs/unified.yaml}"
STAGE="${STAGE:-unified}"
SEED_RESUME="${SEED_RESUME:-runs/d2r/unified/resume_from_v2_epoch8.pt}"
SEED_RESUME_REASON="${SEED_RESUME_REASON:-v2 seed}"
EPOCHS="${EPOCHS:-20}"  # total epochs, not extra epochs
MASTER_PORT="${MASTER_PORT:-29533}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_1}"
BEST_METRIC="${BEST_METRIC:-line_epe}"
LAUNCH_LABEL="${LAUNCH_LABEL:-v3.1}"
MIN_RESUME_COMPLETED_EPOCHS="${MIN_RESUME_COMPLETED_EPOCHS:-0}"
ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"
ISOLATE_CUDA_PER_RANK="${ISOLATE_CUDA_PER_RANK:-0}"
REQUIRE_TEACHER_CAPACITY_EVIDENCE="${REQUIRE_TEACHER_CAPACITY_EVIDENCE:-0}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

PYTHON="${PYTHON:-/usr/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "[error] 找不到可执行 Python；请在 diffsynth:v2-diffusers 容器内运行。" >&2
    exit 70
fi
if [[ "$REQUIRE_TEACHER_CAPACITY_EVIDENCE" != "0" && "$REQUIRE_TEACHER_CAPACITY_EVIDENCE" != "1" ]]; then
    echo "[error] REQUIRE_TEACHER_CAPACITY_EVIDENCE 只能是 0 或 1，当前值：$REQUIRE_TEACHER_CAPACITY_EVIDENCE" >&2
    exit 64
fi

probe_cuda() {
    local cuda_state marker available count extra nvidia_smi
    nvidia_smi="${NVIDIA_SMI:-$(command -v nvidia-smi || true)}"
    if [[ -n "$nvidia_smi" ]] && ! "$nvidia_smi" -L >/dev/null 2>&1; then
        echo "[error] NVIDIA 驱动不可用（nvidia-smi -L 失败）；训练未启动，checkpoint 与日志均未改动。" >&2
        return 69
    fi
    if ! cuda_state="$("$PYTHON" -c \
        'import torch; print("D2R_CUDA", int(torch.cuda.is_available()), torch.cuda.device_count())')"; then
        echo "[error] 无法导入 torch 或探测 CUDA；请使用 diffsynth:v2-diffusers 镜像。" >&2
        return 70
    fi
    read -r marker available count extra <<<"$cuda_state"
    if [[ "$marker" != "D2R_CUDA" || ! "$available" =~ ^[01]$ || ! "$count" =~ ^[0-9]+$ || -n "${extra:-}" ]]; then
        echo "[error] 无法解析 CUDA 探测结果：$cuda_state" >&2
        return 70
    fi
    if [[ "$available" != "1" || "$count" -lt 1 ]]; then
        echo "[error] 当前主机没有可用 NVIDIA CUDA 驱动/GPU；训练未启动，checkpoint 与日志均未改动。" >&2
        return 69
    fi
    CUDA_DEVICE_COUNT="$count"
}

# This check deliberately precedes every mkdir/log/lock/checkpoint operation.
probe_cuda
if [[ "${CHECK_GPU_ONLY:-0}" == "1" ]]; then
    echo "[info] CUDA precheck passed: visible_gpus=$CUDA_DEVICE_COUNT"
    exit 0
fi

if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] EPOCHS 必须是正整数，当前值：$EPOCHS" >&2
    exit 64
fi
if [[ ! "$MASTER_PORT" =~ ^[0-9]+$ || "$MASTER_PORT" -lt 1 || "$MASTER_PORT" -gt 65535 ]]; then
    echo "[error] MASTER_PORT 必须在 1..65535，当前值：$MASTER_PORT" >&2
    exit 64
fi
if [[ ! "$MIN_RESUME_COMPLETED_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "[error] MIN_RESUME_COMPLETED_EPOCHS 必须是非负整数，当前值：$MIN_RESUME_COMPLETED_EPOCHS" >&2
    exit 64
fi
if [[ "$ALLOW_RESUME_BELOW_MIN_EPOCHS" != "0" && "$ALLOW_RESUME_BELOW_MIN_EPOCHS" != "1" ]]; then
    echo "[error] ALLOW_RESUME_BELOW_MIN_EPOCHS 只能是 0 或 1，当前值：$ALLOW_RESUME_BELOW_MIN_EPOCHS" >&2
    exit 64
fi
if [[ "${CHECK_LAUNCH_ONLY:-0}" != "0" && "${CHECK_LAUNCH_ONLY:-0}" != "1" ]]; then
    echo "[error] CHECK_LAUNCH_ONLY 只能是 0 或 1，当前值：${CHECK_LAUNCH_ONLY}" >&2
    exit 64
fi
if [[ "$ISOLATE_CUDA_PER_RANK" != "0" && "$ISOLATE_CUDA_PER_RANK" != "1" ]]; then
    echo "[error] ISOLATE_CUDA_PER_RANK 只能是 0 或 1，当前值：$ISOLATE_CUDA_PER_RANK" >&2
    exit 64
fi

# torch.cuda.device_count() already honours CUDA_VISIBLE_DEVICES.
if [[ -n "${NUM_GPUS:-}" ]]; then
    if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[error] NUM_GPUS 必须是正整数，当前值：$NUM_GPUS" >&2
        exit 64
    fi
    if (( NUM_GPUS > CUDA_DEVICE_COUNT )); then
        echo "[error] NUM_GPUS=$NUM_GPUS 超过 PyTorch 可见 GPU 数 $CUDA_DEVICE_COUNT。" >&2
        exit 69
    fi
    NPROC="$NUM_GPUS"
else
    NPROC="$CUDA_DEVICE_COUNT"
fi

OUTPUT_STAGE_DIR="${OUTPUT_STAGE_DIR:-${OUTPUT_ROOT}/${STAGE}}"
STATE_DIR="${STATE_DIR:-${OUTPUT_STAGE_DIR}/.launcher}"
LOCK_FILE="${LOCK_FILE:-${STATE_DIR}/train.lock}"
PID_FILE="${PID_FILE:-${STATE_DIR}/train.pid}"
STATUS_FILE="${STATUS_FILE:-${STATE_DIR}/last_exit.status}"
LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-${OUTPUT_STAGE_DIR}/latest.pt}"

RESUME_SELECTED=""
RESUME_REASON=""

select_resume_checkpoint() {
    local -a epoch_candidates=()
    local candidate index

    # Rollback is opt-in if a latest checkpoint exists.
    if [[ -n "${RESUME:-}" && "${ALLOW_RESUME_OVERRIDE:-0}" == "1" ]]; then
        RESUME_SELECTED="$RESUME"
        RESUME_REASON="explicit override"
        return
    fi

    if [[ -e "$LATEST_CHECKPOINT" ]]; then
        if [[ ! -f "$LATEST_CHECKPOINT" || ! -s "$LATEST_CHECKPOINT" ]]; then
            echo "[error] latest checkpoint 存在但为空/非普通文件：$LATEST_CHECKPOINT；为避免回退覆盖，停止。" >&2
            return 66
        fi
        if [[ -n "${RESUME:-}" ]]; then
            echo "[warn] 检测到 latest.pt，忽略 RESUME=$RESUME；如确需回滚，设置 ALLOW_RESUME_OVERRIDE=1。" >&2
        fi
        RESUME_SELECTED="$LATEST_CHECKPOINT"
        RESUME_REASON="latest"
        return
    fi

    if [[ -n "${RESUME:-}" ]]; then
        RESUME_SELECTED="$RESUME"
        RESUME_REASON="explicit (latest absent)"
        return
    fi

    shopt -s nullglob
    for candidate in "$OUTPUT_STAGE_DIR"/epoch_*.pt; do
        [[ -f "$candidate" && -s "$candidate" ]] && epoch_candidates+=("$candidate")
    done
    shopt -u nullglob
    if (( ${#epoch_candidates[@]} > 0 )); then
        IFS=$'\n' epoch_candidates=($(LC_ALL=C printf '%s\n' "${epoch_candidates[@]}" | LC_ALL=C sort))
        unset IFS
        index=$((${#epoch_candidates[@]} - 1))
        RESUME_SELECTED="${epoch_candidates[$index]}"
        RESUME_REASON="latest epoch snapshot"
        return
    fi

    if [[ -f "$OUTPUT_STAGE_DIR/best.pt" && -s "$OUTPUT_STAGE_DIR/best.pt" ]]; then
        RESUME_SELECTED="$OUTPUT_STAGE_DIR/best.pt"
        RESUME_REASON="best fallback"
        return
    fi
    RESUME_SELECTED="$SEED_RESUME"
    RESUME_REASON="$SEED_RESUME_REASON"
}

# Resume selection and metadata validation are deliberately read-only and run
# before mkdir/lock/PID publication. An ineligible v3.2 seed must leave no
# launcher state behind.
select_resume_checkpoint
if [[ ! -f "$RESUME_SELECTED" || ! -s "$RESUME_SELECTED" ]]; then
    echo "[error] 恢复 checkpoint 不存在或为空：$RESUME_SELECTED" >&2
    exit 66
fi

# Never fall back from a bad latest.pt to the seed.
if ! CHECKPOINT_META="$("$PYTHON" scripts/checkpoint_status.py \
    --checkpoint "$RESUME_SELECTED" \
    --expect-stage "$STAGE" \
    --require-optimizer \
    --format tsv)"; then
    echo "[error] checkpoint 校验失败，训练未启动：$RESUME_SELECTED" >&2
    exit 65
fi
IFS=$'\t' read -r CHECKPOINT_STAGE CHECKPOINT_EPOCH_INDEX COMPLETED_EPOCHS CHECKPOINT_BEST_NAME CHECKPOINT_BEST_VALUE <<<"$CHECKPOINT_META"
if [[ "$CHECKPOINT_STAGE" != "$STAGE" || ! "$CHECKPOINT_EPOCH_INDEX" =~ ^-?[0-9]+$ || ! "$COMPLETED_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "[error] checkpoint 元数据非法：$CHECKPOINT_META" >&2
    exit 65
fi
if [[ "$RESUME_REASON" != "$SEED_RESUME_REASON" && "$CHECKPOINT_BEST_NAME" != "$BEST_METRIC" ]]; then
    echo "[error] 已训练 checkpoint 的 best metric 是 '$CHECKPOINT_BEST_NAME'，期望 '$BEST_METRIC'；拒绝覆盖 best.pt。" >&2
    exit 65
fi
if (( COMPLETED_EPOCHS < MIN_RESUME_COMPLETED_EPOCHS )); then
    if [[ "$ALLOW_RESUME_BELOW_MIN_EPOCHS" != "1" ]]; then
        echo "[error] ${LAUNCH_LABEL} 恢复点仅完成 $COMPLETED_EPOCHS 个 epoch，必须至少完成 $MIN_RESUME_COMPLETED_EPOCHS 个：$RESUME_SELECTED" >&2
        echo "[error] 默认拒绝提前迁移，未创建运行状态。仅在明确接受风险时设置 ALLOW_RESUME_BELOW_MIN_EPOCHS=1。" >&2
        exit 78
    fi
    echo "[warn] ALLOW_RESUME_BELOW_MIN_EPOCHS=1：允许 ${LAUNCH_LABEL} 从 $COMPLETED_EPOCHS/$MIN_RESUME_COMPLETED_EPOCHS epoch 的恢复点提前启动。" >&2
fi

# Authenticate the selected resume against the approved teacher-capacity
# evidence before CHECK_LAUNCH_ONLY returns and before any launcher state is
# created.  Preserve stdout's trailing newlines with a non-base64 sentinel so
# an empty, multi-line, or otherwise non-canonical receipt cannot be accepted.
if [[ "$REQUIRE_TEACHER_CAPACITY_EVIDENCE" == "1" ]]; then
    if [[ -z "${TEACHER_CAPACITY_POINTER:-}" || ! -f "$TEACHER_CAPACITY_POINTER" || ! -s "$TEACHER_CAPACITY_POINTER" ]]; then
        echo "[error] teacher-capacity evidence pointer 不存在、为空或非普通文件：${TEACHER_CAPACITY_POINTER:-<unset>}" >&2
        exit 79
    fi
    capacity_stdout_sentinel=$'\034'
    set +e
    capacity_stdout="$(
        "$PYTHON" scripts/teacher_capacity_production.py verify \
            --config "$CONFIG" \
            --pointer "$TEACHER_CAPACITY_POINTER" \
            --resume "$RESUME_SELECTED"
        capacity_verify_status=$?
        printf '%s' "$capacity_stdout_sentinel"
        exit "$capacity_verify_status"
    )"
    capacity_verify_status=$?
    set -e
    if (( capacity_verify_status != 0 )); then
        echo "[error] teacher-capacity evidence 校验失败；训练未启动。" >&2
        exit 79
    fi
    if [[ "$capacity_stdout" != *"$capacity_stdout_sentinel" ]]; then
        echo "[error] teacher-capacity verifier 输出捕获失败；训练未启动。" >&2
        exit 79
    fi
    capacity_receipt="${capacity_stdout%"$capacity_stdout_sentinel"}"
    if [[ "$capacity_receipt" == *$'\n' ]]; then
        capacity_receipt="${capacity_receipt%$'\n'}"
    fi
    if [[ -z "$capacity_receipt" || "$capacity_receipt" == *$'\n'* || ! "$capacity_receipt" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "[error] teacher-capacity verifier 必须只输出一行 canonical base64 receipt；训练未启动。" >&2
        exit 79
    fi
    export D2R_TEACHER_CAPACITY_RECEIPT_B64="$capacity_receipt"
elif [[ "${D2R_TEACHER_CAPACITY_RECEIPT_B64+x}" == "x" ]]; then
    echo "[error] 未启用 teacher-capacity evidence 时禁止继承 D2R_TEACHER_CAPACITY_RECEIPT_B64；训练未启动。" >&2
    exit 79
fi

if [[ "${CHECK_LAUNCH_ONLY:-0}" == "1" ]]; then
    echo "[info] 启动预检通过：resume=$RESUME_SELECTED completed_epochs=$COMPLETED_EPOCHS"
    exit 0
fi

# No two launchers may write latest.pt/best.pt concurrently.
mkdir -p "$STATE_DIR"
if [[ "${TRAIN_LOCK_HELD:-0}" == "1" ]]; then
    TRAIN_LOCK_FD="${TRAIN_LOCK_FD:-9}"
    if ! flock -n "$TRAIN_LOCK_FD"; then
        echo "[error] 后台启动器声明了锁，但 fd=$TRAIN_LOCK_FD 无效或未持锁。" >&2
        exit 75
    fi
else
    exec {TRAIN_LOCK_FD}>"$LOCK_FILE"
    if ! flock -n "$TRAIN_LOCK_FD"; then
        existing_pid="$(sed -n 's/^pid=//p' "$PID_FILE" 2>/dev/null | head -n 1 || true)"
        echo "[error] 已有 ${LAUNCH_LABEL} 训练持有锁（pid=${existing_pid:-unknown}）；拒绝重复启动。" >&2
        exit 75
    fi
fi

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PID_TMP="${PID_FILE}.tmp.$$"
STATUS_TMP="${STATUS_FILE}.tmp.$$"
ACTIVE_CHILD_PID=""
FORWARDED_SIGNAL_EXIT=0

forward_active_signal() {
    local signal_name="$1" signal_exit="$2"
    FORWARDED_SIGNAL_EXIT="$signal_exit"
    if [[ -n "$ACTIVE_CHILD_PID" ]] && kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
        kill -s "$signal_name" "$ACTIVE_CHILD_PID" 2>/dev/null || true
    else
        exit "$signal_exit"
    fi
}

run_managed_child() {
    local child_status
    "$@" &
    ACTIVE_CHILD_PID=$!
    while true; do
        set +e
        wait "$ACTIVE_CHILD_PID"
        child_status=$?
        set -e
        # wait is interrupted when a trapped signal is delivered.  Keep
        # waiting until the child that received the forwarded signal is reaped.
        if ! kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
            break
        fi
    done
    ACTIVE_CHILD_PID=""
    if (( FORWARDED_SIGNAL_EXIT != 0 )); then
        return "$FORWARDED_SIGNAL_EXIT"
    fi
    return "$child_status"
}

cleanup_launcher_state() {
    local exit_code=$?
    trap - EXIT
    if [[ -f "$PID_FILE" ]] && grep -qx "pid=$$" "$PID_FILE" 2>/dev/null; then
        rm -f -- "$PID_FILE"
    fi
    {
        printf 'exit_code=%s\n' "$exit_code"
        printf 'started_at=%s\n' "$STARTED_AT"
        printf 'ended_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'resume=%s\n' "${RESUME_SELECTED:-not_selected}"
        printf 'log=%s\n' "${TRAIN_LOG_FILE:-foreground}"
    } >"$STATUS_TMP"
    mv -f -- "$STATUS_TMP" "$STATUS_FILE"
    rm -f -- "$PID_TMP"
    exit "$exit_code"
}
trap cleanup_launcher_state EXIT
trap 'forward_active_signal HUP 129' HUP
trap 'forward_active_signal INT 130' INT
trap 'forward_active_signal TERM 143' TERM

{
    printf 'pid=%s\n' "$$"
    printf 'started_at=%s\n' "$STARTED_AT"
    printf 'log=%s\n' "${TRAIN_LOG_FILE:-foreground}"
} >"$PID_TMP"
mv -f -- "$PID_TMP" "$PID_FILE"

echo "[info] repo=$REPO_ROOT python=$PYTHON visible_gpus=$CUDA_DEVICE_COUNT nproc_per_node=$NPROC"
echo "[info] resume=$RESUME_SELECTED reason='$RESUME_REASON' completed_epochs=$COMPLETED_EPOCHS target_epochs=$EPOCHS"
echo "[info] checkpoint best=$CHECKPOINT_BEST_NAME:$CHECKPOINT_BEST_VALUE; 输出=$OUTPUT_STAGE_DIR"
if [[ "$ISOLATE_CUDA_PER_RANK" == "1" ]]; then
    echo "[info] 每个 DDP rank 将隔离一张物理 GPU，并在 worker 内映射为逻辑 cuda:0。"
fi

if (( COMPLETED_EPOCHS >= EPOCHS )); then
    echo "[info] checkpoint 已完成 $COMPLETED_EPOCHS 个 epoch（目标 $EPOCHS）；无需启动训练。"
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    echo "[info] 运行训练前 checkpoint/manifest/flow-canvas 检查。"
    run_managed_child "$PYTHON" scripts/preflight_v31.py \
        --config "$CONFIG" \
        --checkpoint "$RESUME_SELECTED" \
        --sample-count "${PREFLIGHT_SAMPLES:-3}" \
        --max-mae "${MAX_GT_MAE:-0.08}" \
        --output-dir "${PREFLIGHT_OUTPUT_DIR:-runs/preflight_v31/main}"
elif [[ "$RUN_PREFLIGHT" != "0" ]]; then
    echo "[error] RUN_PREFLIGHT 只能是 0 或 1，当前值：$RUN_PREFLIGHT" >&2
    exit 64
fi

echo "[info] 从显示 epoch $((COMPLETED_EPOCHS + 1)) 恢复，训练到 epoch $EPOCHS。"
if [[ "$ISOLATE_CUDA_PER_RANK" == "1" ]]; then
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    run_managed_child \
        "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" \
        --max_restarts=0 \
        --no_python \
        "$REPO_ROOT/scripts/isolate_cuda_rank.sh" \
        "$PYTHON" -m diffusion2raft.train \
        --config "$CONFIG" \
        --stage "$STAGE" \
        --resume "$RESUME_SELECTED" \
        --output-dir "$OUTPUT_ROOT" \
        --epochs "$EPOCHS"
else
    run_managed_child \
        "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" \
        --max_restarts=0 \
        -m diffusion2raft.train \
        --config "$CONFIG" \
        --stage "$STAGE" \
        --resume "$RESUME_SELECTED" \
        --output-dir "$OUTPUT_ROOT" \
        --epochs "$EPOCHS"
fi
