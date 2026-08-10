#!/usr/bin/env bash
# Start the safe v3.1 continuation detached from the terminal.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_unified_v3.sh"

STAGE="${STAGE:-unified}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_1}"
LAUNCH_LABEL="${LAUNCH_LABEL:-v3.1}"
TRAIN_LOG_PREFIX="${TRAIN_LOG_PREFIX:-train_v31}"
OUTPUT_STAGE_DIR="${OUTPUT_STAGE_DIR:-${OUTPUT_ROOT}/${STAGE}}"
STATE_DIR="${STATE_DIR:-${OUTPUT_STAGE_DIR}/.launcher}"
LOCK_FILE="${LOCK_FILE:-${STATE_DIR}/train.lock}"
PID_FILE="${PID_FILE:-${STATE_DIR}/train.pid}"
STATUS_FILE="${STATUS_FILE:-${STATE_DIR}/last_exit.status}"
PYTHON="${PYTHON:-/usr/bin/python}"

# Reuse the foreground CUDA + resume-metadata checks before creating any pid,
# lock, or log file. This is read-only and also enforces the v3.2 seed gate.
CHECK_LAUNCH_ONLY=1 PYTHON="$PYTHON" "$TRAIN_SCRIPT"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    existing_pid="$(sed -n 's/^pid=//p' "$PID_FILE" 2>/dev/null | head -n 1 || true)"
    existing_log="$(sed -n 's/^log=//p' "$PID_FILE" 2>/dev/null | head -n 1 || true)"
    echo "[error] 已有 ${LAUNCH_LABEL} 训练运行中（pid=${existing_pid:-unknown}, log=${existing_log:-unknown}）；未重复启动。" >&2
    exit 75
fi

LOG_DIR="${LOG_DIR:-${OUTPUT_STAGE_DIR}/logs}"
mkdir -p "$LOG_DIR"
if [[ -n "${LOG_FILE:-}" ]]; then
    TRAIN_LOG_FILE="$LOG_FILE"
else
    TRAIN_LOG_FILE="$LOG_DIR/${TRAIN_LOG_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
fi
mkdir -p "$(dirname "$TRAIN_LOG_FILE")"

echo "[info] 启动 detached ${LAUNCH_LABEL} 续训；日志：$TRAIN_LOG_FILE"

# fd 9 is inherited by the detached process, keeping the lock after this
# launcher exits. nohup + setsid + /dev/null survives terminal disconnects.
TRAIN_LOCK_HELD=1 \
TRAIN_LOCK_FD=9 \
TRAIN_LOG_FILE="$TRAIN_LOG_FILE" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
OUTPUT_STAGE_DIR="$OUTPUT_STAGE_DIR" \
STATE_DIR="$STATE_DIR" \
LOCK_FILE="$LOCK_FILE" \
PID_FILE="$PID_FILE" \
STATUS_FILE="$STATUS_FILE" \
PYTHON="$PYTHON" \
nohup setsid --wait "$TRAIN_SCRIPT" </dev/null >>"$TRAIN_LOG_FILE" 2>&1 9>&9 &
launcher_pid=$!

# Wait only for pid publication, never for training completion.
published_pid=""
for _ in $(seq 1 150); do
    published_pid="$(sed -n 's/^pid=//p' "$PID_FILE" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$published_pid" ]]; then
        break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
        set +e
        wait "$launcher_pid"
        exit_code=$?
        set -e
        echo "[error] 后台进程在发布 PID 前退出（code=$exit_code）；查看 $TRAIN_LOG_FILE" >&2
        exit "$exit_code"
    fi
    sleep 0.2
done

if [[ -z "$published_pid" ]]; then
    echo "[error] 后台进程仍在，但 30 秒内未发布 PID；请查看 $TRAIN_LOG_FILE" >&2
    exit 70
fi

echo "[info] 已启动 pid=$published_pid"
echo "[info] 查看进度：tail -f '$TRAIN_LOG_FILE'"
echo "[info] 查看状态：cat '$PID_FILE'；完成后查看 '$STATUS_FILE'"
