#!/usr/bin/env bash
# Real foreground Qwen+teacher DDP smoke.  Formal run directories are never used.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

fail() {
    echo "[error] v3.3 smoke: $*" >&2
    exit 64
}

PYTHON="${PYTHON:-/juicefs-algorithm/workspace/IPT/zhuochu_yang/miniconda3/bin/python}"
CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"
SEED_CHECKPOINT="${SEED_CHECKPOINT:-runs/d2r_v3_1/unified/latest.pt}"
TEACHER_CAPACITY_POINTER="${TEACHER_CAPACITY_POINTER:-runs/preflight_v33_teacher_capacity/approved.json}"
MASTER_PORT="${MASTER_PORT:-29633}"
MIN_SEED_COMPLETED_EPOCHS="${MIN_SEED_COMPLETED_EPOCHS:-20}"
ALLOW_INCOMPLETE_SEED="${ALLOW_INCOMPLETE_SEED:-0}"
ALLOW_PARTIAL_SMOKE="${ALLOW_PARTIAL_SMOKE:-0}"
FAILURE_TIMEOUT_SECONDS="${FAILURE_TIMEOUT_SECONDS:-120}"
FUNCTIONAL_TIMEOUT_SECONDS="${FUNCTIONAL_TIMEOUT_SECONDS:-3600}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}}"
REPORT_ROOT="${REPORT_ROOT:-runs/v33_smoke_reports/${RUN_ID}}"

[[ -x "$PYTHON" ]] || fail "Python 不可执行：$PYTHON"
[[ -f "$CONFIG" ]] || fail "配置不存在：$CONFIG"
[[ -s "$SEED_CHECKPOINT" ]] || fail "seed checkpoint 不存在或为空：$SEED_CHECKPOINT"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || fail "MASTER_PORT 必须是整数"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65532 )) || fail "MASTER_PORT 必须在 1..65532"
[[ "$MIN_SEED_COMPLETED_EPOCHS" =~ ^[0-9]+$ ]] || fail "最小 seed epoch 必须是非负整数"
[[ "$ALLOW_INCOMPLETE_SEED" == "0" || "$ALLOW_INCOMPLETE_SEED" == "1" ]] \
    || fail "ALLOW_INCOMPLETE_SEED 只能是 0 或 1"
[[ "$ALLOW_PARTIAL_SMOKE" == "0" || "$ALLOW_PARTIAL_SMOKE" == "1" ]] \
    || fail "ALLOW_PARTIAL_SMOKE 只能是 0 或 1"
[[ "$FAILURE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "failure timeout 必须为正整数"
[[ "$FUNCTIONAL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || fail "functional timeout 必须为正整数"
[[ "$RUN_ID" != */* && "$RUN_ID" != "." && "$RUN_ID" != ".." ]] \
    || fail "RUN_ID 必须是单个路径组件"
command -v timeout >/dev/null 2>&1 || fail "缺少 timeout 命令"
command -v realpath >/dev/null 2>&1 || fail "缺少 realpath 命令"

formal_run_root="$(realpath -m -- "$REPO_ROOT/runs/d2r_v3_3_teacher_anchor")"
CONFIG="$(realpath -e -- "$CONFIG")"
formal_config="$(realpath -e -- "$REPO_ROOT/configs/unified_v3_3_teacher_anchor.yaml")"
REPORT_ROOT="$(realpath -m -- "$REPORT_ROOT")"
case "$REPORT_ROOT" in
    "$formal_run_root"|"$formal_run_root"/*)
        fail "smoke report 不得写入正式 v3.3 run：$REPORT_ROOT"
        ;;
esac
[[ ! -e "$REPORT_ROOT" && ! -L "$REPORT_ROOT" ]] \
    || fail "smoke report 目录已存在：$REPORT_ROOT"

cuda_probe="$("$PYTHON" -c 'import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())')"
read -r cuda_available cuda_count extra <<<"$cuda_probe"
[[ "$cuda_available" == "1" && "$cuda_count" =~ ^[1-9][0-9]*$ && -z "${extra:-}" ]] \
    || fail "CUDA 不可用或探测结果异常：$cuda_probe"
SMOKE_NPROC="${SMOKE_NPROC:-8}"
[[ "$SMOKE_NPROC" =~ ^[1-9][0-9]*$ ]] || fail "SMOKE_NPROC 必须是正整数"
(( SMOKE_NPROC <= cuda_count )) || fail "SMOKE_NPROC 超过可见 GPU 数"

FAILURE_WORLD_SIZES="${FAILURE_WORLD_SIZES:-2 8}"
declare -A requested_failure_sizes=()
failure_world_sizes=()
for failure_world_size in $FAILURE_WORLD_SIZES; do
    [[ "$failure_world_size" =~ ^[1-9][0-9]*$ ]] \
        || fail "FAILURE_WORLD_SIZES 包含非正整数：$failure_world_size"
    (( failure_world_size >= 2 )) || fail "failure propagation 至少需要 2 ranks"
    (( failure_world_size <= cuda_count )) \
        || fail "failure world size $failure_world_size 超过可见 GPU 数 $cuda_count"
    [[ -z "${requested_failure_sizes[$failure_world_size]+x}" ]] || continue
    requested_failure_sizes[$failure_world_size]=1
    failure_world_sizes+=("$failure_world_size")
done
(( ${#failure_world_sizes[@]} > 0 )) || fail "FAILURE_WORLD_SIZES 不得为空"
if [[ "$ALLOW_PARTIAL_SMOKE" == "0" ]]; then
    [[ "$CONFIG" == "$formal_config" ]] \
        || fail "正式 smoke 必须使用项目固定的 unified_v3_3_teacher_anchor.yaml；其他配置只能做 partial smoke"
    (( SMOKE_NPROC == 8 )) \
        || fail "正式 smoke 必须使用 SMOKE_NPROC=8；诊断运行需设置 ALLOW_PARTIAL_SMOKE=1"
    [[ -n "${requested_failure_sizes[2]+x}" && -n "${requested_failure_sizes[8]+x}" ]] \
        || fail "正式 smoke 的 FAILURE_WORLD_SIZES 必须同时包含 2 和 8"
fi

SMOKE_BASE="${SLURM_TMPDIR:-/tmp}"
[[ -d "$SMOKE_BASE" && -w "$SMOKE_BASE" ]] || fail "临时目录不可写：$SMOKE_BASE"
SMOKE_TMP="$(mktemp -d "${SMOKE_BASE%/}/d2r-v33-smoke.${SLURM_JOB_ID:-manual}.XXXXXX")"
cleanup() {
    case "$SMOKE_TMP" in
        "${SMOKE_BASE%/}"/d2r-v33-smoke.*)
            rm -rf -- "$SMOKE_TMP"
            ;;
        *)
            echo "[error] 拒绝清理非 smoke 临时路径：$SMOKE_TMP" >&2
            ;;
    esac
}
trap cleanup EXIT

# Freeze a mutable latest.pt before any metadata read or rank load.  An atomic
# trainer replacement of the source can no longer make different smoke stages
# observe different seeds.
SEED_SOURCE="$SEED_CHECKPOINT"
FROZEN_SEED_CHECKPOINT="$SMOKE_TMP/seed.pt"
cp --reflink=auto -- "$SEED_SOURCE" "$FROZEN_SEED_CHECKPOINT"
[[ -s "$FROZEN_SEED_CHECKPOINT" ]] || fail "冻结 seed checkpoint 失败"
SEED_CHECKPOINT="$FROZEN_SEED_CHECKPOINT"

checkpoint_tsv="$("$PYTHON" scripts/checkpoint_status.py \
    --checkpoint "$SEED_CHECKPOINT" \
    --expect-stage unified \
    --require-optimizer \
    --format tsv)"
IFS=$'\t' read -r seed_stage seed_epoch completed_epochs best_name best_value <<<"$checkpoint_tsv"
[[ "$seed_stage" == "unified" && "$completed_epochs" =~ ^[0-9]+$ ]] \
    || fail "无法解析 seed checkpoint：$checkpoint_tsv"
if (( completed_epochs < MIN_SEED_COMPLETED_EPOCHS )); then
    if [[ "$ALLOW_INCOMPLETE_SEED" != "1" ]]; then
        fail "seed 仅完成 ${completed_epochs} epochs；正式 smoke 要求至少 ${MIN_SEED_COMPLETED_EPOCHS}。提前诊断需显式设置 ALLOW_INCOMPLETE_SEED=1"
    fi
    [[ "$ALLOW_PARTIAL_SMOKE" == "1" ]] \
        || fail "未满 ${MIN_SEED_COMPLETED_EPOCHS} epochs 的诊断不得输出正式 PASS；请同时设置 ALLOW_PARTIAL_SMOKE=1"
    echo "[warn] 使用未满 ${MIN_SEED_COMPLETED_EPOCHS} epochs 的 seed 做提前 smoke；不得据此启动正式 v3.3。" >&2
fi
if [[ "$ALLOW_PARTIAL_SMOKE" == "0" ]]; then
    (( MIN_SEED_COMPLETED_EPOCHS == 20 && completed_epochs >= 20 )) \
        || fail "正式 smoke 固定要求 learned seed 至少完成 20 epochs"
fi

[[ -f "$TEACHER_CAPACITY_POINTER" && -s "$TEACHER_CAPACITY_POINTER" ]] \
    || fail "teacher-capacity evidence pointer 不存在、为空或非普通文件：$TEACHER_CAPACITY_POINTER"
capacity_stdout_sentinel=$'\034'
set +e
capacity_stdout="$(
    "$PYTHON" scripts/teacher_capacity_production.py verify \
        --config "$CONFIG" \
        --pointer "$TEACHER_CAPACITY_POINTER" \
        --resume "$SEED_CHECKPOINT"
    capacity_verify_status=$?
    printf '%s' "$capacity_stdout_sentinel"
    exit "$capacity_verify_status"
)"
capacity_verify_status=$?
set -e
(( capacity_verify_status == 0 )) \
    || fail "teacher-capacity evidence 校验失败；smoke 未启动"
[[ "$capacity_stdout" == *"$capacity_stdout_sentinel" ]] \
    || fail "teacher-capacity verifier 输出捕获失败；smoke 未启动"
capacity_receipt="${capacity_stdout%"$capacity_stdout_sentinel"}"
if [[ "$capacity_receipt" == *$'\n' ]]; then
    capacity_receipt="${capacity_receipt%$'\n'}"
fi
[[ -n "$capacity_receipt" && "$capacity_receipt" != *$'\n'* \
    && "$capacity_receipt" =~ ^[A-Za-z0-9_-]+$ ]] \
    || fail "teacher-capacity verifier 必须只输出一行 canonical base64 receipt；smoke 未启动"
export D2R_TEACHER_CAPACITY_RECEIPT_B64="$capacity_receipt"

mkdir -p "$REPORT_ROOT"
FUNCTIONAL_ROOT="$SMOKE_TMP/functional"
TARGET_EPOCHS=$((completed_epochs + 1))

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "[info] v3.3 smoke seed_source=$SEED_SOURCE frozen_seed=$SEED_CHECKPOINT completed=$completed_epochs nproc=$SMOKE_NPROC temp=$SMOKE_TMP report=$REPORT_ROOT"
"$PYTHON" scripts/preflight_v31.py \
    --config "$CONFIG" \
    --checkpoint "$SEED_CHECKPOINT" \
    --sample-count 1 \
    --output-dir "$REPORT_ROOT/preflight"

timeout --signal=TERM --kill-after=30s "${FUNCTIONAL_TIMEOUT_SECONDS}s" \
    "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$SMOKE_NPROC" \
    --master_port="$MASTER_PORT" \
    --max_restarts=0 \
    --no_python \
    "$REPO_ROOT/scripts/isolate_cuda_rank.sh" \
    "$PYTHON" -m diffusion2raft.train \
    --config "$CONFIG" \
    --stage unified \
    --resume "$SEED_CHECKPOINT" \
    --epochs "$TARGET_EPOCHS" \
    --max-train-steps 1 \
    --max-val-batches 1 \
    --preview-every 0 \
    --output-dir "$FUNCTIONAL_ROOT" \
    2>&1 | tee "$REPORT_ROOT/functional.log"

"$PYTHON" scripts/verify_v33_smoke_checkpoint.py \
    --config "$CONFIG" \
    --seed-checkpoint "$SEED_CHECKPOINT" \
    --smoke-output-root "$FUNCTIONAL_ROOT" \
    --expected-seed-completed-epochs "$completed_epochs" \
    --invoked-world-size "$SMOKE_NPROC" \
    --functional-log "$REPORT_ROOT/functional.log" \
    --seed-source "$SEED_SOURCE" \
    --report "$REPORT_ROOT/functional_report.json"

failure_index=0
overall_evidence_args=()
for failure_world_size in "${failure_world_sizes[@]}"; do
    failure_port=$((MASTER_PORT + 1 + failure_index))
    (( failure_port <= 65535 )) || fail "failure probe master port 越界"
    failure_index=$((failure_index + 1))
    fail_rank=$((failure_world_size - 1))
    token="v33-smoke-${RUN_ID}-w${failure_world_size}-p$$"
    log="$REPORT_ROOT/ddp_failure_w${failure_world_size}.log"
    command=(
        "$PYTHON" -m torch.distributed.run
        --nproc_per_node="$failure_world_size"
        --master_port="$failure_port"
        --max_restarts=0
        --no_python
        "$REPO_ROOT/scripts/isolate_cuda_rank.sh"
        "$PYTHON" scripts/probe_v33_ddp_isolation.py
        --expected-world-size "$failure_world_size"
        --fail-rank "$fail_rank"
        --failure-token "$token"
    )
    set +e
    timeout --signal=TERM --kill-after=10s "${FAILURE_TIMEOUT_SECONDS}s" \
        "${command[@]}" >"$log" 2>&1
    status=$?
    set -e
    [[ "$status" -ne 0 ]] || fail "intentional DDP failure unexpectedly returned success"
    case "$status" in
        124|137|143)
            fail "DDP failure propagation timed out or required forced termination (status=$status); see $log"
            ;;
    esac
    grep -F "D2R_EXPECTED_RANK_FAILURE $token" "$log" >/dev/null \
        || fail "DDP failure log lacks the unique expected token: $log"
    grep -F "D2R_DDP_ISOLATION_OK" "$log" >/dev/null \
        || fail "DDP isolation did not complete before intentional failure: $log"
    overall_evidence_args+=(
        --failure-evidence "$failure_world_size" "$status" "$token" "$log"
    )
    echo "D2R_DDP_FAILURE_PROPAGATION_OK world_size=$failure_world_size log=$log"
done

if [[ "$ALLOW_PARTIAL_SMOKE" == "1" ]]; then
    overall_mode="partial"
else
    overall_mode="formal"
fi
"$PYTHON" scripts/finalize_v33_smoke_report.py \
    --mode "$overall_mode" \
    --functional-report "$REPORT_ROOT/functional_report.json" \
    --preflight-report "$REPORT_ROOT/preflight/preflight_report.json" \
    "${overall_evidence_args[@]}" \
    --report "$REPORT_ROOT/overall_report.json"
