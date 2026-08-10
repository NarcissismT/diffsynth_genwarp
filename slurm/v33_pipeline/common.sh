#!/usr/bin/env bash
# Shared, immutable production bindings for the v3.3 Slurm job bodies.

readonly D2R_WORKSPACE_ROOT="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
readonly D2R_PROJECT_ROOT="${D2R_WORKSPACE_ROOT}/diffusion2raft_unified_v3_1"
readonly D2R_PYTHON="/usr/bin/python"
readonly D2R_CONFIG="configs/unified_v3_3_teacher_anchor.yaml"
readonly D2R_SEED_CHECKPOINT="runs/d2r_v3_1/unified/epoch_0020.pt"
readonly D2R_EXPECTED_SEED_SHA256="7f5f743de8994907b916182fd3d1c1e81c0015b5b53fb8bf5038ff4c2ad17fe5"
readonly D2R_EXPECTED_TEACHER_SHA256="3d079e19445168169144f2af741362f673289b6510df4a4c1af348449ae045b9"
readonly D2R_CAPACITY_DIR="runs/preflight_v33_teacher_capacity"
readonly D2R_CAPACITY_POINTER="${D2R_CAPACITY_DIR}/approved.json"
readonly D2R_SMOKE_RUN_ID="epoch0020_formal"
readonly D2R_SMOKE_REPORT_ROOT="runs/v33_smoke_reports/${D2R_SMOKE_RUN_ID}"
readonly D2R_SMOKE_OVERALL_REPORT="${D2R_SMOKE_REPORT_ROOT}/overall_report.json"
readonly D2R_V33_OUTPUT_ROOT="runs/d2r_v3_3_teacher_anchor"

d2r_fail() {
    echo "[error] v3.3 Slurm pipeline: $*" >&2
    exit 64
}

d2r_cd_project() {
    [[ -d "$D2R_PROJECT_ROOT" ]] || d2r_fail "项目目录不存在：$D2R_PROJECT_ROOT"
    [[ -x "$D2R_PYTHON" ]] || d2r_fail "Python 不可执行：$D2R_PYTHON；请使用 diffsynth:v2-diffusers 镜像"
    command -v sha256sum >/dev/null 2>&1 || d2r_fail "缺少 sha256sum"
    command -v flock >/dev/null 2>&1 || d2r_fail "缺少 flock"
    cd "$D2R_PROJECT_ROOT"
}

d2r_checkpoint_status() {
    local checkpoint="$1" expected_epoch="$2" expected_completed="$3"
    local checkpoint_tsv stage epoch_index completed_epochs best_name best_value extra

    [[ -f "$checkpoint" && -s "$checkpoint" && ! -L "$checkpoint" ]] \
        || d2r_fail "checkpoint 不存在、为空或为符号链接：$checkpoint"
    checkpoint_tsv="$(
        "$D2R_PYTHON" scripts/checkpoint_status.py \
            --checkpoint "$checkpoint" \
            --expect-stage unified \
            --require-optimizer \
            --format tsv
    )" || d2r_fail "checkpoint 深检失败：$checkpoint"
    IFS=$'\t' read -r stage epoch_index completed_epochs best_name best_value extra \
        <<<"$checkpoint_tsv"
    [[ "$stage" == "unified" \
        && "$epoch_index" == "$expected_epoch" \
        && "$completed_epochs" == "$expected_completed" \
        && "$best_name" == "line_epe" \
        && -n "$best_value" \
        && -z "${extra:-}" ]] \
        || d2r_fail "checkpoint 元数据不符合预期：$checkpoint_tsv"
}

d2r_verify_seed() {
    local digest_line actual_sha
    d2r_checkpoint_status "$D2R_SEED_CHECKPOINT" 19 20
    digest_line="$(sha256sum -- "$D2R_SEED_CHECKPOINT")" \
        || d2r_fail "无法计算 seed SHA-256"
    actual_sha="${digest_line%% *}"
    [[ "$actual_sha" == "$D2R_EXPECTED_SEED_SHA256" ]] \
        || d2r_fail "epoch_0020.pt SHA-256 已变化：expected=$D2R_EXPECTED_SEED_SHA256 actual=$actual_sha"
}

d2r_require_visible_gpus() {
    local expected="$1" probe available count extra
    probe="$(
        "$D2R_PYTHON" -c \
            'import torch; print("D2R_CUDA", int(torch.cuda.is_available()), torch.cuda.device_count())'
    )" || d2r_fail "无法导入 torch 或探测 CUDA"
    read -r extra available count <<<"$probe"
    [[ "$extra" == "D2R_CUDA" && "$available" == "1" && "$count" == "$expected" ]] \
        || d2r_fail "要求恰好 ${expected} 张可见 GPU，探测结果：$probe"
}

d2r_require_capacity_pointer() {
    [[ -f "$D2R_CAPACITY_POINTER" && -s "$D2R_CAPACITY_POINTER" \
        && ! -L "$D2R_CAPACITY_POINTER" ]] \
        || d2r_fail "production capacity pointer 不存在、为空或为符号链接：$D2R_CAPACITY_POINTER"
}

d2r_acquire_job_lock() {
    local name="$1" lock_dir lock_path
    lock_dir="${D2R_PROJECT_ROOT}/runs/.v33_pipeline_locks"
    mkdir -p "$lock_dir"
    lock_path="${lock_dir}/${name}.lock"
    [[ ! -L "$lock_path" && ( ! -e "$lock_path" || -f "$lock_path" ) ]] \
        || d2r_fail "作业锁是符号链接或非普通文件：$lock_path"
    exec {D2R_PIPELINE_LOCK_FD}>"$lock_path"
    flock -n "$D2R_PIPELINE_LOCK_FD" \
        || d2r_fail "已有同阶段作业持有锁：$lock_path"
}

d2r_verify_receipt_text() {
    local receipt="$1"
    [[ -n "$receipt" && "$receipt" != *$'\n'* \
        && "$receipt" =~ ^[A-Za-z0-9_-]+$ ]] \
        || d2r_fail "capacity verify 必须只输出一行 canonical base64 receipt"
}
