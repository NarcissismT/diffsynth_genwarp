#!/usr/bin/env bash
# Mirror explicitly selected source files from workspace to the data-side tree.
# This script never deletes destination files and never copies runs/logs/checkpoints.
set -euo pipefail

WORKSPACE_ROOT="/juicefs-algorithm/workspace/IPT/zhuochu_yang/diffsynth_genwarp"
DATA_ROOT="/juicefs-algorithm/data/IPT/zhuochu_yang/diffsynth_genwarp"
V31_PROJECT="diffusion2raft_unified_v3_1"
CP_PROJECT="cp_docflow_v1"

fail() {
    echo "[error] $*" >&2
    exit 64
}

[[ -d "$WORKSPACE_ROOT" ]] || fail "workspace root 不存在：$WORKSPACE_ROOT"
[[ -d "$DATA_ROOT" ]] || fail "data mirror root 不存在：$DATA_ROOT"
[[ "$(realpath -e "$WORKSPACE_ROOT")" != "$(realpath -e "$DATA_ROOT")" ]] \
    || fail "源目录和目标目录不能相同"
command -v rsync >/dev/null 2>&1 || fail "缺少 rsync"

sync_relative_path() {
    local relative="$1"
    case "$relative" in
        ""|/*|..|../*|*/../*|*/..)
            fail "只允许 workspace 根目录下不含 .. 的相对路径：$relative"
            ;;
    esac
    [[ -e "$WORKSPACE_ROOT/$relative" || -L "$WORKSPACE_ROOT/$relative" ]] \
        || fail "源路径不存在：$WORKSPACE_ROOT/$relative"
    (
        cd "$WORKSPACE_ROOT"
        rsync -aR \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            -- "$relative" "$DATA_ROOT/"
    )
}

verify_relative_path() {
    local relative="$1"
    diff -qr \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        -- "$WORKSPACE_ROOT/$relative" "$DATA_ROOT/$relative"
}

sync_v31_code() {
    local paths=(
        "PROGRESS_PLAN.md"
        "Diffusion2RAFT_Plan_and_Goals.md"
        "Diffusion2RAFT_Plan_and_Goals_newest.md"
        "$V31_PROJECT/.gitignore"
        "$V31_PROJECT/README.md"
        "$V31_PROJECT/V3_1_NOTES.md"
        "$V31_PROJECT/pyproject.toml"
        "$V31_PROJECT/requirements.txt"
        "$V31_PROJECT/configs"
        "$V31_PROJECT/docs"
        "$V31_PROJECT/examples"
        "$V31_PROJECT/scripts"
        "$V31_PROJECT/src"
        "$V31_PROJECT/tests"
        "$CP_PROJECT/.gitignore"
        "$CP_PROJECT/README.md"
        "$CP_PROJECT/pyproject.toml"
        "$CP_PROJECT/configs"
        "$CP_PROJECT/docs"
        "$CP_PROJECT/examples"
        "$CP_PROJECT/src"
        "$CP_PROJECT/slurm"
        "$CP_PROJECT/tests"
        "slurm/v33_pipeline"
        "scripts/sync_code_mirror.sh"
    )
    local relative
    for relative in "${paths[@]}"; do
        sync_relative_path "$relative"
    done
    for relative in "${paths[@]}"; do
        verify_relative_path "$relative"
    done
    echo "[ok] v3.1 + DocGrid-Flow v2 代码与计划已同步并校验：$WORKSPACE_ROOT -> $DATA_ROOT"
}

if [[ $# -eq 0 ]]; then
    fail "用法：$0 --v31 或 $0 RELATIVE_PATH [RELATIVE_PATH ...]"
fi

if [[ "$1" == "--v31" ]]; then
    [[ $# -eq 1 ]] || fail "--v31 不接受额外路径参数"
    sync_v31_code
    exit 0
fi

for relative in "$@"; do
    sync_relative_path "$relative"
done
for relative in "$@"; do
    verify_relative_path "$relative"
done
echo "[ok] 已同步并校验 $# 个明确路径：$WORKSPACE_ROOT -> $DATA_ROOT"
