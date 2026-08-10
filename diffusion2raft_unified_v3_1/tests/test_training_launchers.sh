#!/usr/bin/env bash
# Static/simulated tests for the v3.1/v3.2/v3.3 shell launchers. No GPU or torch required.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_unified_v3.sh"
BACKGROUND_SCRIPT="$REPO_ROOT/scripts/run_unified_v31_background.sh"
V32_TRAIN_SCRIPT="$REPO_ROOT/scripts/train_unified_v32.sh"
V32_BACKGROUND_SCRIPT="$REPO_ROOT/scripts/run_unified_v32_background.sh"
V33_TRAIN_SCRIPT="$REPO_ROOT/scripts/train_unified_v33_teacher.sh"
V33_BACKGROUND_SCRIPT="$REPO_ROOT/scripts/run_unified_v33_teacher_background.sh"
ISOLATED_RANK_SCRIPT="$REPO_ROOT/scripts/isolate_cuda_rank.sh"
V33_SMOKE_SCRIPT="$REPO_ROOT/scripts/smoke_unified_v33_teacher.sh"

# Do not let a caller's training receipt alter the launcher's default-profile
# tests. Dedicated cases below exercise inherited-receipt rejection.
unset D2R_TEACHER_CAPACITY_RECEIPT_B64
unset REQUIRE_TEACHER_CAPACITY_EVIDENCE
unset TEACHER_CAPACITY_POINTER

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_status() {
    local expected=$1 actual=$2 context=$3
    [[ "$actual" -eq "$expected" ]] || fail "$context: expected status $expected, got $actual"
}

wait_for_nonempty_file() {
    local path=$1 context=$2
    for _ in $(seq 1 100); do
        [[ -s "$path" ]] && return 0
        sleep 0.05
    done
    fail "$context: timed out waiting for $path"
}

assert_lock_available() {
    local path=$1 context=$2
    flock -n "$path" -c true || fail "$context: launcher lock remained held"
}

bash -n "$TRAIN_SCRIPT"
bash -n "$BACKGROUND_SCRIPT"
bash -n "$V32_TRAIN_SCRIPT"
bash -n "$V32_BACKGROUND_SCRIPT"
bash -n "$V33_TRAIN_SCRIPT"
bash -n "$V33_BACKGROUND_SCRIPT"
bash -n "$ISOLATED_RANK_SCRIPT"
bash -n "$V33_SMOKE_SCRIPT"

# Keep the v3.2 contract visible at its dedicated entrypoint.
grep -F 'export CONFIG="${CONFIG:-configs/unified_v3_2.yaml}"' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 config default changed"
grep -F 'export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_2}"' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 output default changed"
grep -F 'export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 seed default changed"
grep -F 'export EPOCHS="${EPOCHS:-32}"' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 epoch target changed"
grep -Fx 'export MIN_RESUME_COMPLETED_EPOCHS=20' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 minimum seed epoch changed"
grep -F 'export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"' "$V32_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.2 incomplete-seed override default changed"
grep -F 'export CONFIG="${CONFIG:-configs/unified_v3_2.yaml}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background config default changed"
grep -F 'export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_2}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background output default changed"
grep -F 'export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background seed default changed"
grep -F 'export EPOCHS="${EPOCHS:-32}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background epoch target changed"
grep -F 'export TRAIN_LOG_PREFIX="${TRAIN_LOG_PREFIX:-train_v32}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background log prefix changed"
grep -Fx 'export MIN_RESUME_COMPLETED_EPOCHS=20' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background minimum seed epoch changed"
grep -F 'export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"' "$V32_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.2 background incomplete-seed override default changed"

# v3.3 teacher-anchor has its own immutable profile floor and output/log tree.
grep -F 'export CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher config default changed"
grep -F 'export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_3_teacher_anchor}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher output default changed"
grep -F 'export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher seed default changed"
grep -F 'export EPOCHS="${EPOCHS:-32}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher epoch target changed"
grep -Fx 'export MIN_RESUME_COMPLETED_EPOCHS=20' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher minimum seed epoch changed"
grep -F 'export ALLOW_RESUME_BELOW_MIN_EPOCHS="${ALLOW_RESUME_BELOW_MIN_EPOCHS:-0}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher incomplete-seed override default changed"
grep -Fx 'export ISOLATE_CUDA_PER_RANK=1' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher no longer isolates one physical GPU per rank"
grep -Fx 'export REQUIRE_TEACHER_CAPACITY_EVIDENCE=1' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher no longer requires capacity evidence"
grep -F 'export TEACHER_CAPACITY_POINTER="${TEACHER_CAPACITY_POINTER:-runs/preflight_v33_teacher_capacity/approved.json}"' "$V33_TRAIN_SCRIPT" >/dev/null \
    || fail "v3.3 teacher capacity pointer default changed"
grep -F 'export CONFIG="${CONFIG:-configs/unified_v3_3_teacher_anchor.yaml}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background config default changed"
grep -F 'export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/d2r_v3_3_teacher_anchor}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background output default changed"
grep -F 'export SEED_RESUME="${SEED_RESUME:-runs/d2r_v3_1/unified/latest.pt}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background seed default changed"
grep -F 'export EPOCHS="${EPOCHS:-32}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background epoch target changed"
grep -F 'export TRAIN_LOG_PREFIX="${TRAIN_LOG_PREFIX:-train_v33_teacher_anchor}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background log prefix changed"
grep -Fx 'export MIN_RESUME_COMPLETED_EPOCHS=20' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background minimum seed epoch changed"
grep -Fx 'export ISOLATE_CUDA_PER_RANK=1' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background no longer isolates one physical GPU per rank"
grep -Fx 'export REQUIRE_TEACHER_CAPACITY_EVIDENCE=1' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background no longer requires capacity evidence"
grep -F 'export TEACHER_CAPACITY_POINTER="${TEACHER_CAPACITY_POINTER:-runs/preflight_v33_teacher_capacity/approved.json}"' "$V33_BACKGROUND_SCRIPT" >/dev/null \
    || fail "v3.3 teacher background capacity pointer default changed"

[[ 'runs/d2r_v3_3_teacher_anchor' != 'runs/d2r_v3_1' ]] \
    || fail "v3.3 teacher shares the v3.1 output root"
[[ 'runs/d2r_v3_3_teacher_anchor' != 'runs/d2r_v3_2' ]] \
    || fail "v3.3 teacher shares the v3.2 output root"

TEST_TMP="$(mktemp -d)"
cleanup() {
    local child_pid_file pid pid_file
    for pid_file in \
        "$TEST_TMP/background/unified/.launcher/train.pid" \
        "$TEST_TMP/background-v32/unified/.launcher/train.pid" \
        "$TEST_TMP/background-v33-teacher/unified/.launcher/train.pid" \
        "$TEST_TMP/preflight-term/unified/.launcher/train.pid" \
        "$TEST_TMP/training-term/unified/.launcher/train.pid"; do
        if [[ -f "$pid_file" ]]; then
            pid="$(sed -n 's/^pid=//p' "$pid_file" | head -n 1)"
            [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
        fi
    done
    for child_pid_file in \
        "$TEST_TMP/preflight-child.pid" \
        "$TEST_TMP/training-child.pid"; do
        if [[ -s "$child_pid_file" ]]; then
            pid="$(head -n 1 "$child_pid_file")"
            [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
        fi
    done
    rm -rf -- "$TEST_TMP"
}
trap cleanup EXIT

ENV_DUMP="$TEST_TMP/dump-rank-env"
cat >"$ENV_DUMP" <<'DUMP'
#!/usr/bin/env bash
set -euo pipefail
output="$1"
{
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'LOCAL_RANK=%s\n' "${LOCAL_RANK:-}"
    printf 'LOCAL_WORLD_SIZE=%s\n' "${LOCAL_WORLD_SIZE:-}"
    printf 'RANK=%s\n' "${RANK:-}"
    printf 'WORLD_SIZE=%s\n' "${WORLD_SIZE:-}"
    printf 'D2R_PHYSICAL_LOCAL_RANK=%s\n' "${D2R_PHYSICAL_LOCAL_RANK:-}"
    printf 'D2R_LAUNCH_CUDA_VISIBLE_DEVICES=%s\n' "${D2R_LAUNCH_CUDA_VISIBLE_DEVICES:-}"
    printf 'D2R_LAUNCH_LOCAL_WORLD_SIZE=%s\n' "${D2R_LAUNCH_LOCAL_WORLD_SIZE:-}"
} >"$output"
DUMP
chmod +x "$ENV_DUMP"

# The teacher rank wrapper must narrow the mask before Python starts while
# preserving torchrun's global rank/rendezvous environment.
rank_env="$TEST_TMP/rank-env.txt"
CUDA_VISIBLE_DEVICES='GPU-A, GPU-B' LOCAL_RANK=1 LOCAL_WORLD_SIZE=2 \
    RANK=1 WORLD_SIZE=2 \
    "$ISOLATED_RANK_SCRIPT" "$ENV_DUMP" "$rank_env" >/dev/null 2>&1
grep -Fx 'CUDA_VISIBLE_DEVICES=GPU-B' "$rank_env" >/dev/null \
    || fail "isolated worker selected the wrong physical GPU token"
grep -Fx 'LOCAL_RANK=0' "$rank_env" >/dev/null \
    || fail "isolated worker did not remap its only GPU to logical rank 0"
grep -Fx 'LOCAL_WORLD_SIZE=2' "$rank_env" >/dev/null \
    || fail "isolated worker changed torchrun's local worker count"
grep -Fx 'RANK=1' "$rank_env" >/dev/null \
    || fail "isolated worker changed the global rank"
grep -Fx 'WORLD_SIZE=2' "$rank_env" >/dev/null \
    || fail "isolated worker changed the global world size"
grep -Fx 'D2R_PHYSICAL_LOCAL_RANK=1' "$rank_env" >/dev/null \
    || fail "isolated worker did not retain its original local rank"
grep -Fx 'D2R_LAUNCH_CUDA_VISIBLE_DEVICES=GPU-A, GPU-B' "$rank_env" >/dev/null \
    || fail "isolated worker did not retain the original device mask"
grep -Fx 'D2R_LAUNCH_LOCAL_WORLD_SIZE=2' "$rank_env" >/dev/null \
    || fail "isolated worker did not retain the original local world size"

rank_env_unset="$TEST_TMP/rank-env-unset.txt"
env -u CUDA_VISIBLE_DEVICES LOCAL_RANK=3 LOCAL_WORLD_SIZE=4 \
    RANK=3 WORLD_SIZE=4 \
    "$ISOLATED_RANK_SCRIPT" "$ENV_DUMP" "$rank_env_unset" >/dev/null 2>&1
grep -Fx 'CUDA_VISIBLE_DEVICES=3' "$rank_env_unset" >/dev/null \
    || fail "isolated worker did not map an implicit visible index"

set +e
CUDA_VISIBLE_DEVICES='GPU-A,GPU-A' LOCAL_RANK=0 RANK=0 WORLD_SIZE=2 \
    "$ISOLATED_RANK_SCRIPT" "$ENV_DUMP" "$TEST_TMP/duplicate-rank.txt" \
    >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "duplicate isolated GPU token"
[[ ! -e "$TEST_TMP/duplicate-rank.txt" ]] \
    || fail "duplicate GPU mask executed the worker"

set +e
CUDA_VISIBLE_DEVICES='GPU-A,GPU-B' LOCAL_RANK=2 RANK=2 WORLD_SIZE=3 \
    "$ISOLATED_RANK_SCRIPT" "$ENV_DUMP" "$TEST_TMP/out-of-range-rank.txt" \
    >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "out-of-range isolated GPU token"
[[ ! -e "$TEST_TMP/out-of-range-rank.txt" ]] \
    || fail "out-of-range GPU mask executed the worker"

FAKE_PYTHON="$TEST_TMP/fake-python"
cat >"$FAKE_PYTHON" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    -c)
        printf 'D2R_CUDA %s %s\n' "${FAKE_CUDA_AVAILABLE:-1}" "${FAKE_CUDA_COUNT:-2}"
        ;;
    scripts/checkpoint_status.py)
        if [[ -n "${FAKE_CHECKPOINT_META:-}" ]]; then
            printf '%b\n' "$FAKE_CHECKPOINT_META"
        else
            printf 'unified\t9\t10\tline_epe\t4.492\n'
        fi
        ;;
    scripts/teacher_capacity_production.py)
        printf 'capacity %s\n' "$*" >>"${FAKE_CALLS:?}"
        if [[ "${FAKE_CAPACITY_OUTPUT+x}" == "x" ]]; then
            printf '%b' "$FAKE_CAPACITY_OUTPUT"
        else
            printf 'ZmFrZV9jYXBhY2l0eV9yZWNlaXB0\n'
        fi
        exit "${FAKE_CAPACITY_EXIT:-0}"
        ;;
    scripts/preflight_v31.py)
        printf 'preflight %s\n' "$*" >>"${FAKE_CALLS:?}"
        if [[ -n "${FAKE_PREFLIGHT_PID_FILE:-}" ]]; then
            printf '%s\n' "$$" >"$FAKE_PREFLIGHT_PID_FILE"
        fi
        if [[ "${FAKE_PREFLIGHT_SLEEP:-0}" != "0" ]]; then
            exec sleep "$FAKE_PREFLIGHT_SLEEP"
        fi
        exit "${FAKE_PREFLIGHT_EXIT:-0}"
        ;;
    -m)
        printf 'train %s\n' "$*" >>"${FAKE_CALLS:?}"
        printf 'train_receipt %s\n' "${D2R_TEACHER_CAPACITY_RECEIPT_B64-<unset>}" >>"${FAKE_CALLS:?}"
        if [[ -n "${FAKE_TRAIN_PID_FILE:-}" ]]; then
            printf '%s\n' "$$" >"$FAKE_TRAIN_PID_FILE"
        fi
        if [[ "${FAKE_TRAIN_SLEEP:-0}" != "0" ]]; then
            exec sleep "$FAKE_TRAIN_SLEEP"
        fi
        exit "${FAKE_TRAIN_EXIT:-0}"
        ;;
    *)
        printf 'unexpected fake-python args: %s\n' "$*" >&2
        exit 2
        ;;
esac
FAKE
chmod +x "$FAKE_PYTHON"
FAKE_CALLS="$TEST_TMP/calls.log"
touch "$FAKE_CALLS"
export NVIDIA_SMI=/bin/true

# No GPU: foreground and background must fail before creating output state/logs.
set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu" bash "$TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "foreground no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu" ]] || fail "foreground no-GPU guard created output files"

set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu-background" bash "$BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "background no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu-background" ]] || fail "background no-GPU guard created output files"

set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu-v32" bash "$V32_TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "v3.2 foreground no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu-v32" ]] || fail "v3.2 foreground no-GPU guard created output files"

set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu-background-v32" bash "$V32_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "v3.2 background no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu-background-v32" ]] || fail "v3.2 background no-GPU guard created output files"

set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu-v33-teacher" bash "$V33_TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "v3.3 teacher foreground no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu-v33-teacher" ]] || fail "v3.3 teacher foreground no-GPU guard created output files"

set +e
FAKE_CUDA_AVAILABLE=0 FAKE_CUDA_COUNT=0 PYTHON="$FAKE_PYTHON" \
    OUTPUT_ROOT="$TEST_TMP/no-gpu-background-v33-teacher" bash "$V33_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 69 "$status" "v3.3 teacher background no-GPU guard"
[[ ! -e "$TEST_TMP/no-gpu-background-v33-teacher" ]] || fail "v3.3 teacher background no-GPU guard created output files"

# A valid latest.pt must beat both RESUME and the v2 seed unless rollback is explicit.
LATEST_ROOT="$TEST_TMP/latest"
mkdir -p "$LATEST_ROOT/unified"
printf 'latest fixture\n' >"$LATEST_ROOT/unified/latest.pt"
printf 'seed fixture\n' >"$TEST_TMP/seed.pt"
CAPACITY_POINTER="$TEST_TMP/approved-capacity.json"
printf 'capacity pointer fixture\n' >"$CAPACITY_POINTER"

# v3.1/v3.2 retain their original no-capacity profile, including launch-only
# checks. Neither path may invoke the production verifier.
: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    CHECK_LAUNCH_ONLY=1 OUTPUT_ROOT="$TEST_TMP/v31-capacity-disabled" \
    SEED_RESUME="$TEST_TMP/seed.pt" bash "$TRAIN_SCRIPT" >/dev/null
grep -F 'capacity scripts/teacher_capacity_production.py verify' "$FAKE_CALLS" >/dev/null \
    && fail "v3.1 unexpectedly invoked the teacher-capacity verifier"
[[ ! -e "$TEST_TMP/v31-capacity-disabled" ]] \
    || fail "v3.1 launch-only check created output state"

: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    CHECK_LAUNCH_ONLY=1 FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    OUTPUT_ROOT="$TEST_TMP/v32-capacity-disabled" \
    SEED_RESUME="$TEST_TMP/seed.pt" bash "$V32_TRAIN_SCRIPT" >/dev/null
grep -F 'capacity scripts/teacher_capacity_production.py verify' "$FAKE_CALLS" >/dev/null \
    && fail "v3.2 unexpectedly invoked the teacher-capacity verifier"
[[ ! -e "$TEST_TMP/v32-capacity-disabled" ]] \
    || fail "v3.2 launch-only check created output state"

# A receipt from an unrelated parent environment must never leak into a
# capacity-disabled launch.
: >"$FAKE_CALLS"
set +e
D2R_TEACHER_CAPACITY_RECEIPT_B64=stale_receipt \
    PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    CHECK_LAUNCH_ONLY=1 OUTPUT_ROOT="$TEST_TMP/capacity-receipt-leak" \
    SEED_RESUME="$TEST_TMP/seed.pt" bash "$TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 79 "$status" "capacity-disabled inherited receipt rejection"
[[ ! -e "$TEST_TMP/capacity-receipt-leak" ]] \
    || fail "inherited receipt rejection created launcher state"

run_v33_capacity_failure() {
    local name=$1 mode=$2 root="$TEST_TMP/capacity-$1" pointer="$CAPACITY_POINTER"
    local -a fake_capacity_env=()
    case "$mode" in
        missing)
            pointer="$TEST_TMP/missing-capacity-pointer.json"
            ;;
        nonzero)
            fake_capacity_env+=(FAKE_CAPACITY_EXIT=23)
            ;;
        empty)
            fake_capacity_env+=(FAKE_CAPACITY_OUTPUT=)
            ;;
        multiline)
            fake_capacity_env+=("FAKE_CAPACITY_OUTPUT=ZmFrZQ\nYmFk\n")
            ;;
        illegal)
            fake_capacity_env+=(FAKE_CAPACITY_OUTPUT=ZmFrZQ==)
            ;;
        *)
            fail "unknown capacity failure mode: $mode"
            ;;
    esac
    : >"$FAKE_CALLS"
    set +e
    env -u FAKE_CAPACITY_EXIT -u FAKE_CAPACITY_OUTPUT \
        "${fake_capacity_env[@]}" \
        PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
        FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
        OUTPUT_ROOT="$root" SEED_RESUME="$TEST_TMP/seed.pt" \
        TEACHER_CAPACITY_POINTER="$pointer" \
        bash "$V33_TRAIN_SCRIPT" >"$TEST_TMP/capacity-$name.out" 2>&1
    status=$?
    set -e
    assert_status 79 "$status" "v3.3 capacity $name fail-closed gate"
    [[ ! -e "$root" ]] || fail "capacity $name failure created launcher state/logs"
    grep -E '^(preflight|train) ' "$FAKE_CALLS" >/dev/null \
        && fail "capacity $name failure reached preflight or training"
    return 0
}

# Every malformed production-verifier outcome fails before state/log/training.
# RUN_PREFLIGHT=0 is deliberate: the capacity gate is independent of it.
run_v33_capacity_failure missing missing
run_v33_capacity_failure nonzero nonzero
run_v33_capacity_failure empty empty
run_v33_capacity_failure multiline multiline
run_v33_capacity_failure illegal illegal

# The detached wrapper must also fail during its read-only launch check, before
# it creates its state directory or a detached log.
: >"$FAKE_CALLS"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    OUTPUT_ROOT="$TEST_TMP/capacity-missing-background" \
    LOG_DIR="$TEST_TMP/capacity-missing-background-logs" \
    SEED_RESUME="$TEST_TMP/seed.pt" \
    TEACHER_CAPACITY_POINTER="$TEST_TMP/missing-capacity-pointer.json" \
    bash "$V33_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 79 "$status" "v3.3 background missing capacity pointer"
[[ ! -e "$TEST_TMP/capacity-missing-background" \
    && ! -e "$TEST_TMP/capacity-missing-background-logs" ]] \
    || fail "background capacity failure created state or logs"

# CHECK_LAUNCH_ONLY still authenticates the selected resume, but publishes no
# launcher state and never reaches torchrun.
: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    CHECK_LAUNCH_ONLY=1 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    OUTPUT_ROOT="$TEST_TMP/capacity-launch-only" \
    SEED_RESUME="$TEST_TMP/seed.pt" \
    TEACHER_CAPACITY_POINTER="$CAPACITY_POINTER" \
    bash "$V33_TRAIN_SCRIPT" >/dev/null
grep -F "capacity scripts/teacher_capacity_production.py verify --config configs/unified_v3_3_teacher_anchor.yaml --pointer $CAPACITY_POINTER --resume $TEST_TMP/seed.pt" "$FAKE_CALLS" >/dev/null \
    || fail "CHECK_LAUNCH_ONLY did not call the fixed production verifier CLI"
grep -E '^(preflight|train) ' "$FAKE_CALLS" >/dev/null \
    && fail "CHECK_LAUNCH_ONLY reached preflight or training"
[[ ! -e "$TEST_TMP/capacity-launch-only" ]] \
    || fail "capacity launch-only check created launcher state"

# v3.2 must not bootstrap from the currently incomplete v3.1 epoch-10 seed.
# The refusal happens before any launcher directory/log/PID/lock is created.
INCOMPLETE_FG_ROOT="$TEST_TMP/v32-incomplete-foreground"
: >"$FAKE_CALLS"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t9\t10\tline_epe\t4.492' \
    MIN_RESUME_COMPLETED_EPOCHS=0 \
    OUTPUT_ROOT="$INCOMPLETE_FG_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    bash "$V32_TRAIN_SCRIPT" >"$TEST_TMP/v32-incomplete-foreground.out" 2>&1
status=$?
set -e
assert_status 78 "$status" "v3.2 incomplete foreground seed gate"
[[ ! -e "$INCOMPLETE_FG_ROOT" ]] || fail "incomplete v3.2 foreground seed created launcher state"
grep -F 'ALLOW_RESUME_BELOW_MIN_EPOCHS=1' "$TEST_TMP/v32-incomplete-foreground.out" >/dev/null \
    || fail "v3.2 incomplete seed error omitted the explicit override"
grep -F -- 'train -m torch.distributed.run' "$FAKE_CALLS" >/dev/null \
    && fail "incomplete v3.2 foreground seed launched training"

INCOMPLETE_BG_ROOT="$TEST_TMP/v32-incomplete-background"
: >"$FAKE_CALLS"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t9\t10\tline_epe\t4.492' \
    OUTPUT_ROOT="$INCOMPLETE_BG_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    RESUME="$TEST_TMP/seed.pt" \
    bash "$V32_BACKGROUND_SCRIPT" >"$TEST_TMP/v32-incomplete-background.out" 2>&1
status=$?
set -e
assert_status 78 "$status" "v3.2 incomplete background seed gate"
[[ ! -e "$INCOMPLETE_BG_ROOT" ]] || fail "incomplete v3.2 background seed created launcher state"
grep -F -- 'train -m torch.distributed.run' "$FAKE_CALLS" >/dev/null \
    && fail "incomplete v3.2 background seed launched training"

# The escape hatch is deliberately verbose and opt-in.
OVERRIDE_ROOT="$TEST_TMP/v32-incomplete-override"
: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t9\t10\tline_epe\t4.492' \
    ALLOW_RESUME_BELOW_MIN_EPOCHS=1 \
    OUTPUT_ROOT="$OVERRIDE_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    bash "$V32_TRAIN_SCRIPT" >/dev/null 2>&1
grep -F -- 'train -m torch.distributed.run' "$FAKE_CALLS" >/dev/null \
    || fail "explicit incomplete-seed override did not reach training"

# v3.3 teacher-anchor inherits the same checkpoint-first gate in both modes.
V33_INCOMPLETE_FG_ROOT="$TEST_TMP/v33-teacher-incomplete-foreground"
: >"$FAKE_CALLS"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t9\t10\tline_epe\t4.492' \
    MIN_RESUME_COMPLETED_EPOCHS=0 \
    OUTPUT_ROOT="$V33_INCOMPLETE_FG_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    bash "$V33_TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 78 "$status" "v3.3 teacher incomplete foreground seed gate"
[[ ! -e "$V33_INCOMPLETE_FG_ROOT" ]] || fail "incomplete v3.3 teacher foreground seed created launcher state"
grep -F -- 'train -m torch.distributed.run' "$FAKE_CALLS" >/dev/null \
    && fail "incomplete v3.3 teacher foreground seed launched training"

V33_INCOMPLETE_BG_ROOT="$TEST_TMP/v33-teacher-incomplete-background"
: >"$FAKE_CALLS"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t9\t10\tline_epe\t4.492' \
    OUTPUT_ROOT="$V33_INCOMPLETE_BG_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    bash "$V33_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 78 "$status" "v3.3 teacher incomplete background seed gate"
[[ ! -e "$V33_INCOMPLETE_BG_ROOT" ]] || fail "incomplete v3.3 teacher background seed created launcher state"
grep -F -- 'train -m torch.distributed.run' "$FAKE_CALLS" >/dev/null \
    && fail "incomplete v3.3 teacher background seed launched training"

# The dedicated foreground wrapper injects v3.2 defaults into the shared
# implementation. Redirect only launcher state to tmp so the default output
# root can be asserted without writing into it.
: >"$FAKE_CALLS"
env -u CONFIG -u OUTPUT_ROOT -u SEED_RESUME -u EPOCHS \
    -u MIN_RESUME_COMPLETED_EPOCHS -u ALLOW_RESUME_BELOW_MIN_EPOCHS \
    PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    RESUME="$TEST_TMP/seed.pt" \
    OUTPUT_STAGE_DIR="$TEST_TMP/v32-default-stage" \
    STATE_DIR="$TEST_TMP/v32-default-state" \
    bash "$V32_TRAIN_SCRIPT" >/dev/null
grep -F -- "--config configs/unified_v3_2.yaml" "$FAKE_CALLS" >/dev/null \
    || fail "v3.2 wrapper did not select its config"
grep -F -- "--output-dir runs/d2r_v3_2" "$FAKE_CALLS" >/dev/null \
    || fail "v3.2 wrapper did not select its output root"
grep -F -- "--epochs 32" "$FAKE_CALLS" >/dev/null \
    || fail "v3.2 wrapper did not preserve its total epoch target"

# Verify the v3.3 profile defaults at the actual shared train invocation while
# redirecting launcher state to tmp.
: >"$FAKE_CALLS"
env -u CONFIG -u OUTPUT_ROOT -u SEED_RESUME -u EPOCHS \
    -u MIN_RESUME_COMPLETED_EPOCHS -u ALLOW_RESUME_BELOW_MIN_EPOCHS \
    PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    TEACHER_CAPACITY_POINTER="$CAPACITY_POINTER" \
    RESUME="$TEST_TMP/seed.pt" \
    OUTPUT_STAGE_DIR="$TEST_TMP/v33-default-stage" \
    STATE_DIR="$TEST_TMP/v33-default-state" \
    bash "$V33_TRAIN_SCRIPT" >/dev/null
grep -F -- "--config configs/unified_v3_3_teacher_anchor.yaml" "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher wrapper did not select its config"
grep -F -- "--output-dir runs/d2r_v3_3_teacher_anchor" "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher wrapper did not select its output root"
grep -F -- "--epochs 32" "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher wrapper did not preserve its total epoch target"
grep -F -- "--no_python $ISOLATED_RANK_SCRIPT $FAKE_PYTHON -m diffusion2raft.train" "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher did not launch through the pre-Python CUDA isolation wrapper"
grep -F -- "--max_restarts=0" "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher torchrun restart policy changed"
grep -Fx 'train_receipt ZmFrZV9jYXBhY2l0eV9yZWNlaXB0' "$FAKE_CALLS" >/dev/null \
    || fail "v3.3 teacher training did not inherit the verified capacity receipt"

: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    OUTPUT_ROOT="$LATEST_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    RESUME="$TEST_TMP/seed.pt" EPOCHS=20 bash "$TRAIN_SCRIPT" >/dev/null
grep -F -- "--resume $LATEST_ROOT/unified/latest.pt" "$FAKE_CALLS" >/dev/null \
    || fail "training did not resume latest.pt"
grep -F -- "--epochs 20" "$FAKE_CALLS" >/dev/null \
    || fail "training did not preserve total epoch target"
grep -F -- "--output-dir $LATEST_ROOT" "$FAKE_CALLS" >/dev/null \
    || fail "training output directory diverges from resume directory"

# Already-complete checkpoint is a clean no-op.
: >"$FAKE_CALLS"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    OUTPUT_ROOT="$LATEST_ROOT" EPOCHS=20 bash "$TRAIN_SCRIPT" >/dev/null
grep -F -- "train -m torch.distributed.run" "$FAKE_CALLS" >/dev/null \
    && fail "already-complete checkpoint launched training"

# A corrupt/empty latest.pt must fail closed instead of falling back to the seed.
CORRUPT_ROOT="$TEST_TMP/corrupt"
mkdir -p "$CORRUPT_ROOT/unified"
: >"$CORRUPT_ROOT/unified/latest.pt"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    OUTPUT_ROOT="$CORRUPT_ROOT" SEED_RESUME="$TEST_TMP/seed.pt" \
    bash "$TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 66 "$status" "empty latest fail-closed"

# A training-agent failure must propagate unchanged and release every piece of
# launcher state needed by a later retry.
FAILED_ROOT="$TEST_TMP/failed-training"
mkdir -p "$FAILED_ROOT/unified"
printf 'latest fixture\n' >"$FAILED_ROOT/unified/latest.pt"
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" FAKE_TRAIN_EXIT=42 \
    RUN_PREFLIGHT=0 OUTPUT_ROOT="$FAILED_ROOT" \
    bash "$TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 42 "$status" "training-agent failure propagation"
FAILED_PID_FILE="$FAILED_ROOT/unified/.launcher/train.pid"
FAILED_STATUS_FILE="$FAILED_ROOT/unified/.launcher/last_exit.status"
[[ ! -e "$FAILED_PID_FILE" ]] || fail "failed training left a launcher PID file"
grep -Fx 'exit_code=42' "$FAILED_STATUS_FILE" >/dev/null \
    || fail "failed training did not record its exit code"
assert_lock_available "$FAILED_ROOT/unified/.launcher/train.lock" \
    "failed training cleanup"

# The PID file is published before preflight so detached callers can observe
# progress. TERM sent to that PID must therefore also stop and reap preflight.
PREFLIGHT_TERM_ROOT="$TEST_TMP/preflight-term"
mkdir -p "$PREFLIGHT_TERM_ROOT/unified"
printf 'latest fixture\n' >"$PREFLIGHT_TERM_ROOT/unified/latest.pt"
PREFLIGHT_CHILD_PID_FILE="$TEST_TMP/preflight-child.pid"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" \
    FAKE_PREFLIGHT_SLEEP=30 FAKE_PREFLIGHT_PID_FILE="$PREFLIGHT_CHILD_PID_FILE" \
    RUN_PREFLIGHT=1 OUTPUT_ROOT="$PREFLIGHT_TERM_ROOT" \
    bash "$TRAIN_SCRIPT" >/dev/null 2>&1 &
preflight_launcher_pid=$!
PREFLIGHT_PID_FILE="$PREFLIGHT_TERM_ROOT/unified/.launcher/train.pid"
PREFLIGHT_STATUS_FILE="$PREFLIGHT_TERM_ROOT/unified/.launcher/last_exit.status"
wait_for_nonempty_file "$PREFLIGHT_PID_FILE" "preflight TERM launcher PID"
wait_for_nonempty_file "$PREFLIGHT_CHILD_PID_FILE" "preflight TERM child PID"
published_pid="$(sed -n 's/^pid=//p' "$PREFLIGHT_PID_FILE" | head -n 1)"
[[ "$published_pid" == "$preflight_launcher_pid" ]] \
    || fail "preflight TERM published the wrong launcher PID"
kill -TERM "$published_pid"
set +e
wait "$preflight_launcher_pid"
status=$?
set -e
assert_status 143 "$status" "preflight TERM propagation"
preflight_child_pid="$(head -n 1 "$PREFLIGHT_CHILD_PID_FILE")"
if kill -0 "$preflight_child_pid" 2>/dev/null; then
    fail "preflight child survived launcher TERM"
fi
[[ ! -e "$PREFLIGHT_PID_FILE" ]] || fail "preflight TERM left a launcher PID file"
grep -Fx 'exit_code=143' "$PREFLIGHT_STATUS_FILE" >/dev/null \
    || fail "preflight TERM did not record signal exit code"
assert_lock_available "$PREFLIGHT_TERM_ROOT/unified/.launcher/train.lock" \
    "preflight TERM cleanup"

# Apply the same contract to the torchrun agent: TERM is forwarded, the child
# is reaped, status is recorded, and the writer lock is immediately reusable.
TRAINING_TERM_ROOT="$TEST_TMP/training-term"
mkdir -p "$TRAINING_TERM_ROOT/unified"
printf 'latest fixture\n' >"$TRAINING_TERM_ROOT/unified/latest.pt"
TRAINING_CHILD_PID_FILE="$TEST_TMP/training-child.pid"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" \
    FAKE_TRAIN_SLEEP=30 FAKE_TRAIN_PID_FILE="$TRAINING_CHILD_PID_FILE" \
    RUN_PREFLIGHT=0 OUTPUT_ROOT="$TRAINING_TERM_ROOT" \
    bash "$TRAIN_SCRIPT" >/dev/null 2>&1 &
training_launcher_pid=$!
TRAINING_PID_FILE="$TRAINING_TERM_ROOT/unified/.launcher/train.pid"
TRAINING_STATUS_FILE="$TRAINING_TERM_ROOT/unified/.launcher/last_exit.status"
wait_for_nonempty_file "$TRAINING_PID_FILE" "training TERM launcher PID"
wait_for_nonempty_file "$TRAINING_CHILD_PID_FILE" "training TERM child PID"
published_pid="$(sed -n 's/^pid=//p' "$TRAINING_PID_FILE" | head -n 1)"
[[ "$published_pid" == "$training_launcher_pid" ]] \
    || fail "training TERM published the wrong launcher PID"
kill -TERM "$published_pid"
set +e
wait "$training_launcher_pid"
status=$?
set -e
assert_status 143 "$status" "training TERM propagation"
training_child_pid="$(head -n 1 "$TRAINING_CHILD_PID_FILE")"
if kill -0 "$training_child_pid" 2>/dev/null; then
    fail "training child survived launcher TERM"
fi
[[ ! -e "$TRAINING_PID_FILE" ]] || fail "training TERM left a launcher PID file"
grep -Fx 'exit_code=143' "$TRAINING_STATUS_FILE" >/dev/null \
    || fail "training TERM did not record signal exit code"
assert_lock_available "$TRAINING_TERM_ROOT/unified/.launcher/train.lock" \
    "training TERM cleanup"

# Holding the shared flock must reject a second writer.
LOCK_ROOT="$TEST_TMP/locked"
mkdir -p "$LOCK_ROOT/unified/.launcher"
printf 'latest fixture\n' >"$LOCK_ROOT/unified/latest.pt"
flock "$LOCK_ROOT/unified/.launcher/train.lock" -c 'sleep 5' &
lock_holder=$!
sleep 0.1
set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    OUTPUT_ROOT="$LOCK_ROOT" bash "$TRAIN_SCRIPT" >/dev/null 2>&1
status=$?
set -e
kill "$lock_holder" 2>/dev/null || true
wait "$lock_holder" 2>/dev/null || true
assert_status 75 "$status" "duplicate writer lock"

# Detached launcher publishes pid/log, holds the lock, then records clean exit.
BACKGROUND_ROOT="$TEST_TMP/background"
mkdir -p "$BACKGROUND_ROOT/unified"
printf 'latest fixture\n' >"$BACKGROUND_ROOT/unified/latest.pt"
background_output="$TEST_TMP/background.out"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" FAKE_TRAIN_SLEEP=4 \
    RUN_PREFLIGHT=0 OUTPUT_ROOT="$BACKGROUND_ROOT" \
    bash "$BACKGROUND_SCRIPT" >"$background_output"
PID_FILE="$BACKGROUND_ROOT/unified/.launcher/train.pid"
STATUS_FILE="$BACKGROUND_ROOT/unified/.launcher/last_exit.status"
[[ -s "$PID_FILE" ]] || fail "background launcher did not publish pid"
LOG_FILE="$(sed -n 's/^log=//p' "$PID_FILE" | head -n 1)"
[[ -n "$LOG_FILE" && -f "$LOG_FILE" ]] || fail "background launcher did not create log"
[[ "$(basename "$LOG_FILE")" == train_v31_* ]] || fail "v3.1 log prefix changed"

set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    OUTPUT_ROOT="$BACKGROUND_ROOT" bash "$BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 75 "$status" "background duplicate writer lock"

# v3.2 may run alongside v3.1 because its default output/state tree is
# independent, and its log is unambiguously labelled v32.
V32_BACKGROUND_ROOT="$TEST_TMP/background-v32"
mkdir -p "$V32_BACKGROUND_ROOT/unified"
printf 'latest fixture\n' >"$V32_BACKGROUND_ROOT/unified/latest.pt"
v32_background_output="$TEST_TMP/background-v32.out"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" FAKE_TRAIN_SLEEP=4 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    RUN_PREFLIGHT=0 OUTPUT_ROOT="$V32_BACKGROUND_ROOT" \
    bash "$V32_BACKGROUND_SCRIPT" >"$v32_background_output"
V32_PID_FILE="$V32_BACKGROUND_ROOT/unified/.launcher/train.pid"
V32_STATUS_FILE="$V32_BACKGROUND_ROOT/unified/.launcher/last_exit.status"
[[ -s "$V32_PID_FILE" ]] || fail "v3.2 background launcher did not publish pid"
V32_LOG_FILE="$(sed -n 's/^log=//p' "$V32_PID_FILE" | head -n 1)"
[[ -n "$V32_LOG_FILE" && -f "$V32_LOG_FILE" ]] || fail "v3.2 background launcher did not create log"
[[ "$(basename "$V32_LOG_FILE")" == train_v32_* ]] || fail "v3.2 log is not clearly labelled"
[[ "$V32_LOG_FILE" != "$LOG_FILE" ]] || fail "v3.1 and v3.2 shared a log file"

set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    OUTPUT_ROOT="$V32_BACKGROUND_ROOT" bash "$V32_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 75 "$status" "v3.2 background duplicate writer lock"

# v3.3 teacher-anchor has a third independent lock/status/log tree.
V33_BACKGROUND_ROOT="$TEST_TMP/background-v33-teacher"
mkdir -p "$V33_BACKGROUND_ROOT/unified"
printf 'latest fixture\n' >"$V33_BACKGROUND_ROOT/unified/latest.pt"
v33_background_output="$TEST_TMP/background-v33-teacher.out"
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" FAKE_TRAIN_SLEEP=4 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    RUN_PREFLIGHT=0 OUTPUT_ROOT="$V33_BACKGROUND_ROOT" \
    TEACHER_CAPACITY_POINTER="$CAPACITY_POINTER" \
    bash "$V33_BACKGROUND_SCRIPT" >"$v33_background_output"
V33_PID_FILE="$V33_BACKGROUND_ROOT/unified/.launcher/train.pid"
V33_STATUS_FILE="$V33_BACKGROUND_ROOT/unified/.launcher/last_exit.status"
[[ -s "$V33_PID_FILE" ]] || fail "v3.3 teacher background launcher did not publish pid"
V33_LOG_FILE="$(sed -n 's/^log=//p' "$V33_PID_FILE" | head -n 1)"
[[ -n "$V33_LOG_FILE" && -f "$V33_LOG_FILE" ]] || fail "v3.3 teacher background launcher did not create log"
[[ "$(basename "$V33_LOG_FILE")" == train_v33_teacher_anchor_* ]] || fail "v3.3 teacher log is not clearly labelled"
[[ "$V33_LOG_FILE" != "$LOG_FILE" && "$V33_LOG_FILE" != "$V32_LOG_FILE" ]] \
    || fail "v3.3 teacher shared a v3.1/v3.2 log file"

set +e
PYTHON="$FAKE_PYTHON" FAKE_CALLS="$FAKE_CALLS" RUN_PREFLIGHT=0 \
    FAKE_CHECKPOINT_META='unified\t19\t20\tline_epe\t4.0' \
    TEACHER_CAPACITY_POINTER="$CAPACITY_POINTER" \
    OUTPUT_ROOT="$V33_BACKGROUND_ROOT" bash "$V33_BACKGROUND_SCRIPT" >/dev/null 2>&1
status=$?
set -e
assert_status 75 "$status" "v3.3 teacher background duplicate writer lock"

for _ in $(seq 1 60); do
    [[ -f "$STATUS_FILE" && ! -e "$PID_FILE" ]] && break
    sleep 0.1
done
[[ ! -e "$PID_FILE" ]] || fail "background pid file was not cleaned"
grep -Fx 'exit_code=0' "$STATUS_FILE" >/dev/null \
    || fail "background completion status is not success"

for _ in $(seq 1 60); do
    [[ -f "$V32_STATUS_FILE" && ! -e "$V32_PID_FILE" ]] && break
    sleep 0.1
done
[[ ! -e "$V32_PID_FILE" ]] || fail "v3.2 background pid file was not cleaned"
grep -Fx 'exit_code=0' "$V32_STATUS_FILE" >/dev/null \
    || fail "v3.2 background completion status is not success"

for _ in $(seq 1 60); do
    [[ -f "$V33_STATUS_FILE" && ! -e "$V33_PID_FILE" ]] && break
    sleep 0.1
done
[[ ! -e "$V33_PID_FILE" ]] || fail "v3.3 teacher background pid file was not cleaned"
grep -Fx 'exit_code=0' "$V33_STATUS_FILE" >/dev/null \
    || fail "v3.3 teacher background completion status is not success"

echo "training launcher tests: PASS"
