#!/usr/bin/env bash
# --------------------------------------------------------------
# Run all ablation sweeps across tasks 2, 9 (and optionally 6).
#
# Usage:
#   ./run_ablations.sh                      # run all ablations (tasks 2, 9)
#   ./run_ablations.sh --with-task6         # include task 6
#   ./run_ablations.sh --with-task6 penalty # include task 6, one group
#   ./run_ablations.sh dueling double       # run specific groups
# --------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARNER="${SCRIPT_DIR}/minecraft/dqn_online_learner.py"
PYTHON="${PYTHON:-python3}"

declare -A TASK_ALGO
TASK_ALGO["task2"]="t2_baseline"
TASK_ALGO["task6"]="t6_baseline"
TASK_ALGO["task9"]="t9_baseline"

TASK_LIST=(task2 task6 task9)

# Parse --no-task6 flag
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-task6" ]]; then
        TASK_LIST=(task2 task9)
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

mkdir -p experiments

run_sweep() {
    local name="$1"
    local task="$2"
    local algo="${TASK_ALGO[$task]}"
    shift 2
    echo ""
    echo "--- Ablation: ${name} | ${task} | algo=${algo} ---"
    local logfile="experiments/ablation_${name}_${task}_$(date +%Y%m%d_%H%M%S).log"
    ${PYTHON} "${LEARNER}" --multirun "task=${task}" "algo=${algo}" "$@" 2>&1 | tee -a "${logfile}"
    return "${PIPESTATUS[0]}"
}

run_penalty() {
    local rc=0
    for task in "${TASK_LIST[@]}"; do
        run_sweep "penalty" "$task" \
            "algo.step_penalty=0.0" \
            "experiment_name=ablate_penalty_\${task.task_num}_\${algo.step_penalty}" \
            || rc=1
    done
    return $rc
}

run_objects() {
    local rc=0
    for task in "${TASK_LIST[@]}"; do
        run_sweep "objects" "$task" \
            "env=full_objects" \
            "experiment_name=ablate_objects_\${task.task_num}_\${env.only_needed_objects}_\${env.vanishing}" \
            || rc=1
    done
    return $rc
}

run_dueling() {
    local rc=0
    for task in "${TASK_LIST[@]}"; do
        run_sweep "dueling" "$task" \
            "algo.dueling_dqn=false" \
            "experiment_name=ablate_dueling_\${task.task_num}_\${algo.dueling_dqn}" \
            || rc=1
    done
    return $rc
}

run_double() {
    local rc=0
    for task in "${TASK_LIST[@]}"; do
        run_sweep "double" "$task" \
            "algo.double_dqn=false" \
            "experiment_name=ablate_double_\${task.task_num}_\${algo.double_dqn}" \
            || rc=1
    done
    return $rc
}

ALL_RUN_GROUPS=(penalty objects dueling double)

if [[ $# -eq 0 ]]; then
    RUN_GROUPS=("${ALL_RUN_GROUPS[@]}")
else
    RUN_GROUPS=("$@")
fi

FAILED=()
SUCCEEDED=()

echo "Tasks: ${TASK_LIST[*]}"
echo "Groups: ${RUN_GROUPS[*]}"

for group in "${RUN_GROUPS[@]}"; do
    case "$group" in
        penalty) if run_penalty; then SUCCEEDED+=("$group"); else FAILED+=("$group"); fi ;;
        objects) if run_objects; then SUCCEEDED+=("$group"); else FAILED+=("$group"); fi ;;
        dueling) if run_dueling; then SUCCEEDED+=("$group"); else FAILED+=("$group"); fi ;;
        double)  if run_double;  then SUCCEEDED+=("$group"); else FAILED+=("$group"); fi ;;
        *)       echo "[WARN] Unknown group: $group. Options: ${ALL_RUN_GROUPS[*]}"
                 FAILED+=("$group") ;;
    esac
done

echo ""
echo "------------------------------------------------------"
echo " Ablation Summary"
echo "------------------------------------------------------"

tasks_per_group=${#TASK_LIST[@]}
total_runs=$(( ${#RUN_GROUPS[@]} * tasks_per_group ))

echo " Tasks: ${TASK_LIST[*]}"
echo " Total runs: ${total_runs}"
[[ ${#SUCCEEDED[@]} -gt 0 ]] && echo " Succeeded: ${SUCCEEDED[*]}"
[[ ${#FAILED[@]} -gt 0 ]]    && echo " FAILED:    ${FAILED[*]}" && exit 1
echo " All ablations completed."