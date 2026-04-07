#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Run ablation experiments via Hydra.
#
# Usage:
#   ./run_all.sh                        # run all experiment presets
#   ./run_all.sh task2_baseline         # run one preset
#   ./run_all.sh --grid                 # full grid sweep
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARNER="${SCRIPT_DIR}/minecraft/dqn_online_learner.py"
PYTHON="${PYTHON:-python3}"

# ── Experiment presets (one per file in configs/experiment/) ──
PRESET_DIR="${SCRIPT_DIR}/configs/experiment"

run_preset() {
    local name="$1"
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Running preset: ${name}"
    echo "═══════════════════════════════════════════════════════"
    ${PYTHON} "${LEARNER}" +experiment="${name}" 2>&1 | tee \
        "experiments/${name}_$(date +%Y%m%d_%H%M%S).log"
}

run_grid() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Running full grid sweep"
    echo "═══════════════════════════════════════════════════════"
    ${PYTHON} "${LEARNER}" --multirun \
        task=task2,task3 \
        env=default,vanishing \
        algo=baseline \
        dfa=default,disabled \
        experiment_name='grid_${task}_${env}_${dfa}'
}

# ── Parse args ───────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    # Run all presets sequentially
    FAILED=()
    SUCCEEDED=()
    for preset_file in "${PRESET_DIR}"/*.yaml; do
        name="$(basename "${preset_file}" .yaml)"
        if run_preset "${name}"; then
            SUCCEEDED+=("${name}")
        else
            FAILED+=("${name}")
        fi
    done

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " Summary"
    echo "═══════════════════════════════════════════════════════"
    [[ ${#SUCCEEDED[@]} -gt 0 ]] && echo " Succeeded: ${SUCCEEDED[*]}"
    [[ ${#FAILED[@]} -gt 0 ]] && echo " FAILED:    ${FAILED[*]}" && exit 1
    echo " All experiments completed."

elif [[ "$1" == "--grid" ]]; then
    run_grid

else
    # Run specific presets
    for name in "$@"; do
        run_preset "${name}"
    done
fi