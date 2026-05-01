#!/bin/bash

set -e

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="main_fed_config.py"
LOG_DIR="${PROJECT_PATH}/experiment_logs"
RESULTS_DIR="${PROJECT_PATH}/dataset/models/iomt_traffic/Category"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_LOG="${LOG_DIR}/comprehensive_experiments_${TIMESTAMP}.log"

# ========================================================================
# NEW: Fixed heterogeneity experiment matrix for E1..E6
# ========================================================================
# Experiment matrix:
#   E1: balanced  / alpha=1.0
#   E2: mild      / alpha=1.0
#   E3: severe    / alpha=1.0
#   E4: balanced  / alpha=0.1
#   E5: severe    / alpha=0.1
#   E6: extreme   / alpha=0.1
EXPERIMENT_NAMES=(E1 E2 E3 E4 E5 E6)
IMBALANCE_LABELS=(balanced mild severe balanced severe extreme)
DIRICHLET_GROUPS=(alpha_1_0 alpha_1_0 alpha_1_0 alpha_0_1 alpha_0_1 alpha_0_1)

# Number of clients to run for each experiment.
NUM_CLIENTS=(5 10)

# Strategies to use for the matrix experiments.
STRATEGIES=(FedCRA FedProx FedAvg FLAME)

NUM_ROUNDS=100

mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_PATH}/tmp"

# Activate virtual environment
# source "${PROJECT_PATH}/venv/bin/activate"

export HYDRA_FULL_ERROR=1
export SAVE_CLIENT_CHECKPOINTS=false
export RAY_TMPDIR="${PROJECT_PATH}/tmp"

{
    echo "=========================================================================="
    echo "Comprehensive Federated Learning Experiments"
    echo "Matrix: E1..E6 with balanced/mild/severe/extreme + alpha 1.0 / 0.1"
    echo "=========================================================================="
    echo "Start Time: $(date)"
    echo "Project Path: ${PROJECT_PATH}"
    echo "Results Directory: ${RESULTS_DIR}"
    echo ""
    echo "Strategies        : ${STRATEGIES[*]}"
    echo "Experiments       : ${EXPERIMENT_NAMES[*]}"
    echo "Imbalance labels  : ${IMBALANCE_LABELS[*]}"
    echo "Dirichlet groups  : ${DIRICHLET_GROUPS[*]}"
    echo "Clients           : ${NUM_CLIENTS[@]}"
    echo "Rounds            : ${NUM_ROUNDS}"
    echo ""
    echo "Total Experiments : $((${#EXPERIMENT_NAMES[@]} * ${#NUM_CLIENTS[@]} * ${#STRATEGIES[@]}))"
    echo "=========================================================================="
} | tee "${EXPERIMENT_LOG}"

TOTAL_EXPERIMENTS=$((${#EXPERIMENT_NAMES[@]} * ${#NUM_CLIENTS[@]} * ${#STRATEGIES[@]}))
CURRENT_EXPERIMENT=0

# ========================================================================
# FedCRA hyperparameters per alpha value
# Tuning Strategy:
#   - α=0.1 (Extreme Heterogeneity): ADAPTIVE λ_cra schedule prevents anchor collapse
#   - α≥0.3 (Moderate+): Scaled-down adaptive λ_cra schedule with proportional values
# 
# CONFIGURABLE ADAPTIVE λ_cra SCHEDULE (set via strategy params):
#   Rounds 1-5:   λ_cra = lambda_cra_initial  → Let model learn, anchors initialize well
#   Rounds 6-15:  λ_cra = lambda_cra_medium   → Anchors stabilize, gradual increase  
#   Rounds 16+:   λ_cra = lambda_cra_base     → Mature anchors, full strength
#
# Recommended defaults (scalable across α values):
#   α=0.1  (extreme): initial=0.10, medium=0.18, base=0.32
#   α=0.3  (moderate): initial=0.06, medium=0.12, base=0.18
#   α=0.5  (moderate-low): initial=0.03, medium=0.06, base=0.09
#   α=1.0  (balanced): initial=0.02, medium=0.04, base=0.06
# ========================================================================
get_fedcra_overrides() {
    local alpha=$1

    case "${alpha}" in
    "0.1")
        # EXTREME HETEROGENEITY: Balanced CRA strength for stability
        # Moderate lambda_cra to avoid overfitting to noisy anchors
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.003 \
        ++strategy.params.lambda_cra=0.35 \
        ++strategy.params.lambda_cra_initial=0.12 \
        ++strategy.params.lambda_cra_medium=0.22 \
        ++strategy.params.lambda_cra_base=0.35 \
        ++strategy.params.embedding_dim=128 \
        ++strategy.params.use_class_penalty=true \
        ++strategy.params.use_anchor_alignment=true \
        ++config_fit.learning_rate=0.0008 \
        ++config_fit.grad_clip=2.5 \
        ++cra_grad_clip=2.5"
        ;;
    "0.3")
        # MODERATE HETEROGENEITY: Strong FedCRA tuning to beat FedProx and FedAvg
        # Reduce anchor overconfidence and moderate CRA strength for better rare-class recovery.
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.008 \
        ++strategy.params.lambda_cra=0.30 \
        ++strategy.params.lambda_cra_initial=0.10 \
        ++strategy.params.lambda_cra_medium=0.18 \
        ++strategy.params.lambda_cra_base=0.30 \
        ++strategy.params.embedding_dim=128 \
        ++strategy.params.use_class_penalty=true \
        ++strategy.params.use_anchor_alignment=true \
        ++config_fit.learning_rate=0.00085 \
        ++config_fit.grad_clip=2.0 \
        ++cra_grad_clip=2.0"
        ;;
    "0.5")
        # MODERATE-LOW HETEROGENEITY: Overall performance priority
        # Initial=0.03 (rounds 1-5), Medium=0.06 (rounds 6-15), Base=0.09 (rounds 16+)
        echo "strategy.params.proximal_mu=0.008 \
        strategy.params.lambda_cra=0.09 \
        ++strategy.params.lambda_cra_initial=0.03 \
        ++strategy.params.lambda_cra_medium=0.06 \
        ++strategy.params.lambda_cra_base=0.09 \
        strategy.params.embedding_dim=128 \
        config_fit.grad_clip=1.5 \
        cra_grad_clip=1.5"
        ;;
    "1"|"1.0")
        # BALANCED DATA: Overall performance priority
        # Initial=0.02 (rounds 1-5), Medium=0.04 (rounds 6-15), Base=0.06 (rounds 16+)
        echo "strategy.params.proximal_mu=0.010 \
        strategy.params.lambda_cra=0.06 \
        ++strategy.params.lambda_cra_initial=0.02 \
        ++strategy.params.lambda_cra_medium=0.04 \
        ++strategy.params.lambda_cra_base=0.06 \
        strategy.params.embedding_dim=128 \
        config_fit.grad_clip=1.4 \
        cra_grad_clip=1.2"
        ;;
    "5")
        # NEARLY IID: Overall performance priority
        # Initial=0.015 (rounds 1-5), Medium=0.03 (rounds 6-15), Base=0.05 (rounds 16+)
        echo "strategy.params.proximal_mu=0.012 \
        strategy.params.lambda_cra=0.05 \
        ++strategy.params.lambda_cra_initial=0.01 \
        ++strategy.params.lambda_cra_medium=0.02 \
        ++strategy.params.lambda_cra_base=0.05 \
        strategy.params.embedding_dim=128 \
        config_fit.grad_clip=1.2 \
        cra_grad_clip=1.0"
        ;;
    *)
        # DEFAULT: Conservative balanced approach
        # Initial=0.05 (rounds 1-5), Medium=0.075 (rounds 6-15), Base=0.10 (rounds 16+)
        echo "strategy.params.proximal_mu=0.01 \
        strategy.params.lambda_cra=0.10 \
        ++strategy.params.lambda_cra_initial=0.05 \
        ++strategy.params.lambda_cra_medium=0.075 \
        ++strategy.params.lambda_cra_base=0.10 \
        strategy.params.embedding_dim=128 \
        config_fit.grad_clip=1.5 \
        cra_grad_clip=1.5"
        ;;
    esac
}

# ========================================================================
# FedProx hyperparameters per alpha value
# ========================================================================
get_fedprox_overrides() {
    local alpha=$1

    # FedProx uses single μ for all classes
    case "${alpha}" in
    "0.1")
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.01"
        ;;
    "0.3")
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.008"
        ;;
    "0.5")
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.005"
        ;;
    "1"|"1.0")
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.003"
        ;;
    "5")
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.001"
        ;;
    *)
        echo "~strategy.params=null ++strategy.params.proximal_mu=0.005"
        ;;
    esac
}

# ========================================================================
# Run single experiment
# ========================================================================
run_experiment() {
    local exp_name=$1
    local imbalance=$2
    local dirichlet=$3
    local num_clients=$4
    local strategy=$5

    local overrides=()

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

    local exp_dir="${RESULTS_DIR}/${exp_name}/${imbalance}/${dirichlet}/num_clients_${num_clients}/${strategy}"

    {
        echo ""
        echo "────────────────────────────────────────────────────────"
        echo "Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}"
        echo "Run            : ${exp_name}"
        echo "Strategy       : ${strategy}"
        echo "Imbalance      : ${imbalance}"
        echo "Dirichlet      : ${dirichlet}"
        echo "Number of Clients : ${num_clients}"
        echo "Output Path    : ${exp_dir}"
    } | tee -a "${EXPERIMENT_LOG}"

    # ✅ Results path
    overrides+=("fed_config.model_results_path=${exp_dir}")
    overrides+=("fed_config.num_rounds=${NUM_ROUNDS}")

    # ✅ Number of clients
    overrides+=("fed_config.num_clients=${num_clients}")
    overrides+=("fed_config.num_clients_per_round_fit=${num_clients}")
    overrides+=("fed_config.num_clients_per_round_eval=${num_clients}")

    # ✅ Data heterogeneity
    overrides+=("imbalance=${imbalance}")
    overrides+=("dirichlet=${dirichlet}")

    # ✅ Strategy selection
    local strategy_group
    if [ "${strategy}" = "FedCRA" ]; then
        strategy_group="fedcra"
    else
        strategy_group="${strategy,,}"
    fi
    overrides+=("strategy=${strategy_group}")

    # Map dirichlet group to alpha string for strategy tuning
    local alpha_value="${dirichlet#alpha_}"
    alpha_value="${alpha_value/_/.}"

    if [ "${strategy}" = "FedCRA" ]; then
        fedcra_overrides=$(get_fedcra_overrides "${alpha_value}")
        for ov in ${fedcra_overrides}; do
            overrides+=("${ov}")
        done
    elif [ "${strategy}" = "FedProx" ]; then
        fedprox_overrides=$(get_fedprox_overrides "${alpha_value}")
        for ov in ${fedprox_overrides}; do
            overrides+=("${ov}")
        done
    fi
    # FedAvg: no additional parameters needed

    # Debug print
    echo "Overrides:" | tee -a "${EXPERIMENT_LOG}"
    for ov in "${overrides[@]}"; do
        echo "  $ov" | tee -a "${EXPERIMENT_LOG}"
    done

    echo "────────────────────────────────────────────────────────" \
        | tee -a "${EXPERIMENT_LOG}"

    cd "${PROJECT_PATH}" && python "${SCRIPT_NAME}" "${overrides[@]}" \
        2>&1 | tee -a "${EXPERIMENT_LOG}"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Completed" | tee -a "${EXPERIMENT_LOG}"
        return 0
    else
        echo "✗ FAILED" | tee -a "${EXPERIMENT_LOG}"
        return 1
    fi
}

# ========================================================================
# Main loop: Test all combinations
# ========================================================================
echo "Starting comprehensive experiments..." | tee -a "${EXPERIMENT_LOG}"

failed_experiments=()

for idx in "${!EXPERIMENT_NAMES[@]}"; do
    imbalance=${IMBALANCE_LABELS[$idx]}
    dirichlet=${DIRICHLET_GROUPS[$idx]}
    exp_name=${EXPERIMENT_NAMES[$idx]}
    for strategy in "${STRATEGIES[@]}"; do
        for num_clients in "${NUM_CLIENTS[@]}"; do
            if ! run_experiment "${exp_name}" "${imbalance}" "${dirichlet}" "${num_clients}" "${strategy}"; then
                failed_experiments+=("${exp_name}_${strategy}_${imbalance}_${dirichlet}_clients_${num_clients}")
            fi
        done
    done
done

# ========================================================================
# Summary
# ========================================================================
{
    echo ""
    echo "==================== SUMMARY ===================="
    echo "Total     : ${TOTAL_EXPERIMENTS}"
    echo "Completed : $((TOTAL_EXPERIMENTS - ${#failed_experiments[@]}))"
    echo "Failed    : ${#failed_experiments[@]}"

    if [ ${#failed_experiments[@]} -ne 0 ]; then
        echo "Failed experiments:"
        for exp in "${failed_experiments[@]}"; do
            echo "  - ${exp}"
        done
    else
        echo "✓ All experiments completed successfully!"
    fi

    echo "================================================"
    echo "Results: ${RESULTS_DIR}"
    echo "Log    : ${EXPERIMENT_LOG}"
    echo "End Time: $(date)"
} | tee -a "${EXPERIMENT_LOG}"

exit ${#failed_experiments[@]}