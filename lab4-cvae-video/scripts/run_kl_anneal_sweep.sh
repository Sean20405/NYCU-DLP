#!/usr/bin/env bash

set -u

# Force UTC+8 timezone for all date commands
export TZ='Asia/Taipei'

# KL annealing sweep with LR-sensitive defaults.
# Keep teacher forcing fixed to isolate KL strategy effects.
#
# Usage examples:
#   bash scripts/run_kl_anneal_sweep.sh
#   EPOCHS=70 bash scripts/run_kl_anneal_sweep.sh
#   SAFE_LR_LIST="1e-4 2e-4 3e-4" bash scripts/run_kl_anneal_sweep.sh
#   INCLUDE_HIGH_RISK_LR=1 HIGH_RISK_LR_LIST="8e-4" bash scripts/run_kl_anneal_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINER="src/Trainer.py"
EPOCHS="${EPOCHS:-70}"
RUN_TAG="${RUN_TAG:-kl_sweep_$(date +%Y%m%d_%H%M%S)}"
EXP_GROUP="${EXP_GROUP:-exp_klAnneal}"
CKPT_ROOT="${CKPT_ROOT:-ckpt/${EXP_GROUP}}"
TB_ROOT="${TB_ROOT:-runs/${EXP_GROUP}}"

# Fixed training controls to avoid introducing extra variables.
TFR="${TFR:-0.7}"
TFR_SDE="${TFR_SDE:-20}"
TFR_D_STEP="${TFR_D_STEP:-0.05}"
SCHEDULED_SAMPLING="${SCHEDULED_SAMPLING:-0}"

# LR sensitivity controls:
# - SAFE_LR_LIST is used by default
# - HIGH_RISK_LR_LIST is optional and disabled unless INCLUDE_HIGH_RISK_LR=1
SAFE_LR_LIST="${SAFE_LR_LIST:-1e-4 5e-4}"
HIGH_RISK_LR_LIST="${HIGH_RISK_LR_LIST:-8e-4 1e-3}"
INCLUDE_HIGH_RISK_LR="${INCLUDE_HIGH_RISK_LR:-0}"

# KL annealing grid
KL_TYPES="${KL_TYPES:-mono cyc}"
MONO_RATIO_LIST="${MONO_RATIO_LIST:-0.5 0.75}"
CYC_CYCLE_LIST="${CYC_CYCLE_LIST:-2 5 7 10}"
CYC_RATIO_LIST="${CYC_RATIO_LIST:-0.5 0.75}"
NON_FIXED_BETA_MIN="${NON_FIXED_BETA_MIN:-1e-2}"

# Runtime controls
DRY_RUN="${DRY_RUN:-0}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
SKIP_IF_RECORDED="${SKIP_IF_RECORDED:-1}"
COMMON_ARGS="${COMMON_ARGS:-}"

LOG_DIR="logs/${RUN_TAG}"
SUMMARY_DIR="results"
SUMMARY_CSV="${SUMMARY_DIR}/kl_annealing_best_psnr.csv"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "timestamp,run_tag,exp_name,lr,kl_anneal_type,kl_anneal_cycle,kl_anneal_ratio,kl_beta_min,tfr,tfr_sde,tfr_d_step,scheduled_sampling,best_psnr,status,log_file" > "$SUMMARY_CSV"
fi

if [[ "$INCLUDE_HIGH_RISK_LR" == "1" ]]; then
    LR_LIST="$SAFE_LR_LIST $HIGH_RISK_LR_LIST"
else
    LR_LIST="$SAFE_LR_LIST"
fi

echo "[INFO] Run tag: $RUN_TAG"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Ckpt root: $CKPT_ROOT"
echo "[INFO] TensorBoard root: $TB_ROOT"
echo "[INFO] Fixed TF: tfr=$TFR, tfr_sde=$TFR_SDE, tfr_d_step=$TFR_D_STEP, scheduled_sampling=$SCHEDULED_SAMPLING"
echo "[INFO] LR list: [$LR_LIST] (include high risk: $INCLUDE_HIGH_RISK_LR)"
echo "[INFO] KL types: [$KL_TYPES]"
echo "[INFO] Dry run: $DRY_RUN | Max experiments: $MAX_EXPERIMENTS | Skip if recorded: $SKIP_IF_RECORDED"

DONE_COUNT=0

lr_to_tag() {
    local raw="$1"
    echo "$raw" | sed -e 's/-/m/g' -e 's/\./p/g'
}

append_result() {
    local exp_name="$1"
    local lr="$2"
    local kl_type="$3"
    local kl_cycle="$4"
    local kl_ratio="$5"
    local kl_beta_min="$6"
    local best_psnr="$7"
    local status="$8"
    local log_file="$9"

    echo "$(date '+%F %T'),$RUN_TAG,$exp_name,$lr,$kl_type,$kl_cycle,$kl_ratio,$kl_beta_min,$TFR,$TFR_SDE,$TFR_D_STEP,$SCHEDULED_SAMPLING,$best_psnr,$status,$log_file" >> "$SUMMARY_CSV"
}

run_one() {
    local exp_name="$1"
    local lr="$2"
    local kl_type="$3"
    local kl_cycle="$4"
    local kl_ratio="$5"
    local kl_beta_min="$6"

    if [[ "$MAX_EXPERIMENTS" -gt 0 && "$DONE_COUNT" -ge "$MAX_EXPERIMENTS" ]]; then
        return 10
    fi

    local log_file="${LOG_DIR}/${exp_name}.log"

    if [[ "$SKIP_IF_RECORDED" == "1" ]]; then
        if awk -F, -v e="$exp_name" 'NR>1 && $3==e {found=1} END{exit found?0:1}' "$SUMMARY_CSV"; then
            echo "[INFO] Skip recorded experiment: $exp_name"
            DONE_COUNT=$((DONE_COUNT + 1))
            return 0
        fi
    fi

    local cmd=(
        "$PYTHON_BIN" "$TRAINER"
        --exp_name "$exp_name"
        --num_epoch "$EPOCHS"
        --save_root "$CKPT_ROOT"
        --log_root "$TB_ROOT"
        --lr "$lr"
        --tfr "$TFR"
        --tfr_sde "$TFR_SDE"
        --tfr_d_step "$TFR_D_STEP"
        --kl_anneal_type "$kl_type"
        --kl_anneal_cycle "$kl_cycle"
        --kl_anneal_ratio "$kl_ratio"
        --kl_beta_min "$kl_beta_min"
    )

    if [[ "$SCHEDULED_SAMPLING" == "1" ]]; then
        cmd+=(--scheduled_sampling)
        scheduled_text="true"
    else
        scheduled_text="false"
    fi

    if [[ -n "$COMMON_ARGS" ]]; then
        # shellcheck disable=SC2206
        extra_args=( $COMMON_ARGS )
        cmd+=("${extra_args[@]}")
    fi

    echo "[INFO]============================================================"
    echo "[INFO] Start: $exp_name"
    echo "[INFO] Config: lr=$lr, kl_type=$kl_type, cycle=$kl_cycle, ratio=$kl_ratio, beta_min=$kl_beta_min"
    echo "[INFO] Command: ${cmd[*]}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[INFO] Skip execution due to DRY_RUN=1"
        DONE_COUNT=$((DONE_COUNT + 1))
        return 0
    fi

    set +e
    {
        echo "[META] timestamp=$(date '+%F %T')"
        echo "[META] run_tag=$RUN_TAG"
        echo "[META] exp_name=$exp_name"
        echo "[META] lr=$lr"
        echo "[META] kl_anneal_type=$kl_type"
        echo "[META] kl_anneal_cycle=$kl_cycle"
        echo "[META] kl_anneal_ratio=$kl_ratio"
        echo "[META] kl_beta_min=$kl_beta_min"
        echo "[META] tfr=$TFR"
        echo "[META] tfr_sde=$TFR_SDE"
        echo "[META] tfr_d_step=$TFR_D_STEP"
        echo "[META] scheduled_sampling=$scheduled_text"
        echo "[META] command=${cmd[*]}"
        echo
        "${cmd[@]}"
    } 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local best_psnr
    best_psnr=$(grep -E "Best PSNR:" "$log_file" | tail -n 1 | sed -E 's/.*Best PSNR:[[:space:]]*//')
    if [[ -z "$best_psnr" ]]; then
        best_psnr="NA"
    fi

    local status
    if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
    else
        status="failed($exit_code)"
    fi

    append_result "$exp_name" "$lr" "$kl_type" "$kl_cycle" "$kl_ratio" "$kl_beta_min" "$best_psnr" "$status" "$log_file"

    echo "[INFO] End: $exp_name | status=$status | best_psnr=$best_psnr"
    DONE_COUNT=$((DONE_COUNT + 1))
    return 0
}

set +e

for lr in $LR_LIST; do
    lr_tag=$(lr_to_tag "$lr")

    for kl_type in $KL_TYPES; do
        case "$kl_type" in
            mono)
                for ratio in $MONO_RATIO_LIST; do
                    ratio_tag="${ratio//./p}"
                    exp_name="kl_mono_lr${lr_tag}_r${ratio_tag}"
                    run_one "$exp_name" "$lr" "$kl_type" "1" "$ratio" "$NON_FIXED_BETA_MIN"
                    rc=$?
                    if [[ "$rc" -eq 10 ]]; then
                        break 3
                    fi
                done
                ;;
            cyc)
                for cycle in $CYC_CYCLE_LIST; do
                    for ratio in $CYC_RATIO_LIST; do
                        cycle_tag="${cycle//./p}"
                        ratio_tag="${ratio//./p}"
                        exp_name="kl_cyc_lr${lr_tag}_c${cycle_tag}_r${ratio_tag}"
                        run_one "$exp_name" "$lr" "$kl_type" "$cycle" "$ratio" "$NON_FIXED_BETA_MIN"
                        rc=$?
                        if [[ "$rc" -eq 10 ]]; then
                            break 4
                        fi
                    done
                done
                ;;
            *)
                echo "[WARN] Unknown kl type: $kl_type, skip"
                ;;
        esac
    done
done

set -e

echo "[INFO] All experiments finished."
echo "[INFO] Summary file: $SUMMARY_CSV"
