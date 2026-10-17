#!/usr/bin/env bash

set -u

# Usage examples:
#   bash scripts/run_teacher_forcing_sweep.sh
#   CUDA_VISIBLE_DEVICES=0 EPOCHS=70 bash scripts/run_teacher_forcing_sweep.sh
#   PYTHON_BIN=python3.13 COMMON_ARGS="--kl_anneal_type mono --batch_size 4" bash scripts/run_teacher_forcing_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINER="src/Trainer.py"
EPOCHS="${EPOCHS:-70}"
COMMON_ARGS="${COMMON_ARGS:---kl_anneal_type mono}"
RUN_TAG="${RUN_TAG:-tf_sweep_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
RECORD_DRY_RUN="${RECORD_DRY_RUN:-0}"

LOG_DIR="logs/${RUN_TAG}"
SUMMARY_DIR="results"
SUMMARY_CSV="${SUMMARY_DIR}/teacher_forcing_best_psnr.csv"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "timestamp,run_tag,exp_name,tfr,tfr_sde,tfr_d_step,scheduled_sampling,best_psnr,status,log_file" > "$SUMMARY_CSV"
fi

# Format per row:
# exp_name|tfr|tfr_sde|tfr_d_step|scheduled_sampling(0/1)
EXPERIMENTS=(
    "tf_epoch_tfr1p0_sde20_d0p05|1.0|20|0.05|0"
    "tf_epoch_tfr1p0_sde25_d0p10|1.0|25|0.10|0"
    "tf_epoch_tfr1p0_sde30_d0p05|1.0|30|0.05|0"
    "tf_epoch_tfr0p7_sde20_d0p05|0.7|20|0.05|0"
    "tf_epoch_tfr0p7_sde25_d0p10|0.7|25|0.10|0"
    "tf_frame_tfr1p0_sde20_d0p05|1.0|20|0.05|1"
    "tf_frame_tfr1p0_sde25_d0p10|1.0|25|0.10|1"
    "tf_frame_tfr0p7_sde30_d0p05|0.7|30|0.05|1"
)

echo "[INFO] Run tag: $RUN_TAG"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Dry run: $DRY_RUN | Record dry run: $RECORD_DRY_RUN | Max experiments: $MAX_EXPERIMENTS"

DONE_COUNT=0

for row in "${EXPERIMENTS[@]}"; do
    if [[ "$MAX_EXPERIMENTS" -gt 0 && "$DONE_COUNT" -ge "$MAX_EXPERIMENTS" ]]; then
        echo "[INFO] Reached MAX_EXPERIMENTS=$MAX_EXPERIMENTS, stop here."
        break
    fi

    IFS='|' read -r EXP_NAME TFR TFR_SDE TFR_D_STEP SCHEDULED <<< "$row"

    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    CMD=(
        "$PYTHON_BIN" "$TRAINER"
        --exp_name "$EXP_NAME"
        --num_epoch "$EPOCHS"
        --tfr "$TFR"
        --tfr_sde "$TFR_SDE"
        --tfr_d_step "$TFR_D_STEP"
    )

    if [[ "$SCHEDULED" == "1" ]]; then
        CMD+=(--scheduled_sampling)
        SCHEDULED_TEXT="true"
    else
        SCHEDULED_TEXT="false"
    fi

    if [[ -n "$COMMON_ARGS" ]]; then
        # shellcheck disable=SC2206
        EXTRA_ARGS=( $COMMON_ARGS )
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo "[INFO]============================================================"
    echo "[INFO] Start: $EXP_NAME"
    echo "[INFO] Config: tfr=$TFR, tfr_sde=$TFR_SDE, tfr_d_step=$TFR_D_STEP, scheduled_sampling=$SCHEDULED_TEXT"
    echo "[INFO] Command: ${CMD[*]}"

    if [[ "$DRY_RUN" == "1" ]]; then
        STATUS="dry-run"
        BEST_PSNR="NA"
        if [[ "$RECORD_DRY_RUN" == "1" ]]; then
            echo "$(date '+%F %T'),$RUN_TAG,$EXP_NAME,$TFR,$TFR_SDE,$TFR_D_STEP,$SCHEDULED_TEXT,$BEST_PSNR,$STATUS,$LOG_FILE" >> "$SUMMARY_CSV"
        fi
        echo "[INFO] Skip execution due to DRY_RUN=1"
        DONE_COUNT=$((DONE_COUNT + 1))
        continue
    fi

    set +e
    {
        echo "[META] timestamp=$(date '+%F %T')"
        echo "[META] run_tag=$RUN_TAG"
        echo "[META] exp_name=$EXP_NAME"
        echo "[META] tfr=$TFR"
        echo "[META] tfr_sde=$TFR_SDE"
        echo "[META] tfr_d_step=$TFR_D_STEP"
        echo "[META] scheduled_sampling=$SCHEDULED_TEXT"
        echo "[META] command=${CMD[*]}"
        echo
        "${CMD[@]}"
    } 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    BEST_PSNR=$(grep -E "Best PSNR:" "$LOG_FILE" | tail -n 1 | sed -E 's/.*Best PSNR:[[:space:]]*//')
    if [[ -z "$BEST_PSNR" ]]; then
        BEST_PSNR="NA"
    fi

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        STATUS="ok"
    else
        STATUS="failed($EXIT_CODE)"
    fi

    echo "$(date '+%F %T'),$RUN_TAG,$EXP_NAME,$TFR,$TFR_SDE,$TFR_D_STEP,$SCHEDULED_TEXT,$BEST_PSNR,$STATUS,$LOG_FILE" >> "$SUMMARY_CSV"

    echo "[INFO] End: $EXP_NAME | status=$STATUS | best_psnr=$BEST_PSNR"
    DONE_COUNT=$((DONE_COUNT + 1))
done

echo "[INFO] All experiments finished."
echo "[INFO] Summary file: $SUMMARY_CSV"
