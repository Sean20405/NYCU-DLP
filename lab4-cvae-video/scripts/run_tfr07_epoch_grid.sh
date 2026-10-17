#!/usr/bin/env bash

set -u

# Fixed setting for this grid search:
# - per-epoch teacher forcing (scheduled_sampling disabled)
# - tfr=0.7
# Sweep dimensions:
# - tfr_sde in SDE_LIST
# - tfr_d_step in DSTEP_LIST

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINER="src/Trainer.py"
EPOCHS="${EPOCHS:-70}"
COMMON_ARGS="${COMMON_ARGS:---kl_anneal_type mono}"
RUN_TAG="${RUN_TAG:-tf_sde_dstep_tfr0p7_$(date +%Y%m%d_%H%M%S)}"

# Grid defaults (can override by env)
SDE_LIST="${SDE_LIST:-10 15 20 25 30}"
DSTEP_LIST="${DSTEP_LIST:-0.05 0.10}"

# Runtime controls
DRY_RUN="${DRY_RUN:-0}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
SKIP_IF_RECORDED="${SKIP_IF_RECORDED:-1}"

LOG_DIR="logs/${RUN_TAG}"
SUMMARY_DIR="results"
SUMMARY_CSV="${SUMMARY_DIR}/teacher_forcing_best_psnr.csv"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

if [[ ! -f "$SUMMARY_CSV" ]]; then
    echo "timestamp,run_tag,exp_name,tfr,tfr_sde,tfr_d_step,scheduled_sampling,best_psnr,status,log_file" > "$SUMMARY_CSV"
fi

echo "[INFO] Run tag: $RUN_TAG"
echo "[INFO] Logs: $LOG_DIR"
echo "[INFO] Summary CSV: $SUMMARY_CSV"
echo "[INFO] Fixed config: scheduled_sampling=false, tfr=0.7"
echo "[INFO] Grid: SDE_LIST=[$SDE_LIST], DSTEP_LIST=[$DSTEP_LIST]"
echo "[INFO] Dry run: $DRY_RUN | Max experiments: $MAX_EXPERIMENTS | Skip if recorded: $SKIP_IF_RECORDED"

DONE_COUNT=0

for SDE in $SDE_LIST; do
    for DSTEP in $DSTEP_LIST; do
        if [[ "$MAX_EXPERIMENTS" -gt 0 && "$DONE_COUNT" -ge "$MAX_EXPERIMENTS" ]]; then
            echo "[INFO] Reached MAX_EXPERIMENTS=$MAX_EXPERIMENTS, stop here."
            break 2
        fi

        DSTEP_TAG="${DSTEP//./p}"
        EXP_NAME="tf_epoch_tfr0p7_sde${SDE}_d${DSTEP_TAG}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

        if [[ "$SKIP_IF_RECORDED" == "1" ]]; then
            if awk -F, -v e="$EXP_NAME" 'NR>1 && $3==e {found=1} END{exit found?0:1}' "$SUMMARY_CSV"; then
                echo "[INFO] Skip recorded experiment: $EXP_NAME"
                DONE_COUNT=$((DONE_COUNT + 1))
                continue
            fi
        fi

        CMD=(
            "$PYTHON_BIN" "$TRAINER"
            --exp_name "$EXP_NAME"
            --num_epoch "$EPOCHS"
            --tfr "0.7"
            --tfr_sde "$SDE"
            --tfr_d_step "$DSTEP"
        )

        if [[ -n "$COMMON_ARGS" ]]; then
            # shellcheck disable=SC2206
            EXTRA_ARGS=( $COMMON_ARGS )
            CMD+=("${EXTRA_ARGS[@]}")
        fi

        echo "[INFO]============================================================"
        echo "[INFO] Start: $EXP_NAME"
        echo "[INFO] Config: tfr=0.7, tfr_sde=$SDE, tfr_d_step=$DSTEP, scheduled_sampling=false"
        echo "[INFO] Command: ${CMD[*]}"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[INFO] Skip execution due to DRY_RUN=1"
            DONE_COUNT=$((DONE_COUNT + 1))
            continue
        fi

        set +e
        {
            echo "[META] timestamp=$(date '+%F %T')"
            echo "[META] run_tag=$RUN_TAG"
            echo "[META] exp_name=$EXP_NAME"
            echo "[META] tfr=0.7"
            echo "[META] tfr_sde=$SDE"
            echo "[META] tfr_d_step=$DSTEP"
            echo "[META] scheduled_sampling=false"
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

        echo "$(date '+%F %T'),$RUN_TAG,$EXP_NAME,0.7,$SDE,$DSTEP,false,$BEST_PSNR,$STATUS,$LOG_FILE" >> "$SUMMARY_CSV"

        echo "[INFO] End: $EXP_NAME | status=$STATUS | best_psnr=$BEST_PSNR"
        DONE_COUNT=$((DONE_COUNT + 1))
    done
done

echo "[INFO] All experiments finished."
echo "[INFO] Summary file: $SUMMARY_CSV"
