#!/bin/bash
# =============================================================================
# Array revision of the HO paper (ARRAY-D-26-01445), Reviewer #2 comment 2:
# multi-task training with an auxiliary HO objective.
#
# One DeBERTa-base with a shared encoder and a single multi-label head over 27 labels
# (19 values + 8 HO categories, MODEL_CONFIG["moral_values_ho_aux"]). The training
# command, hyperparameters and flags are those of the Direct Baseline in
# ../moral-values/train_all.sh; only the label set changes. Only the 19 values are
# evaluated, against the cached Direct Baseline of the same seed
# (../moral-values/output/Baseline[-s<seed>]-{val,test}.tsv), by
# ../ho-revision/multitask_eval.py.
#
# Seeds in priority order: 42 (matches the paper), then 7 and 1701 (match the cached
# Direct seeds). Each seed prints SAFE TO STOP when its predictions are written.
#
# Run from anywhere:
#   bash approaches/moral-values-ho-aux/run-ho-aux.sh
# =============================================================================
set -e
set -o pipefail
cd "$(dirname "$0")"

TRAIN=../../data/training-english/
VAL=../../data/validation-english/
TEST=../../data/test-english/
mkdir -p results output

for SEED in 42 7 1701; do
  if [ "$SEED" = "42" ]; then NAME=MT-Baseline; else NAME=MT-Baseline-s${SEED}; fi
  if [ -d "models/${NAME}" ]; then
    echo "----- SKIP train ${NAME}: models/${NAME} already exists -----"
  else
    echo "===== train ${NAME} (seed ${SEED}) $(date) ====="
    accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s "$SEED" \
        --model-name "$NAME" | tee "results/${NAME}.txt"
  fi
  for SPLIT in validation test; do
    OUT="output/${NAME}-$([ $SPLIT = validation ] && echo val || echo test).tsv"
    if [ -f "$OUT" ]; then
      echo "----- SKIP predict ${NAME} on ${SPLIT}: ${OUT} exists -----"
    else
      python3 predict.py --${SPLIT}-dataset "../../data/${SPLIT}-english/" --model-name "$NAME"
    fi
  done
  echo "===== SAFE TO STOP: ${NAME} done $(date) ====="
done
