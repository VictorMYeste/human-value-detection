#!/bin/bash
# =============================================================================
# IP&M resubmission — cross-encoder replication (RoBERTa-base)
#
# WHY THIS EXISTS
#   Reviewer #4 (W2, W6, SC2) and the Associate Editor's decision summary both
#   require evidence that the findings generalise beyond a single encoder family.
#   The paper's headline result is the threshold-calibration decomposition, so
#   the question that matters is narrow and specific:
#
#       does decision-threshold calibration dominate for a second encoder too?
#
#   This script answers exactly that and nothing more. Text-only baseline, no
#   auxiliary features, no hierarchy, no ensembling — the same three seeds and
#   the same training protocol as the published DeBERTa runs, with the encoder
#   swapped. Any difference in the result is therefore attributable to the
#   encoder and nothing else.
#
# WHAT IT PRODUCES  (per seed, mirroring the DeBERTa Baseline rows)
#   - test macro-F1 at the default threshold t = 0.5
#   - test macro-F1 at t* tuned on validation
#   -> mean +/- std over seeds 42, 7, 1701, directly comparable to the DeBERTa
#      0.282 -> 0.315 decomposition already in the paper.
#
# HOW THE ENCODER IS SWAPPED
#   core/config.py reads HVD_PRETRAINED_MODEL, defaulting to microsoft/deberta-base.
#   Only the "moral_values" group is affected; presence and hierarchical configs
#   are untouched. With the variable unset, every existing run behaves byte-for-byte
#   as before.
#
# RUN IT
#   cd approaches/moral-values && bash run-roberta-cross-encoder.sh
#
# IMPORTANT — keep the 8 GB config.
#   Do NOT raise batch_size, lower grad-accum, or change sequence length, even on
#   a larger GPU. The paper's claim is reproducibility within an 8 GB budget, and
#   the DeBERTa contrast is only valid if the protocol is identical. A faster GPU
#   buys wall-clock time, nothing else.
#
# NAMING
#   Models are named RoBERTa-Baseline-s<seed>, so nothing can overwrite the
#   published DeBERTa artefacts in models/ or output/.
# =============================================================================
set -e
set -o pipefail
cd "$(dirname "$0")"

export HVD_PRETRAINED_MODEL="FacebookAI/roberta-base"

TRAIN=../../data/training-english/
VAL=../../data/validation-english/
TEST=../../data/test-english/

SEEDS="42 7 1701"
SWEEP="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0"

mkdir -p results

echo "############################################################"
echo "# Cross-encoder replication"
echo "#   encoder : ${HVD_PRETRAINED_MODEL}"
echo "#   seeds   : ${SEEDS}"
echo "#   config  : text-only baseline, no features, no gate"
echo "############################################################"
echo

for SEED in $SEEDS; do
  NAME="RoBERTa-Baseline-s${SEED}"

  echo "===== [${NAME}] train ====="
  if [ -d "models/${NAME}" ]; then
    echo "----- SKIP: models/${NAME} already exists (reusing checkpoint) -----"
  else
    accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s "$SEED" \
        --model-name "$NAME" | tee "results/${NAME}.txt"
  fi

  echo "===== [${NAME}] predict val + test ====="
  python3 predict.py --validation-dataset "$VAL" --model-name "$NAME"
  python3 predict.py --test-dataset       "$TEST" --model-name "$NAME"

  # --- default threshold: the "uncalibrated" half of the decomposition --------
  echo "===== [${NAME}] test @ t = 0.5 (default) ====="
  python3 eval.py --test-dataset "$TEST" --model-name "$NAME" --threshold 0.5 \
    | tee "results/${NAME}-test-t050.txt"

  # --- tuned threshold: sweep on validation, apply the winner to test ---------
  echo "===== [${NAME}] threshold sweep on validation -> test @ t* ====="
  ./eval-threshold.sh "$NAME" $SWEEP | tee "results/${NAME}-direct-tuned.txt"

  echo "----- ${NAME} complete -----"
  echo
done

echo "############################################################"
echo "# ALL DONE"
echo "#"
echo "# Per seed, read off:"
echo "#   results/RoBERTa-Baseline-s<seed>-test-t050.txt      -> macro-F1 @ 0.5"
echo "#   results/RoBERTa-Baseline-s<seed>-direct-tuned.txt   -> best t* and macro-F1 @ t*"
echo "#"
echo "# Then report mean +/- std over seeds 42, 7, 1701 alongside the DeBERTa"
echo "# baseline rows (0.282 @ 0.5 -> 0.315 @ t*)."
echo "############################################################"
