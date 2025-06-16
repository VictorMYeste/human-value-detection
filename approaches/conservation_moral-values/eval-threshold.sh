#!/usr/bin/env bash
set -e
shopt -s lastpipe        # read … | while read … works as expected

# ---------- 0) parse arguments ---------------------------------------------
if [ $# -lt 1 ]; then
  echo "Usage: $0  MODEL_NAME  [thresholds…]" >&2
  exit 1
fi

MODEL="$1"         # first positional argument
shift              # remove it from $@
THRESHOLDS=("$@")  # remaining args become threshold list

# If no thresholds were supplied, fall back to a default sweep
if [ ${#THRESHOLDS[@]} -eq 0 ]; then
  THRESHOLDS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0)
fi

# ---------- 1) sweep thresholds on the validation split --------------------
bestF1=0
bestT=

for T in "${THRESHOLDS[@]}"; do
  F1=$(python3 eval.py \
          --validation-dataset ../../data/validation-english/ \
          --model-name "$MODEL" \
          --threshold  "$T" \
        | awk '/Macro-average F1 \(fixed threshold\)/ {print $NF}')

  printf "T=%-4s  F1=%s\n" "$T" "$F1"

  # update best if higher (bc does reliable FP compare)
  if (( $(echo "$F1 > $bestF1" | bc -l) )); then   # ← capture bc’s 0/1 result
        bestF1="$F1"
        bestT="$T"
    fi
done

printf "\nBest threshold on validation for %s: %s  (Macro-F1 %s)\n" "$MODEL" "$bestT" "$bestF1"
echo   "-------------------------------------------------------------------"

# ---------- 2) final evaluation on the test split --------------------------
python3 eval.py \
        --test-dataset ../../data/test-english/ \
        --model-name   "$MODEL" \
        --threshold    "$bestT"