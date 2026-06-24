#!/bin/bash
# =============================================================================
# IP&M major revision — multi-seed reruns for mean ± std (run on the GPU box)
#
# Three phases, in PRIORITY order so you can Ctrl-C when the deadline is close.
# Each phase prints a "SAFE TO STOP" banner when it finishes.
#
#   PHASE 1  (required)     A1 — Table 4 / tab:lightweight-signals direct configs
#                           (Baseline, Prev-2, LIWC-22, BERTopic) + A3 direct MJD
#   PHASE 2  (recommended)  Table 3 / tab:direct-vs-hier gated pipeline
#                           (direct+MJD, gated+MJD, gated+LIWC; direct+LIWC is in P1)
#   PHASE 3  (optional)     Table 2 / tab:presence-gate presence models
#
# Run from inside  human-value-detection/approaches/moral-values/
#   cd approaches/moral-values && bash run-revision-A1-A3.sh
#
# Notes
# - Mirrors train_all.sh / eval.sh / eval-threshold.sh and presence|p_moral-values
#   exactly; only seed + model-name change.
# - predict.py / eval.py ALWAYS key off models/<model-name> and
#   output/[1_<gate>_<th>_]<model-name>-<split>.tsv (core/prediction.py:237,
#   core/evaluation.py:209), so each seed uses a unique -s<N> model-name and never
#   overwrites the published seed-42 artefacts in output/.
# - Seed 42 == the existing paper outputs; this adds seeds 7,1701 -> mean ± std over
#   {42,7,1701}. Running all seeds on the SAME machine keeps std = seed variance only.
#
# IMPORTANT — keep the 8 GB config even on a bigger GPU.
#   Do NOT raise batch_size / lower grad-accum / change seq-len. The paper's
#   contribution is reproducibility within an 8 GB budget; a faster GPU is for
#   wall-clock speed only. The 7.5 GB peak (Section 5.4) stays as measured on the
#   RTX 3070.
# =============================================================================
set -e
set -o pipefail   # so a failed `accelerate ... | tee` aborts instead of silently continuing to predict
cd "$(dirname "$0")"

TRAIN=../../data/training-english/
VAL=../../data/validation-english/
TEST=../../data/test-english/

SEEDS="7 1701"                       # seed 42 already on disk
GATE_BASE="Previous-Sentences-2-Lex-LIWC-22"   # presence gate = Prev-2 + LIWC-22
GATE_TH=0.1                                     # t_gate from the paper (Table 3)
SWEEP="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0"
mkdir -p results

# ----------------------------- helpers ---------------------------------------
# Train a moral-values config + predict val & test (no eval).  $1=name $2=seed; rest=flags
mv_train () {
  local NAME=$1 SEED=$2; shift 2; local FLAGS="$*"
  if [ -d "models/${NAME}" ]; then
    echo "----- SKIP train ${NAME}: models/${NAME} already exists (reusing checkpoint) -----"
  else
    echo "----- train ${NAME} (seed ${SEED}) flags:[${FLAGS}] -----"
    accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s "$SEED" \
        $FLAGS --model-name "$NAME" | tee "results/${NAME}.txt"
  fi
  python3 predict.py --validation-dataset "$VAL" $FLAGS --model-name "$NAME"
  python3 predict.py --test-dataset      "$TEST" $FLAGS --model-name "$NAME"
}

# Direct value eval: sweep global threshold on val, apply best to test.  $1=name
mv_eval_direct () {
  ./eval-threshold.sh "$1" | tee "results/${1}-direct-tuned.txt"
}

# Gated pipeline via p_moral-values (the published hierarchical design): the value
# head is trained on PRESENCE-POSITIVE rows only (p_moral-values/main.py hardcodes
# filter_labels=['Presence']; confirmed in its training logs). Trains the filtered
# value head, gated-predicts (reads the gate preds from ../presence/output/), and
# runs a tuned gated eval (sweep value threshold on val, apply to test).
#   $1=value-name  $2=seed  $3=gate-name ; rest=value flags (e.g. --lexicon MJD)
pmv_gated () {
  local VNAME=$1 SEED=$2 GATE=$3; shift 3; local FLAGS="$*"
  ( cd ../p_moral-values
    if [ -d "models/${VNAME}" ]; then
      echo "----- SKIP p_moral-values train ${VNAME}: models/${VNAME} already exists (reusing checkpoint) -----"
    else
      echo "----- p_moral-values (FILTERED) train ${VNAME} (seed ${SEED}) flags:[${FLAGS}] -----"
      accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s "$SEED" \
          $FLAGS --model-name "$VNAME" | tee "results/${VNAME}.txt"
    fi
    python3 predict.py --validation-dataset "$VAL" $FLAGS --model-name "$VNAME" \
        --filter-1-model "$GATE" --filter-1-th "$GATE_TH"
    python3 predict.py --test-dataset      "$TEST" $FLAGS --model-name "$VNAME" \
        --filter-1-model "$GATE" --filter-1-th "$GATE_TH"
    local bestF1=0 bestT=
    for T in $SWEEP; do
      F1=$(python3 eval.py --validation-dataset "$VAL" --model-name "$VNAME" \
              --filter-1-model "$GATE" --filter-1-th "$GATE_TH" --threshold "$T" \
            | awk '/Macro-average F1 \(fixed threshold\)/ {print $NF}')
      printf "  T=%-4s F1=%s\n" "$T" "$F1"
      if (( $(echo "${F1:-0} > $bestF1" | bc -l) )); then bestF1=$F1; bestT=$T; fi
    done
    echo "  best gated value-threshold for ${VNAME} (gate ${GATE} @ ${GATE_TH}): ${bestT} (F1 ${bestF1})"
    python3 eval.py --test-dataset "$TEST" --model-name "$VNAME" \
        --filter-1-model "$GATE" --filter-1-th "$GATE_TH" --threshold "$bestT" \
      | tee "results/gated_${GATE}_${GATE_TH}_${VNAME}-tuned.txt" )
}

# Presence: train + predict val & test, in ../presence.  $1=name $2=seed; rest=flags
pres_train_predict () {
  local NAME=$1 SEED=$2; shift 2; local FLAGS="$*"
  ( cd ../presence
    if [ -d "models/${NAME}" ]; then
      echo "----- SKIP presence train ${NAME}: models/${NAME} already exists (reusing checkpoint) -----"
    else
      echo "----- presence train ${NAME} (seed ${SEED}) flags:[${FLAGS}] -----"
      accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s "$SEED" \
          $FLAGS --model-name "$NAME" | tee "results/${NAME}.txt"
    fi
    python3 predict.py --validation-dataset "$VAL" $FLAGS --model-name "$NAME"
    python3 predict.py --test-dataset      "$TEST" $FLAGS --model-name "$NAME" )
}

# Presence eval: val (writes per-label/global threshold) -> test @ tuned and @ 0.5.  $1=name
pres_eval () {
  local NAME=$1
  ( cd ../presence
    python3 eval.py --validation-dataset "$VAL" --model-name "$NAME"
    echo "--- presence ${NAME} @ tuned t* ---"
    python3 eval.py --test-dataset "$TEST" --model-name "$NAME"
    echo "--- presence ${NAME} @ 0.5 ---"
    python3 eval.py --test-dataset "$TEST" --model-name "$NAME" --threshold 0.5
  ) | tee "results/presence_${NAME}.txt"
}

# =============================================================================
# PHASE 1 — A1 (Table 4 direct value configs) + A3 (direct MJD, seed 42)
# =============================================================================
echo "######## PHASE 1 :: A1 Table-4 direct configs + A3 direct MJD ########"
for SEED in $SEEDS; do
  mv_train "Baseline-s${SEED}"             "$SEED";                          mv_eval_direct "Baseline-s${SEED}"
  mv_train "Previous-Sentences-2-s${SEED}" "$SEED" --previous-sentences;     mv_eval_direct "Previous-Sentences-2-s${SEED}"
  mv_train "Lex-LIWC-22-s${SEED}"          "$SEED" --lexicon LIWC-22;        mv_eval_direct "Lex-LIWC-22-s${SEED}"
  mv_train "TD-BERTopic-s${SEED}"          "$SEED" --topic-detection bertopic; mv_eval_direct "TD-BERTopic-s${SEED}"
done

# A3 — direct (un-gated) Lex-MJD at seed 42. PREFER reusing the original checkpoint
# (the gated+MJD preds on disk came from it -> contrast stays feature+seed matched).
echo "===== A3 :: direct Lex-MJD (seed 42) ====="
if [ ! -d models/Lex-MJD ]; then
  echo "WARNING: no models/Lex-MJD checkpoint -> retraining (seed 42)."
  echo "         A fresh checkpoint != the one behind the gated+MJD preds, so ALSO"
  echo "         regenerate gated+MJD afterwards (cd ../p_moral-values; predict.py"
  echo "         --lexicon MJD --model-name Lex-MJD --filter-1-model ${GATE_BASE}"
  echo "         --filter-1-th ${GATE_TH}  for both --validation-dataset and --test-dataset)."
  accelerate launch --multi_gpu main.py -t "$TRAIN" -v "$VAL" -s 42 \
      --lexicon MJD --model-name Lex-MJD | tee results/Lex-MJD.txt
else
  echo "Reusing existing models/Lex-MJD checkpoint -> predict only."
fi
python3 predict.py --validation-dataset "$VAL" --lexicon MJD --model-name Lex-MJD
python3 predict.py --test-dataset      "$TEST" --lexicon MJD --model-name Lex-MJD
mv_eval_direct Lex-MJD
echo "######## SAFE TO STOP — Table 4 multi-seed + A3 complete ########"

# =============================================================================
# PHASE 2 — Table 3 gated pipeline across seeds (feature-matched direct-vs-gated)
#   The published hierarchical design trains the value head on PRESENCE-POSITIVE
#   rows only (p_moral-values, filter_labels=['Presence']); the DIRECT rows use the
#   full-data heads (moral-values). So per seed:
#     gate  : presence Prev-2+LIWC-22            (in ../presence)
#     direct: Lex-MJD (full data)                -> direct+MJD  (direct+LIWC = PHASE 1)
#     gated : Lex-MJD, Lex-LIWC-22 (filtered)    -> gated+MJD, gated+LIWC  (in ../p_moral-values)
#   NOTE: direct (full) vs gated (filtered) is confounded by training distribution,
#   not just architecture — see the Sec 5.2 correction / R3-C6 note in the tasks file.
#   (The gate trained here doubles as Table 2's "Prev-2 + LIWC-22" row -> reused in P3.)
# =============================================================================
echo "######## PHASE 2 :: Table 3 gated pipeline ########"
for SEED in $SEEDS; do
  GATE="${GATE_BASE}-s${SEED}"
  pres_train_predict "$GATE" "$SEED" --previous-sentences --lexicon LIWC-22     # presence gate

  mv_train       "Lex-MJD-s${SEED}" "$SEED" --lexicon MJD                       # DIRECT (full) MJD head
  mv_eval_direct "Lex-MJD-s${SEED}"                                             # direct+MJD row

  pmv_gated "Lex-MJD-s${SEED}"     "$SEED" "$GATE" --lexicon MJD                 # gated+MJD  (filtered)
  pmv_gated "Lex-LIWC-22-s${SEED}" "$SEED" "$GATE" --lexicon LIWC-22            # gated+LIWC (filtered)
done
echo "######## SAFE TO STOP — Table 3 gated multi-seed complete ########"

# =============================================================================
# PHASE 3 — Table 2 presence models across seeds
#   Prev-2+LIWC-22 already trained in PHASE 2 -> only eval it here.
# =============================================================================
echo "######## PHASE 3 :: Table 2 presence ########"
for SEED in $SEEDS; do
  pres_train_predict "Baseline-s${SEED}" "$SEED";                                  pres_eval "Baseline-s${SEED}"
  pres_train_predict "Lex-LIWC-22_LingFeat-s${SEED}" "$SEED" --lexicon LIWC-22 --linguistic-features
  pres_eval          "Lex-LIWC-22_LingFeat-s${SEED}"
  pres_eval          "${GATE_BASE}-s${SEED}"                                       # gate from PHASE 2
  pres_train_predict "Previous-Sentences-2-Lex-EmoLex-s${SEED}" "$SEED" --previous-sentences --lexicon EmoLex
  pres_eval          "Previous-Sentences-2-Lex-EmoLex-s${SEED}"
  pres_train_predict "Previous-Sentences-2-Lex-eMFD-s${SEED}"  "$SEED" --previous-sentences --lexicon eMFD
  pres_eval          "Previous-Sentences-2-Lex-eMFD-s${SEED}"
done
echo "######## ALL DONE — Tables 4, 3, 2 multi-seed complete ########"

# -----------------------------------------------------------------------------
# Aggregation (no GPU): tuned test macro-F1 per seed is in
#   results/*-direct-tuned.txt                  (direct value configs: Tables 4 & 3, full-data heads)
#   ../p_moral-values/results/gated_*-tuned.txt (gated value configs: Table 3, filtered heads)
#   results/presence_*.txt + ../presence/results/* (presence: Table 2)
# Combine with the existing seed-42 outputs to report mean ± std.
# Feature-matched A3 contrast (also no GPU), per seed and at seed 42:
#   core/compare_models.py --pred1 <direct,moral-values/output> --pred2 <gated,p_moral-values/output>
#   (each at its tuned threshold)
# -----------------------------------------------------------------------------
