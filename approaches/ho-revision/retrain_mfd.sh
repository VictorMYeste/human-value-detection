#!/usr/bin/env bash
# Step 1.9 of the Array revision: retrain the four Lex-MFD models after the
# compute_mfd_scores fix (the word-start marker is now stripped, as for the other
# lexicons) and redo the 19 pipelines that use them. The commands are those of each
# folder's train*.sh and eval*.sh / predict-val.sh, with --model-name Lex-MFD so that
# the model is saved under models/Lex-MFD.
#
# Usage, from the repository root with the hvd environment active:
#   bash approaches/ho-revision/retrain_mfd.sh [PID to wait for before starting]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$ROOT/approaches/ho-revision/results/retrain-mfd.log"

if [ -n "${1:-}" ]; then
  echo "$(date '+%F %T') waiting for PID $1" | tee -a "$LOG"
  while kill -0 "$1" 2>/dev/null; do sleep 300; done
fi

train() {  # folder
  cd "$ROOT/approaches/$1" || return 1
  echo "$(date '+%F %T') train Lex-MFD in $1" | tee -a "$LOG"
  accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ \
    -s 42 --lexicon MFD --model-name Lex-MFD | tee results/Lex-MFD.txt \
    || echo "$(date '+%F %T') FAILED training in $1" | tee -a "$LOG"
}

predict() {  # folder, then the --filter-1-* options
  local d=$1; shift
  cd "$ROOT/approaches/$d" || return 1
  echo "$(date '+%F %T') predict $d $*" | tee -a "$LOG"
  python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD --lexicon MFD "$@" \
    || echo "$(date '+%F %T') FAILED val $d $*" | tee -a "$LOG"
  python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD "$@" \
    || echo "$(date '+%F %T') FAILED test $d $*" | tee -a "$LOG"
}

# Self-Transcendence values (Table 4 selects WorryWords@0.16 / MFD here)
train self-trans_moral-values
G=(--filter-1-dir ../self-trans-vs-self-enh/output/)
predict self-trans_moral-values "${G[@]}" --filter-1-model Baseline
predict self-trans_moral-values "${G[@]}" --filter-1-model Lex-MFD-20
predict self-trans_moral-values "${G[@]}" --filter-1-model Lex-MFD-20 --filter-1-th 0.21
predict self-trans_moral-values "${G[@]}" --filter-1-model Lex-WorryWords --filter-1-th 0.16
predict self-trans_moral-values "${G[@]}" --filter-1-model TD-BERTopic --filter-1-th 0.19

# Conservation values
train conservation_moral-values
G=(--filter-1-dir ../openness-vs-conservation/output/)
predict conservation_moral-values "${G[@]}" --filter-1-model Baseline
predict conservation_moral-values "${G[@]}" --filter-1-model Lex-LIWC-22_LingFeat
predict conservation_moral-values "${G[@]}" --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

# Social Focus values
train social-focus_moral-values
G=(--filter-1-dir ../social-vs-personal-focus/output/)
predict social-focus_moral-values "${G[@]}" --filter-1-model Lex-LIWC --filter-1-th 0.1
predict social-focus_moral-values "${G[@]}" --filter-1-model Lex-WorryWords --filter-1-th 0.1
predict social-focus_moral-values "${G[@]}" --filter-1-model NER --filter-1-th 0.1
predict social-focus_moral-values "${G[@]}" --filter-1-model Previous-Sentences-2 --filter-1-th 0.1
predict social-focus_moral-values "${G[@]}" --filter-1-model TD-NMF

# Presence -> Social vs. Personal Focus cascade (HO model trained on Presence-positive sentences)
train p_social-vs-personal-focus
predict p_social-vs-personal-focus --filter-1-model Baseline
predict p_social-vs-personal-focus --filter-1-model Lex-LIWC-22_LingFeat
predict p_social-vs-personal-focus --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
predict p_social-vs-personal-focus --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
predict p_social-vs-personal-focus --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
predict p_social-vs-personal-focus --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "$(date '+%F %T') done" | tee -a "$LOG"
