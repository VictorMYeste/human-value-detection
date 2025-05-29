#!/bin/bash

echo "=========="
echo "=========="
echo "=========="
echo "===== Baseline ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Baseline

echo "=========="
echo "===== Baseline (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Previous-Sentences-2 ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Baseline

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Previous-Sentences-2 (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Residual Block ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Residual Block (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Baseline

echo "=========="
echo "===== Residual Block (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Residual Block (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Residual Block (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Residual Block (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Residual Block (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --residualblock --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name ResidualBlock --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-VAD ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-VAD (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Baseline

echo "=========="
echo "===== Lex-VAD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-VAD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --lexicon VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-EmoLex ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Baseline

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-EmotionIntensity ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Baseline

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --lexicon EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-WorryWords ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Baseline

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-WorryWords (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --lexicon WorryWords --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-WorryWords --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-LIWC ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Baseline

echo "=========="
echo "===== Lex-LIWC (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --lexicon LIWC --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-LIWC-22 ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Baseline

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --lexicon LIWC-22 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-MFD-20 ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Baseline

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --lexicon MFD-20 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-eMFD ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-eMFD (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Baseline

echo "=========="
echo "===== Lex-eMFD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-eMFD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --lexicon eMFD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-MJD ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MJD (based on Presence = Baseline) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Baseline

echo "=========="
echo "===== Lex-MJD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MJD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --lexicon MJD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1