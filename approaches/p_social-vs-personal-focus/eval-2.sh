#!/bin/bash

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