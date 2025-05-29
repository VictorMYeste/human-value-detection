#!/bin/bash

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-LIWC-22_LingFeat ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Conservation = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features --filter-1-dir ../openness-vs-conservation/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Baseline

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Conservation = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features --filter-1-dir ../openness-vs-conservation/output/ --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Conservation = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features --filter-1-dir ../openness-vs-conservation/output/ --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1