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
echo "===== Previous-Sentences-2 (based on Growth = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences  --filter-1-dir ../growth-vs-self-protection/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Baseline

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Previous-Sentences-2 (based on Growth = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --previous-sentences  --filter-1-dir ../growth-vs-self-protection/output/ --filter-1-model Baseline --filter-1-th 0.18
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2 --filter-1-model Baseline --filter-1-th 0.18