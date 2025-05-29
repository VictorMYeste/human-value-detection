#!/bin/bash

echo "===== Baseline ====="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline