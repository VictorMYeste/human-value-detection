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
echo "===== Baseline (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model NER --filter-1-th 0.19

echo "=========="
echo "===== Baseline (based on Social Focus = Lex - Schwartz) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-Schwartz --filter-1-th 0.21

echo "=========="
echo "===== Baseline (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-WorryWords --filter-1-th 0.21

echo "=========="
echo "===== Baseline (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC --filter-1-th 0.17

echo "=========="
echo "===== Baseline (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-NMF --filter-1-th 0.2

echo "=========="
echo "===== Baseline (based on Social Focus = TD-BERTopic) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-BERTopic --filter-1-th 0.22

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
echo "===== Lex-MJD (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MJD (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model NER --filter-1-th 0.19

echo "=========="
echo "===== Lex-MJD (based on Social Focus = Lex - Schwartz) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-Schwartz --filter-1-th 0.21

echo "=========="
echo "===== Lex-MJD (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-WorryWords --filter-1-th 0.21

echo "=========="
echo "===== Lex-MJD (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC --filter-1-th 0.17

echo "=========="
echo "===== Lex-MJD (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model TD-NMF --filter-1-th 0.2

echo "=========="
echo "===== Lex-MJD (based on Social Focus = TD-BERTopic) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model TD-BERTopic --filter-1-th 0.22