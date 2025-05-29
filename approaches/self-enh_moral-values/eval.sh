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
echo "===== Baseline (based on Self-Enhancement = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-MFD-20

echo "=========="
echo "===== Baseline (based on Self-Enhancement = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-BERTopic

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Self-Enhancement = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.23
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-WorryWords --filter-1-th 0.23

echo "=========="
echo "===== Baseline (based on Self-Enhancement = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.24
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-MFD-20 --filter-1-th 0.24

echo "=========="
echo "===== Baseline (based on Self-Enhancement = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.29
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-BERTopic --filter-1-th 0.29

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-Schwartz ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-Schwartz (based on Self-Enhancement = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-MFD-20

echo "=========="
echo "===== Lex-Schwartz (based on Self-Enhancement = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model TD-BERTopic

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-Schwartz (based on Self-Enhancement = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.23
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-WorryWords --filter-1-th 0.23

echo "=========="
echo "===== Lex-Schwartz (based on Self-Enhancement = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.24
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-MFD-20 --filter-1-th 0.24

echo "=========="
echo "===== Lex-Schwartz (based on Self-Enhancement = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.29
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model TD-BERTopic --filter-1-th 0.29