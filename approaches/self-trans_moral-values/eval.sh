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
echo "===== Baseline (based on Self-Transcendence = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Baseline

echo "=========="
echo "===== Baseline (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-MFD-20

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Self-Transcendence = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.16
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-WorryWords --filter-1-th 0.16

echo "=========="
echo "===== Baseline (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.21
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-MFD-20 --filter-1-th 0.21

echo "=========="
echo "===== Baseline (based on Self-Transcendence = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.19
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model TD-BERTopic --filter-1-th 0.19

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
echo "===== Lex-Schwartz (based on Self-Transcendence = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Baseline

echo "=========="
echo "===== Lex-Schwartz (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-MFD-20

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-Schwartz (based on Self-Transcendence = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.16
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-WorryWords --filter-1-th 0.16

echo "=========="
echo "===== Lex-Schwartz (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.21
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-MFD-20 --filter-1-th 0.21

echo "=========="
echo "===== Lex-Schwartz (based on Self-Transcendence = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --lexicon Schwartz --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.19
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model TD-BERTopic --filter-1-th 0.19

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
echo "===== Lex-EmoLex (based on Self-Transcendence = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Baseline

echo "=========="
echo "===== Lex-EmoLex (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-MFD-20

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-EmoLex (based on Self-Transcendence = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.16
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-WorryWords --filter-1-th 0.16

echo "=========="
echo "===== Lex-EmoLex (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.21
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-MFD-20 --filter-1-th 0.21

echo "=========="
echo "===== Lex-EmoLex (based on Self-Transcendence = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --lexicon EmoLex --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.19
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model TD-BERTopic --filter-1-th 0.19

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== Lex-MFD ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MFD (based on Self-Transcendence = Baseline) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Baseline

echo "=========="
echo "===== Lex-MFD (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Lex-MFD-20

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-MFD (based on Self-Transcendence = Lex-WorryWords) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-WorryWords --filter-1-th 0.16
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Lex-WorryWords --filter-1-th 0.16

echo "=========="
echo "===== Lex-MFD (based on Self-Transcendence = Lex-MFD-20) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model Lex-MFD-20 --filter-1-th 0.21
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Lex-MFD-20 --filter-1-th 0.21

echo "=========="
echo "===== Lex-MFD (based on Self-Transcendence = TD-BERTopic) ====="
echo "=========="
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --lexicon MFD --filter-1-dir ../self-trans-vs-self-enh/output/ --filter-1-model TD-BERTopic --filter-1-th 0.19
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model TD-BERTopic --filter-1-th 0.19