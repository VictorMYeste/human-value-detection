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
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== NER ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-Schwartz (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-Schwartz (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-Schwartz (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-Schwartz (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-Schwartz --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-VAD (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-EmoLex (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmoLex (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmoLex --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-MFD (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-LIWC-22 (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22 (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22 --filter-1-model Lex-LIWC --filter-1-th 0.1

echo "-----------------------------------------------------"

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
echo "===== Lex-LIWC-22_LingFeat (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-22_LingFeat (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-MFD-20 (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-MFD-20 (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MFD-20 --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Lex-eMFD (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-eMFD (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-eMFD --filter-1-model Lex-LIWC --filter-1-th 0.1

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
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== Lex-MJD (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-MJD --filter-1-model Lex-LIWC --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== TD-LDA ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== TD-LDA (based on Social Focus = TD-NMF) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-LDA --filter-1-model TD-NMF

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Baseline (based on Social Focus = Previous-Sentences-2) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-LDA --filter-1-model Previous-Sentences-2 --filter-1-th 0.1

echo "=========="
echo "===== TD-LDA (based on Social Focus = NER) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-LDA --filter-1-model NER --filter-1-th 0.1

echo "=========="
echo "===== TD-LDA (based on Social Focus = Lex - WorryWords) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-LDA --filter-1-model Lex-WorryWords --filter-1-th 0.1

echo "=========="
echo "===== TD-LDA (based on Social Focus = Lex - LIWC 15) ====="
echo "=========="
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-LDA --filter-1-model Lex-LIWC --filter-1-th 0.1