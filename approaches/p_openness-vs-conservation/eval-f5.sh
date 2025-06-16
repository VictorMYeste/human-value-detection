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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Baseline

echo "=========="
echo "===== Baseline (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Baseline (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

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
echo "===== NER (based on Presence = Baseline) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Baseline

echo "=========="
echo "===== NER (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== NER (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Baseline

echo "=========="
echo "===== Lex-VAD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-VAD (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Baseline

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
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
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

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
echo "===== Lex-LIWC-15 (based on Presence = Baseline) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Baseline

echo "=========="
echo "===== Lex-LIWC-15 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== Lex-LIWC-15 (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-15 (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-15 (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== Lex-LIWC-15 (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-15 --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== TD-NMF ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== TD-NMF (based on Presence = Baseline) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Baseline

echo "=========="
echo "===== TD-NMF (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== TD-NMF (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== TD-NMF (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== TD-NMF (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== TD-NMF (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name TD-NMF --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== NER_Lex-VAD ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Baseline) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Baseline

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-VAD (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-VAD --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1

echo "-----------------------------------------------------"

echo "=========="
echo "=========="
echo "=========="
echo "===== NER_Lex-EmotionIntensity ====="
echo "=========="
echo "=========="
echo "=========="

echo "=========="
echo "=========="
echo "===== Fixed Threshold (0.5) ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Baseline) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Baseline
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Baseline

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat

echo "=========="
echo "=========="
echo "===== Tuned Threshold ====="
echo "=========="
echo "=========="

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Lex-LIWC-22_LingFeat) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Lex-LIWC-22_LingFeat --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-EmoLex) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-EmoLex --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-LIWC-22) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-LIWC-22 --filter-1-th 0.1

echo "=========="
echo "===== NER_Lex-EmotionIntensity (based on Presence = Previous-Sentences-2-Lex-eMFD) ====="
echo "=========="
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name NER_Lex-EmotionIntensity --filter-1-model Previous-Sentences-2-Lex-eMFD --filter-1-th 0.1