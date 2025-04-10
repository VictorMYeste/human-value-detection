#!/bin/bash

echo "===== Baseline ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --model-name Baseline | tee results/Baseline.txt

echo "===== NER ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --model-name NER | tee results/NER.txt

echo "===== Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon VAD --model-name Lex-VAD | tee results/Lex-VAD.txt

echo "===== Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon EmotionIntensity --model-name Lex-EmotionIntensity | tee results/Lex-EmotionIntensity.txt

echo "===== Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC --model-name LIWC-15 | tee results/Lex-LIWC-15.txt

echo "===== Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection nmf --model-name TD-NMF | tee results/TD-NMF.txt