#!/bin/bash

echo "===== NER + Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --lexicon VAD --model-name NER_Lex-VAD | tee results/NER_Lex-VAD.txt

echo "===== NER + Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --lexicon EmotionIntensity --model-name NER_Lex-EmotionIntensity | tee results/NER_Lex-EmotionIntensity.txt