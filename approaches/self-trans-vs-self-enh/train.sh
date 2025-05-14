#!/bin/bash

echo "===== Lex - WorryWords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon WorryWords --model-name Lex-WorryWords | tee results/Lex-WorryWords.txt

echo "===== Lex - MFD-20 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD-20 --model-name Lex-MFD-20 | tee results/Lex-MFD-20.txt

echo "===== Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection bertopic --model-name TD-BERTopic | tee results/TD-BERTopic.txt