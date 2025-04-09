#!/bin/bash

echo "===== Baseline ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --model-name Baseline | tee results/Baseline.txt

echo "===== Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon VAD --model-name Lex-VAD | tee results/Lex-VAD.txt