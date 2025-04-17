#!/bin/bash

echo "===== Lex - Schwartz ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon Schwartz | tee results/Lex-Schwartz.txt

echo "===== Lex - MFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD | tee results/Lex-MFD.txt

echo "===== Lex - LIWC 22 + Linguistic Features ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 --linguistic-features | tee results/Lex-LIWC-22_LingFeat.txt
