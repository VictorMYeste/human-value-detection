#!/bin/bash

echo "===== 2 prev sentences with label + NER  ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --ner-features | tee results/Previous-Sentences-2_NER.txt

echo "===== ResidualBlock ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --residualblock | tee results/ResidualBlock.txt

echo "===== CustomStopwords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --customstopwords | tee results/CustomStopwords.txt