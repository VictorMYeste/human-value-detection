#!/bin/bash

echo "===== ResidualBlock ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --residualblock | tee results/ResidualBlock.txt

echo "===== Data Augmentation ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --augment-data | tee results/Data-Augmentation.txt

echo "===== CustomStopwords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --customstopwords | tee results/CustomStopwords.txt