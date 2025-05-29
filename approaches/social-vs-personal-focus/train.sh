#!/bin/bash

echo "===== 2 prev sentences with label ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --model-name Previous-Sentences-2 | tee results/Previous-Sentences-2.txt

echo "===== NER ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --model-name NER | tee results/NER.txt

echo "===== Lex - Schwartz ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon Schwartz --model-name Lex-Schwartz | tee results/Lex-Schwartz.txt

echo "===== Lex - WorryWords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon WorryWords --model-name Lex-WorryWords | tee results/Lex-WorryWords.txt

echo "===== Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC --model-name Lex-LIWC | tee results/Lex-LIWC.txt

echo "===== Lex - LIWC 22 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 --model-name Lex-LIWC-22 | tee results/Lex-LIWC-22.txt

echo "===== Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection nmf --model-name TD-NMF | tee results/TD-NMF.txt

echo "===== Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection bertopic --model-name TD-BERTopic | tee results/TD-BERTopic.txt
