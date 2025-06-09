#!/bin/bash

echo "===== Baseline ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --model-name Baseline | tee results/Baseline.txt

echo "===== Token Pruning (TP, IDF = 3.0) ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --token-pruning --model-name Token-Pruning-3.0 | tee results/Token-Pruning-3.0.txt

echo "===== 2 prev sentences with label ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --model-name Previous-Sentences-2 | tee results/Previous-Sentences-2.txt

echo "===== MultiLayer ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --multilayer --model-name MultiLayer | tee results/MultiLayer.txt

echo "===== ResidualBlock ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --residualblock --model-name ResidualBlock | tee results/ResidualBlock.txt

echo "===== Data Augmentation ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --augment-data --model-name Data-Augmentation | tee results/Data-Augmentation.txt

echo "===== CustomStopwords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --customstopwords --model-name CustomStopwords | tee results/CustomStopwords.txt

echo "===== NER ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --model-name NER | tee results/NER.txt

echo "===== Lex - Schwartz ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon Schwartz --model-name Lex-Schwartz | tee results/Lex-Schwartz.txt

echo "===== Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon VAD --model-name Lex-VAD | tee results/Lex-VAD.txt

echo "===== Lex - EmoLex ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon EmoLex --model-name Lex-EmoLex | tee results/Lex-EmoLex.txt

echo "===== Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon EmotionIntensity --model-name Lex-EmotionIntensity | tee results/Lex-EmotionIntensity.txt

echo "===== Lex - WorryWords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon WorryWords --model-name Lex-WorryWords | tee results/Lex-WorryWords.txt

echo "===== Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC --model-name Lex-LIWC | tee results/Lex-LIWC.txt

echo "===== Lex - MFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD --model-name Lex-MFD | tee results/Lex-MFD.txt

echo "===== Lex - LIWC 22 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 --model-name Lex-LIWC-22 | tee results/Lex-LIWC-22.txt

echo "===== Lex - LIWC 22 + Linguistic Features ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 --linguistic-features --model-name Lex-LIWC-22_LingFeat | tee results/Lex-LIWC-22_LingFeat.txt

echo "===== Lex - MFD-20 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD-20 --model-name Lex-MFD-20 | tee results/Lex-MFD-20.txt

echo "===== Lex - eMFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon eMFD --model-name Lex-eMFD | tee results/Lex-eMFD.txt

echo "===== Lex - MJD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MJD --model-name Lex-MJD | tee results/Lex-MJD.txt

echo "===== Topic Detection - LDA ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection lda --model-name TD-LDA | tee results/TD-LDA.txt

echo "===== Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection nmf --model-name TD-NMF | tee results/TD-NMF.txt

echo "===== Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection bertopic --model-name TD-BERTopic | tee results/TD-BERTopic.txt