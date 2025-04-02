#!/bin/bash

# echo "===== Baseline ====="
# accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 | tee results/Baseline.txt

echo "===== Token Pruning (TP, IDF = 3.0) ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --token-pruning | tee results/Token-Pruning-3.0.txt

echo "===== 2 prev sentences with label ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences | tee results/Previous-Sentences-2.txt

echo "===== MultiLayer ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --multilayer | tee results/MultiLayer.txt

# Residual Block (Manual)

echo "===== Data Augmentation ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --augment-data | tee results/Data-Augmentation.txt

# Custom Stopwords (Manual)

echo "===== NER ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features | tee results/NER.txt

echo "===== Lex - Schwartz ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon Schwartz | tee results/Lex-Schwartz.txt

echo "===== Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon VAD | tee results/Lex-VAD.txt

echo "===== Lex - EmoLex ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon EmoLex | tee results/Lex-EmoLex.txt

echo "===== Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon EmotionIntensity | tee results/Lex-EmotionIntensity.txt

echo "===== Lex - WorryWords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon WorryWords | tee results/Lex-WorryWords.txt

echo "===== Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC | tee results/Lex-LIWC.txt

echo "===== Lex - MFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD | tee results/Lex-MFD.txt

echo "===== Lex - LIWC 22 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 | tee results/Lex-LIWC-22.txt

echo "===== Lex - LIWC 22 + Linguistic Features ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon LIWC-22 --linguistic-features | tee results/Lex-LIWC-22_LingFeat.txt

echo "===== Lex - MFD-20 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MFD-20 | tee results/Lex-MFD-20.txt

echo "===== Lex - eMFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon eMFD | tee results/Lex-eMFD.txt

echo "===== Lex - MJD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --lexicon MJD | tee results/Lex-MJD.txt

echo "===== Topic Detection - LDA ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection lda | tee results/TD-LDA.txt

echo "===== Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection nmf | tee results/TD-NMF.txt

echo "===== Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --topic-detection bertopic | tee results/TD-BERTopic-v2.txt

###################

# echo "===== 2 prev sentences with label + Token Pruning (TP, IDF = 3.0)  ====="
# accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --token-pruning | tee results/Previous-Sentences-2-TokenPruning-3.0-v2.txt

# echo "===== 2 prev sentences with label + Lex - MFD  ====="
# accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon MFD | tee results/Previous-Sentences-2-Lex-MFD.txt

# echo "===== 2 prev sentences with label + Lex - EmoLex  ====="
# accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon EmoLex | tee results/Previous-Sentences-2-Lex-EmoLex.txt