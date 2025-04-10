echo "===== 2 prev sentences with label + NER  ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --ner-features | tee results/Previous-Sentences-2_NER.txt

echo "===== 2 prev sentences with label + Schwartz  ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon  | tee results/Previous-Sentences-2_Lex-Schwartz.txt

echo "===== 2 prev sentences with label + Lex - Schwartz ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon Schwartz | tee results/Previous-Sentences-2_Lex-Schwartz.txt

echo "===== 2 prev sentences with label + Lex - EmoLex ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon EmoLex | tee results/Previous-Sentences-2_Lex-EmoLex.txt

echo "===== 2 prev sentences with label + Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon EmotionIntensity | tee results/Previous-Sentences-2_Lex-EmotionIntensity.txt

echo "===== 2 prev sentences with label + Lex - WorryWords ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon WorryWords | tee results/Previous-Sentences-2_Lex-WorryWords.txt

echo "===== 2 prev sentences with label + Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon LIWC | tee results/Previous-Sentences-2_Lex-LIWC.txt

echo "===== 2 prev sentences with label + Lex - MFD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon MFD | tee results/Previous-Sentences-2_Lex-MFD.txt

echo "===== 2 prev sentences with label + Lex - LIWC 22 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon LIWC-22 | tee results/Previous-Sentences-2_Lex-LIWC-22.txt

echo "===== 2 prev sentences with label + Lex - MFD-20 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --lexicon MFD-20 | tee results/Previous-Sentences-2_Lex-MFD-20.txt

echo "===== 2 prev sentences with label + Topic Detection - LDA ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection lda | tee results/Previous-Sentences-2_TD-LDA.txt

echo "===== 2 prev sentences with label + Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection nmf | tee results/Previous-Sentences-2_TD-NMF.txt

echo "===== 2 prev sentences with label + Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection bertopic | tee results/Previous-Sentences-2_TD-BERTopic-v2.txt
