echo "===== NER + Lex - VAD ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --lexicon VAD | tee results/NER_Lex-VAD.txt

echo "===== NER + Lex - Emotion Intensity ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --lexicon EmotionIntensity | tee results/NER_Lex-EmotionIntensity.txt

echo "===== NER + Lex - LIWC 15 ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --lexicon LIWC | tee results/NER_Lex-LIWC-15.txt

echo "===== NER + TD - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --ner-features --topic-detection nmf | tee results/NER_Lex-TD-NMF.txt