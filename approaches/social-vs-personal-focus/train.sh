echo "===== 2 prev sentences with label + NER  ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --ner-features | tee results/Previous-Sentences-2_NER.txt

echo "===== 2 prev sentences with label + Topic Detection - LDA ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection lda | tee results/Previous-Sentences-2_TD-LDA.txt

echo "===== 2 prev sentences with label + Topic Detection - NMF ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection nmf | tee results/Previous-Sentences-2_TD-NMF.txt

echo "===== 2 prev sentences with label + Topic Detection - BERTopic ====="
accelerate launch --multi_gpu main.py -t ../../data/training-english/ -v ../../data/validation-english/ -s 42 --previous-sentences --topic-detection bertopic | tee results/Previous-Sentences-2_TD-BERTopic.txt