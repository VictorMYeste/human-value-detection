#!/bin/bash

# echo "=========="
# echo "===== Baseline ====="
# echo "=========="
# echo "----- Predicting Validation -----"
# python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Baseline
# echo "----- Predicting Test -----"
# python3 predict.py --test-dataset ../../data/test-english/ --model-name Baseline
# echo "----- Extracting per-label thresholds from Validation -----"
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Baseline
# echo "----- Evaluating Test -----"
# python3 eval.py --test-dataset ../../data/test-english/ --model-name Baseline

echo "=========="
echo "===== Lex-LIWC-22_LingFeat ====="
echo "=========="
echo "----- Predicting Validation -----"
python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features
echo "----- Predicting Test -----"
python3 predict.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features
echo "----- Extracting per-label thresholds from Validation -----"
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features
echo "----- Evaluating Test -----"
python3 eval.py --test-dataset ../../data/test-english/ --model-name Lex-LIWC-22_LingFeat --lexicon LIWC-22 --linguistic-features

# echo "=========="
# echo "===== Previous-Sentences-2-Lex-EmoLex ====="
# echo "=========="
# echo "----- Predicting Validation -----"
# python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-EmoLex --lexicon EmoLex --previous-sentences
# echo "----- Predicting Test -----"
# python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-EmoLex --lexicon EmoLex --previous-sentences
# echo "----- Extracting per-label thresholds from Validation -----"
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-EmoLex
# echo "----- Evaluating Test -----"
# python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-EmoLex

# echo "=========="
# echo "===== Previous-Sentences-2-Lex-LIWC-22 ====="
# echo "=========="
# echo "----- Predicting Validation -----"
# python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-LIWC-22 --lexicon LIWC-22 --previous-sentences
# echo "----- Predicting Test -----"
# python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-LIWC-22 --lexicon LIWC-22 --previous-sentences
# echo "----- Extracting per-label thresholds from Validation -----"
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-LIWC-22
# echo "----- Evaluating Test -----"
# python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-LIWC-22

# echo "=========="
# echo "===== Previous-Sentences-2-Lex-eMFD ====="
# echo "=========="
# echo "----- Predicting Validation -----"
# python3 predict.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-eMFD --lexicon eMFD --previous-sentences
# echo "----- Predicting Test -----"
# python3 predict.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-eMFD --lexicon eMFD --previous-sentences
# echo "----- Extracting per-label thresholds from Validation -----"
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Previous-Sentences-2-Lex-eMFD
# echo "----- Evaluating Test -----"
# python3 eval.py --test-dataset ../../data/test-english/ --model-name Previous-Sentences-2-Lex-eMFD