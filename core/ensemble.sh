#!/usr/bin/env bash

# Change route to results and the model list in Python files to save all ensemble results

mkdir ensemble_results/llm/self-enhancement-transformers
mkdir ../approaches/ensembles/llm/self-enhancement-transformers

echo "=========="
echo "=========="
echo "=========="
echo "===== Hard ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode hard --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/fixed-hard-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-fixed-hard.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Soft ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode soft --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/fixed-soft-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-fixed-soft.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Hard tuned ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode hard --use-tuned --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/tuned-hard-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-tuned-hard.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Soft tuned ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode soft --threshold 0.17 --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/tuned-soft-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-tuned-soft.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Weighted ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode weighted --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/fixed-weighted-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-fixed-weighted.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Weighted tuned ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode weighted --use-tuned --debug --save-preds ../approaches/ensembles/llm/self-enhancement-transformers/tuned-weighted-champion-test.tsv --subset "Hedonism,Achievement,Power: dominance,Power: resources,Face" --test | tee ensemble_results/llm/self-enhancement-transformers/test-tuned-weighted.txt
# echo "=========="
# echo "=========="
# echo "=========="
# echo "===== Hard per label ====="
# echo "=========="
# echo "=========="
# echo "=========="
# python3 ensemble_per_label.py --mode hard --debug | tee ensemble_results/llm/self-enhancement-transformers/per_label-hard.txt
# echo "=========="
# echo "=========="
# echo "=========="
# echo "===== Soft per label ====="
# echo "=========="
# echo "=========="
# echo "=========="
# python3 ensemble_per_label.py --mode soft --debug | tee ensemble_results/llm/self-enhancement-transformers/per_label-soft.txt