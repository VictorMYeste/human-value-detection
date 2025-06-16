#!/usr/bin/env bash

# Change route to results and the model list in Python files to save all ensemble results

echo "=========="
echo "=========="
echo "=========="
echo "===== Hard ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode hard --debug | tee ensemble_results/presence/Lex-LIWC-22_LingFeat_0.5-fixed-hard.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Soft ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode soft --debug | tee ensemble_results/presence/Lex-LIWC-22_LingFeat_0.5-fixed-soft.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Hard tuned ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode hard --use-tuned --debug | tee ensemble_results/presence/Lex-LIWC-22_LingFeat_0.5-tuned-hard.txt
echo "=========="
echo "=========="
echo "=========="
echo "===== Soft tuned ====="
echo "=========="
echo "=========="
echo "=========="
python3 -u ensemble_voting.py --mode soft --threshold 0.25 --debug | tee ensemble_results/presence/Lex-LIWC-22_LingFeat_0.5-tuned-soft.txt
# python3 ensemble_voting.py --mode weighted --debug | tee ensemble_results/presence/Baseline_0.5-all-weighted.txt
# python3 ensemble_per_label.py --mode hard --debug | tee ensemble_results/presence/Baseline_0.5-per_label-hard.txt
# python3 ensemble_per_label.py --mode soft --debug | tee ensemble_results/presence/Baseline_0.5-per_label-soft.txt
# python3 ensemble_per_label.py --mode weighted --debug | tee ensemble_results/presence/Baseline_0.5-per_label-weighted.txt