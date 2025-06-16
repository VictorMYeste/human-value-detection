"""
Script to compare two multi-label models via Macro-F1 bootstrap and per-label McNemar tests.

Usage:
    python3 compare_models.py \
        --pred1 path/to/model1_predictions.tsv \
        --th1 0.5 \
        --pred2 path/to/model2_predictions.tsv \
         --th2 0.5 \
        --gold  path/to/gold_labels.tsv \
        [--id-cols Text-ID Sentence-ID] \
        [--prob-cols col1,col2,...] \
        [--bootstrap B] \
        [--alpha 0.05]

Outputs:
 - Macro-F1 for each model
 - 100*(1-α)% bootstrap CI and p-value for ΔF1 = F1_2 - F1_1
 - Per-label McNemar p-values (positives only) with FDR correction
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import argparse
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests
from core.eval_utils import load_multilabel, bootstrap_f1_lower, per_label_mcnemar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred1',    required=True)
    parser.add_argument('--th1',type=float, default=0.5, help='Probability threshold for pred1 binarization')
    parser.add_argument('--pred2',    required=True)
    parser.add_argument('--th2',type=float, default=0.5, help='Probability threshold for pred2 binarization')
    parser.add_argument('--gold',     default="../data/test-english/labels-cat.tsv", help='Gold labels TSV')
    parser.add_argument('--id-cols',  nargs='+', default=['Text-ID','Sentence-ID'])
    parser.add_argument('--prob-cols', default=None)
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Number of bootstrap samples')
    parser.add_argument('--alpha',     type=float, default=0.05)
    args = parser.parse_args()

    print(f"Model1 = {args.pred1}")
    print(f"Model1 Threshold = {args.th1}")
    print(f"Model2 = {args.pred2}")
    print(f"Model2 Threshold = {args.th2}\n")

    p1 = load_multilabel(args.pred1, args.id_cols, args.prob_cols, args.th1)
    p2 = load_multilabel(args.pred2, args.id_cols, args.prob_cols, args.th2)
    # gold = load_multilabel(args.gold,  args.id_cols, args.prob_cols, args.thresholdgold)
    gold_df = pd.read_csv(args.gold, sep='\t')
    gold_df.set_index(args.id_cols, inplace=True)
    gold = gold_df.astype(int)

    labels_1 = p1.columns.tolist()
    labels_2 = p2.columns.tolist()
    missing = set(labels_2) - set(labels_1)
    if missing:
        raise ValueError(f"Model 2 missing labels: {missing}")
    
    missing = set(labels_2) - set(gold.columns)
    if missing:
        raise ValueError(f"Gold missing labels: {missing}")

    df = p1.join(p2, how='inner', lsuffix='_1', rsuffix='_2')
    df = df.join(gold[labels_2], how='inner')
    if df.empty:
        print("No overlapping rows after join.")
        sys.exit(1)

    pred1 = df[[f"{l}_1" for l in labels_2]].values
    pred2 = df[[f"{l}_2" for l in labels_2]].values
    true  = df[labels_2].values

    # Macro-F1
    f1_1 = f1_score(true, pred1, average='macro', zero_division=0)
    f1_2 = f1_score(true, pred2, average='macro', zero_division=0)
    print(f"Model1 Macro-F1 = {f1_1:.5f}")
    print(f"Model2 Macro-F1 = {f1_2:.5f}\n")

    # Bootstrap one-sided lower bound
    lower, p_boot = bootstrap_f1_lower(pred1, pred2, true,
                                       B=args.bootstrap, alpha=args.alpha)
    significance = "significant" if p_boot < args.alpha else "not significant"
    print(f"ΔF1 one-sided lower {100*(1-args.alpha):.0f}% bound = {lower:.4f}")
    print(f"one-sided p(Model2>Model1) = {p_boot:.4f} -> {significance} at α={args.alpha}\n")

    # Per-label McNemar on positives
    results = per_label_mcnemar(pred1, pred2, true, labels_2, alpha=args.alpha)
    if results:
        print("Per-label McNemar (FDR-corrected): label_index, p_adj, significant")
        for name, p_adj, sig in results:
            print(f"{name:>3}, {p_adj:.4f}, {sig}")
    else:
        print("No per-label tests performed (no discordant positives).")

if __name__ == '__main__':
    main()