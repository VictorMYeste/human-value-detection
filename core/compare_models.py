"""
Script to compare two multi-label models via Macro-F1 bootstrap and per-label McNemar tests.

Usage:
    python compare_models.py \
        --pred1 path/to/model1_predictions.tsv \
        --pred2 path/to/model2_predictions.tsv \
        --gold  path/to/gold_labels.tsv \
        [--id-cols Text-ID Sentence-ID] \
        [--prob-cols col1,col2,...] \
        [--threshold 0.5] \
        [--bootstrap B] \
        [--alpha 0.05]

Outputs:
 - Macro-F1 for each model
 - 100*(1-α)% bootstrap CI and p-value for ΔF1 = F1_2 - F1_1
 - Per-label McNemar p-values (positives only) with FDR correction
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def load_multilabel(path, id_cols, prob_cols=None, threshold=0.5):
    df = pd.read_csv(path, sep='\t')
    df.set_index(id_cols, inplace=True)
    cols = df.columns.tolist()
    if prob_cols:
        probs = prob_cols.split(',')
        missing = set(probs) - set(cols)
        if missing:
            raise ValueError(f"Missing probability columns in {path}: {missing}")
        return (df[probs] >= threshold).astype(int)
    else:
        float_cols = df.select_dtypes(include=['float']).columns.tolist()
        if float_cols:
            return (df[float_cols] >= threshold).astype(int)
        return df.astype(int)


def bootstrap_f1_lower(pred1, pred2, true, B=2000, alpha=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    n = true.shape[0]
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.choice(n, n, replace=True)
        f1_1 = f1_score(true[idx], pred1[idx], average='macro', zero_division=0)
        f1_2 = f1_score(true[idx], pred2[idx], average='macro', zero_division=0)
        deltas[i] = f1_2 - f1_1
    # one-sided lower confidence bound: the α percentile
    lower = np.percentile(deltas, 100 * alpha)
    # one-sided p-value for Model2 > Model1
    p_one = np.mean(deltas <= 0)
    return lower, p_one


def per_label_mcnemar(pred1, pred2, true, label_names, alpha=0.05):
    pvals = []
    names = []
    for name, col in zip(label_names, range(true.shape[1])):
        y_true = true[:, col]
        y1 = pred1[:, col]
        y2 = pred2[:, col]
        mask = (y_true == 1)
        if mask.sum() == 0:
            # no positives in gold → F1 is 0 for both, skip test
            continue
        c1 = (y1[mask] == y_true[mask])
        c2 = (y2[mask] == y_true[mask])
        a = np.sum(c1 & c2)
        b = np.sum(c1 & ~c2)
        c = np.sum(~c1 & c2)
        if b + c == 0:
            # identical performance on positives → nothing to test
            continue
        result = mcnemar([[a, b], [c, mask.sum() - a - b - c]], exact=True)
        pvals.append(result.pvalue)
        names.append(name)
    if not pvals:
        return []
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return list(zip(names, p_adj, reject))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred1',    required=True)
    parser.add_argument('--pred2',    required=True)
    parser.add_argument('--gold',     default="../data/test-english/labels-cat.tsv", help='Gold labels TSV')
    parser.add_argument('--id-cols',  nargs='+', default=['Text-ID','Sentence-ID'])
    parser.add_argument('--prob-cols', default=None)
    parser.add_argument('--threshold',type=float, default=0.5)
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Number of bootstrap samples')
    parser.add_argument('--alpha',     type=float, default=0.05)
    args = parser.parse_args()

    print(f"Model1 = {args.pred1}")
    print(f"Model2 = {args.pred2}\n")

    p1 = load_multilabel(args.pred1, args.id_cols, args.prob_cols, args.threshold)
    p2 = load_multilabel(args.pred2, args.id_cols, args.prob_cols, args.threshold)
    gold = load_multilabel(args.gold,  args.id_cols, args.prob_cols, args.threshold)

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