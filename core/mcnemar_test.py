"""
Script to compute Macro F1 for two multi-label models and perform McNemar's test on per-label correctness.

Usage:
    python mcnemar_test.py \
        --pred1 path/to/model1_predictions.tsv \
        --pred2 path/to/model2_predictions.tsv \
        --gold  path/to/gold_labels.tsv \
        [--id-cols Text-ID Sentence-ID] \
        [--prob-cols col1,col2,...] \
        [--threshold 0.5] \
        [--exact]

- pred1, pred2: TSVs with model predictions; columns after id-cols are label names.
- gold: TSV with gold binary labels; can have extra columns.
- id-cols: columns to merge on (default: Text-ID Sentence-ID).
- prob-cols: comma-separated list of probability columns; overrides auto-detection.
- threshold: probability cutoff to binarize (default 0.5).
- exact: flag to use exact McNemar test (default asymptotic).

Outputs:
- Macro F1 score for each model.
- Contingency table of per-(instance,label) correctness.
- McNemar statistic, p-value, and significance at α=0.05.
"""
import argparse
import sys
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import f1_score

def load_multilabel(path, id_cols, prob_cols=None, threshold=0.5):
    df = pd.read_csv(path, sep='\t')
    df.set_index(id_cols, inplace=True)
    cols = df.columns.tolist()
    # Determine binarization strategy
    if prob_cols:
        probs = prob_cols.split(',')
        missing = set(probs) - set(cols)
        if missing:
            raise ValueError(f"Missing probability columns in {path}: {missing}")
        df_bin = (df[probs] >= threshold).astype(int)
    else:
        # auto-detect float => probabilities
        float_cols = df.select_dtypes(include=['float']).columns.tolist()
        if float_cols:
            df_bin = (df[float_cols] >= threshold).astype(int)
        else:
            df_bin = df.astype(int)
    return df_bin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred1', required=True, help='Model 1 TSV')
    parser.add_argument('--pred2', required=True, help='Model 2 TSV')
    parser.add_argument('--gold',     default="../data/test-english/labels-cat.tsv", help='Gold labels TSV')
    parser.add_argument('--id-cols', nargs='+', default=['Text-ID','Sentence-ID'],
                        help='Columns to merge on')
    parser.add_argument('--prob-cols', default=None,
                        help='Comma-separated probability columns')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for probabilities')
    parser.add_argument('--exact', action='store_true',
                        help='Use exact McNemar test')
    args = parser.parse_args()

    # Load dataframes
    p1 = load_multilabel(args.pred1, args.id_cols, args.prob_cols, args.threshold)
    p2 = load_multilabel(args.pred2, args.id_cols, args.prob_cols, args.threshold)
    gold = load_multilabel(args.gold,  args.id_cols, args.prob_cols, args.threshold)

    # Ensure predictions share labels
    if not p1.columns.equals(p2.columns):
        raise ValueError("Prediction files have different label columns")
    labels = p1.columns.tolist()

    # Check gold contains needed labels
    missing = set(labels) - set(gold.columns)
    if missing:
        raise ValueError(f"Gold file is missing label columns: {missing}")

    # Align and join
    joined = p1.join(p2, how='inner', lsuffix='_1', rsuffix='_2')
    joined = joined.join(gold[labels], how='inner')
    if joined.empty:
        print("No overlapping samples to compare after merge.")
        sys.exit(1)

    # Arrays
    pred1 = joined[[f"{lbl}_1" for lbl in labels]].values
    pred2 = joined[[f"{lbl}_2" for lbl in labels]].values
    true  = joined[labels].values

    # Macro F1
    f1_1 = f1_score(true, pred1, average='macro', zero_division=0)
    f1_2 = f1_score(true, pred2, average='macro', zero_division=0)

    # Contingency on correctness
    correct1 = (pred1 == true).ravel()
    correct2 = (pred2 == true).ravel()
    a = int(((correct1) & (correct2)).sum())
    b = int(((correct1) & (~correct2)).sum())
    c = int(((~correct1) & (correct2)).sum())
    d = int(((~correct1) & (~correct2)).sum())

    # Handle zero discordant
    if (b + c) == 0:
        print(f"Model1 Macro F1 = {f1_1:.4f}")
        print(f"Model2 Macro F1 = {f1_2:.4f}\n")
        print("No discordant pairs (b+c=0). Per-instance–label correctness is identical — McNemar test not applicable.")
        sys.exit(0)

    # McNemar's test
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=args.exact)
    alpha = 0.05

    # Output
    print(f"Model1 Macro F1 = {f1_1:.4f}")
    print(f"Model2 Macro F1 = {f1_2:.4f}\n")
    print("Contingency table (correctness per instance-label):")
    print(f"                 Model2 correct    Model2 wrong")
    print(f"Model1 correct       {a:>7}              {b:>7}")
    print(f"Model1 wrong         {c:>7}              {d:>7}\n")
    print(f"McNemar statistic = {result.statistic:.4f}")
    print(f"P-value            = {result.pvalue:.4f}")
    if result.pvalue < alpha:
        print(f"p < {alpha}: significant difference detected.")
    else:
        print(f"p >= {alpha}: no significant difference.")

if __name__ == '__main__':
    main()
