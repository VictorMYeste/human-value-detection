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