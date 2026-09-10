#!/usr/bin/env python3
"""
Prevalence-controlled provenance comparison (IP&M resubmission).

analyze_provenance.py finds that macro-F1 is LOWER on English-original sentences
than on machine-translated ones -- the opposite of the direction Reviewer #4 and
the Associate Editor assumed. Before that can be reported, the obvious confound
has to be removed: the two strata differ sharply in value density (28.3% of
English-original sentences carry at least one value, versus 54.4% of translated
ones), and macro-F1 over 19 rare labels is strongly prevalence-dependent.

This script matches the translated stratum to the English-original stratum on the
per-sentence value-count distribution (0, 1, 2, 3+ values) by stratified
resampling, then recomputes macro-F1 over many replicates. Same idea as the
prevalence-controlled confusion-affinity analysis already in Section 6.7.

If the matched translated macro-F1 falls to the English-original level, the raw
gap is a composition artefact and there is no evidence of translation-induced
degradation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from analyze_provenance import VALUES, aligned, tune

RNG = np.random.default_rng(42)
N_REP = 1000


def macro_f1_fast(y: np.ndarray, pb: np.ndarray) -> float:
    """Macro average of positive-class F1 over labels; vectorised equivalent of
    core/evaluation.py's precision_recall_fscore_support(...)[2][1] loop."""
    tp = (y & pb).sum(0)
    fp = ((1 - y) & pb).sum(0)
    fn = (y & (1 - pb)).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), float), where=denom > 0)
    return float(f1.mean())


def check_equivalence(y, pb):
    ref = np.mean([
        precision_recall_fscore_support(
            y[:, j], pb[:, j], average=None, labels=[0, 1], zero_division=0
        )[2][1]
        for j in range(y.shape[1])
    ])
    fast = macro_f1_fast(y, pb)
    assert abs(ref - fast) < 1e-12, (ref, fast)
    return ref


def stratum_composition() -> None:
    """Report the value density of each stratum. This is the confound the matching
    below controls for, and the figures quoted in Section 6.9 of the paper."""
    from analyze_provenance import load_gold
    g = load_gold("test")
    lang = g["Text-ID"].str.split("_").str[0].to_numpy()
    Y = (g[VALUES].to_numpy() >= 0.5).astype(int)
    en = lang == "EN"
    print("stratum composition (test split)")
    print(f"  English-original    n = {en.sum():5d}   "
          f"{100*(Y[en].sum(1) > 0).mean():5.2f}% of sentences carry a value   "
          f"positive rate {100*Y[en].mean():.2f}%")
    print(f"  machine-translated  n = {(~en).sum():5d}   "
          f"{100*(Y[~en].sum(1) > 0).mean():5.2f}% of sentences carry a value   "
          f"positive rate {100*Y[~en].mean():.2f}%")
    print()


def main() -> None:
    stratum_composition()
    for name in ["Baseline", "Baseline-s7", "Baseline-s1701"]:
        t = tune(name)
        y, p, lang = aligned(name, "test", "test")
        pb = (p >= t).astype(int)
        is_en = lang == "EN"

        y_en, pb_en = y[is_en], pb[is_en]
        y_tr, pb_tr = y[~is_en], pb[~is_en]

        if name == "Baseline":
            check_equivalence(y_en, pb_en)
            print("vectorised macro-F1 verified identical to core/evaluation.py\n")

        raw_en = macro_f1_fast(y_en, pb_en)
        raw_tr = macro_f1_fast(y_tr, pb_tr)

        # --- match on per-sentence value count (0, 1, 2, 3+) ------------------
        def bucket(Y):
            c = Y.sum(1)
            return np.clip(c, 0, 3)

        b_en, b_tr = bucket(y_en), bucket(y_tr)
        target = {k: int((b_en == k).sum()) for k in range(4)}
        pools = {k: np.flatnonzero(b_tr == k) for k in range(4)}

        boot = []
        for _ in range(N_REP):
            idx = np.concatenate([
                RNG.choice(pools[k], size=target[k], replace=True)
                for k in range(4) if target[k] > 0
            ])
            boot.append(macro_f1_fast(y_tr[idx], pb_tr[idx]))
        boot = np.array(boot)

        print(f"{name}  (t* = {t:.2f})")
        print(f"  English-original            {raw_en:.4f}   (n={is_en.sum()})")
        print(f"  translated, raw             {raw_tr:.4f}   (n={(~is_en).sum()})")
        print(f"  translated, prevalence-matched  {boot.mean():.4f} "
              f"[95% CI {np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
        print(f"  raw gap        (EN - translated)        {raw_en - raw_tr:+.4f}")
        print(f"  controlled gap (EN - matched)           {raw_en - boot.mean():+.4f}")
        inside = np.percentile(boot, 2.5) <= raw_en <= np.percentile(boot, 97.5)
        print(f"  English-original inside matched 95% CI? {inside}")
        print(f"  value-count distribution matched: {target}")
        print()


if __name__ == "__main__":
    main()
