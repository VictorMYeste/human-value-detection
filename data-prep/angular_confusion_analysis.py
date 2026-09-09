#!/usr/bin/env python3
"""
A2 — Quantitative continuum-confusability analysis (R2-C4 / AE#4).

Tests whether pairwise confusion errors of the best model concentrate among
values that are close on the Schwartz circle.

How confusions are counted (stated explicitly per the reviewer's request):
  For each test sentence, let FN = gold-positive values the model missed and
  FP = values the model predicted but that are absent from gold. Each ordered
  pair (w in FN, v in FP), w != v, counts as one substitution w->v. The
  symmetric pairwise confusion count is M[a,b] = C[a->b] + C[b->a].

Angular distance uses the canonical refined-Schwartz circular order
(Schwartz, 2012; the order of Fig. 1). Distance = circular index gap in
{1..9}; degrees = gap * 360/19.

Usage (run from repo root, gold must be available):
  python3 data-prep/angular_confusion_analysis.py \
      --pred approaches/moral-values/output/direct_champion-tuned-soft-champion-test.tsv \
      --th 0.29 --gold data/test-english/labels-cat.tsv
"""
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

ID = ["Text-ID", "Sentence-ID"]

# Canonical refined-Schwartz circular order (Schwartz, 2012).
CIRCLE = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    "Achievement", "Power: dominance", "Power: resources", "Face",
    "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
    "Conformity: interpersonal", "Humility", "Benevolence: dependability",
    "Benevolence: caring", "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance",
]
N = len(CIRCLE)
POS = {v: i for i, v in enumerate(CIRCLE)}


def circ_gap(a, b):
    d = abs(POS[a] - POS[b])
    return min(d, N - d)            # 1..9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--th", type=float, default=0.29)
    ap.add_argument("--gold", default="data/test-english/labels-cat.tsv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--perm", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = pd.read_csv(args.gold, sep="\t").set_index(ID)[CIRCLE].astype(int)
    pr = pd.read_csv(args.pred, sep="\t").set_index(ID)
    pred = (pr[CIRCLE] >= args.th).astype(int)
    gold, pred = gold.align(pred, join="inner", axis=0)
    print(f"Aligned test sentences: {len(gold)}  (threshold t*={args.th})")

    G, P = gold.values, pred.values
    fn = (G == 1) & (P == 0)        # missed gold values
    fp = (G == 0) & (P == 1)        # spurious predicted values

    C = np.zeros((N, N))
    for i in range(len(G)):
        w = np.where(fn[i])[0]
        v = np.where(fp[i])[0]
        for a in w:
            for b in v:
                if a != b:
                    C[a, b] += 1
    M = C + C.T                     # symmetric pairwise confusion

    pairs = list(combinations(range(N), 2))
    conf = np.array([M[a, b] for a, b in pairs])
    gap = np.array([circ_gap(CIRCLE[a], CIRCLE[b]) for a, b in pairs])
    deg = gap * 360.0 / N

    rho, p_rho = spearmanr(gap, conf)
    r, p_r = pearsonr(deg, conf)
    print("\n=== Angular distance vs pairwise confusion (171 value pairs) ===")
    print(f"Spearman rho (gap vs confusion) = {rho:.3f}  (p={p_rho:.2e})")
    print(f"Pearson  r   (deg vs confusion) = {r:.3f}  (p={p_r:.2e})")

    order = np.argsort(-conf)
    print(f"\n=== Top-{args.topk} most-confused pairs ===")
    print(f"{'count':>6}  {'gap':>3} {'deg':>6}   pair")
    topk_gaps = []
    for idx in order[:args.topk]:
        a, b = pairs[idx]
        topk_gaps.append(gap[idx])
        print(f"{conf[idx]:6.0f}  {gap[idx]:3d} {deg[idx]:6.1f}   {CIRCLE[a]}  <->  {CIRCLE[b]}")
    topk_gaps = np.array(topk_gaps)

    mean_top = topk_gaps.mean()
    mean_all = gap.mean()
    # Confusion-weighted mean angular gap over all pairs (how far the average error is)
    mean_w = float(np.average(gap, weights=conf)) if conf.sum() > 0 else float("nan")
    print(f"\nMean circular gap: top-{args.topk} = {mean_top:.2f} | all pairs = {mean_all:.2f} "
          f"| confusion-weighted = {mean_w:.2f}  (max gap = 9)")

    # Permutation test: is the top-k mean gap smaller than random k-subsets of pairs?
    rng = np.random.default_rng(args.seed)
    npairs = len(pairs)
    rand_means = np.array([
        gap[rng.choice(npairs, args.topk, replace=False)].mean()
        for _ in range(args.perm)
    ])
    p_perm = (rand_means <= mean_top).mean()
    print(f"Permutation p(random top-{args.topk} mean gap <= observed) = {p_perm:.4f} "
          f"over {args.perm} samples")

    # --- Prevalence-controlled confusion affinity (lift = M / (E_a*E_b)) -------
    # Removes the effect that frequent values dominate raw confusion counts.
    E = M.sum(axis=1)
    lift = np.array([
        M[a, b] / (E[a] * E[b]) if E[a] > 0 and E[b] > 0 else np.nan
        for a, b in pairs
    ])
    ok = ~np.isnan(lift)
    rho_l, p_l = spearmanr(gap[ok], lift[ok])
    print("\n=== Prevalence-controlled confusion affinity (lift = M/(E_a*E_b)) ===")
    print(f"Spearman rho (gap vs lift) = {rho_l:.3f}  (p={p_l:.2e})  over {ok.sum()} pairs")
    order_l = np.argsort(-np.where(ok, lift, -np.inf))
    print(f"\nTop-{args.topk} pairs by confusion affinity (prevalence-controlled):")
    topk_gaps_l = []
    for idx in order_l[:args.topk]:
        a, b = pairs[idx]
        topk_gaps_l.append(gap[idx])
        print(f"  lift={lift[idx]*1e4:5.2f}e-4  gap={gap[idx]} ({deg[idx]:5.1f} deg)  "
              f"{CIRCLE[a]}  <->  {CIRCLE[b]}")
    topk_gaps_l = np.array(topk_gaps_l)
    rand_means_l = np.array([
        gap[rng.choice(npairs, args.topk, replace=False)].mean() for _ in range(args.perm)
    ])
    p_perm_l = (rand_means_l <= topk_gaps_l.mean()).mean()
    print(f"\nMean circular gap of top-{args.topk} by affinity = {topk_gaps_l.mean():.2f} "
          f"| all pairs = {mean_all:.2f}")
    print(f"Permutation p(random top-{args.topk} mean gap <= observed) = {p_perm_l:.4f}")


if __name__ == "__main__":
    main()
