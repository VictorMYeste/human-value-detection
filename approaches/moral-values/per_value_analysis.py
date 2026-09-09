#!/usr/bin/env python3
"""
Per-value failure-mode analysis (IP&M resubmission).

Reviewer #4 (specific comment 8) asks for a deeper analysis of failure modes and
class-specific behaviour. The manuscript already states qualitatively that rare
values are harder; this quantifies that, and identifies the values that break the
pattern, which is where the interesting behaviour is.

Reports per-value test F1 for the calibrated text-only baseline (mean over seeds
42, 7, 1701, each at its own tuned global threshold), against per-value
prevalence in the test split.

Usage:
    python3 per_value_analysis.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from analyze_provenance import VALUES, aligned, tune, load_gold

SEEDS = ["Baseline", "Baseline-s7", "Baseline-s1701"]


def main() -> None:
    per_seed = []
    for n in SEEDS:
        t = tune(n)
        y, p, _ = aligned(n, "test", "test")
        pb = (p >= t).astype(int)
        tp = (y & pb).sum(0); fp = ((1 - y) & pb).sum(0); fn = (y & (1 - pb)).sum(0)
        den = 2 * tp + fp + fn
        per_seed.append(np.divide(2 * tp, den, out=np.zeros(len(VALUES), float), where=den > 0))

    F = np.array(per_seed)
    mean, sd = F.mean(0), F.std(0, ddof=1)
    Y = (load_gold("test")[VALUES].to_numpy() >= 0.5).astype(int)
    prev, npos = Y.mean(0) * 100, Y.sum(0)

    print(f"{'value':32s}{'prev%':>7s}{'n+':>7s}{'F1':>8s}{'sd':>7s}")
    for j in np.argsort(-prev):
        print(f"{VALUES[j]:32s}{prev[j]:7.2f}{npos[j]:7d}{mean[j]:8.3f}{sd[j]:7.3f}")

    r, pv = stats.spearmanr(prev, mean)
    lo, hi = prev < 2, prev >= 4
    print(f"\nSpearman rho(prevalence, per-value F1) = {r:.3f}, p = {pv:.4f}")
    print(f"macro-F1 = {mean.mean():.4f}")
    print(f"mean F1, rare (<2%,  {lo.sum()} values): {mean[lo].mean():.3f}")
    print(f"mean F1, frequent (>=4%, {hi.sum()} values): {mean[hi].mean():.3f}")

    # Values that break the prevalence pattern, which is where the insight is.
    resid = stats.zscore(mean) - stats.zscore(prev)
    print("\nover-performing relative to prevalence:",
          [VALUES[j] for j in np.argsort(-resid)[:3]])
    print("under-performing relative to prevalence:",
          [VALUES[j] for j in np.argsort(resid)[:3]])


if __name__ == "__main__":
    main()
