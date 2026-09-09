#!/usr/bin/env python3
"""
Paired per-seed tests across seed-matched configurations (IP&M resubmission).

Reviewer #1 (weakness 2, second part) argues that gains of ~0.007-0.009 macro-F1
"could stem from random variance", and that "bootstrapping on test samples alone
cannot substitute for experiments using multiple random seeds". That concern is
legitimate: the manuscript's significance tests bootstrap over test *sentences*
with the seed held fixed, which estimates sampling noise but not seed noise.

This script adds the missing analysis. For every contrast it pairs the two
configurations *within* each seed -- each at its own validation-tuned threshold,
exactly as in the published protocol -- and tests the per-seed differences.

Honest statement of power: three seeds gives a paired t-test 2 degrees of
freedom. That is weak, and no p-value here should be read as strong evidence on
its own. The informative quantities are (i) whether the difference keeps the same
sign across all three seeds, and (ii) how the effect compares with the
seed-to-seed standard deviation of the configurations being compared. Both are
reported alongside the test.

Usage:
    python3 paired_seed_tests.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_provenance import VALUES, GRID, KEY, load_gold

HERE = Path(__file__).resolve().parent
DIRECT = HERE / "output"
GATED = HERE / ".." / "p_moral-values" / "output"

SEEDS = ["", "-s7", "-s1701"]           # "" == seed 42 (published artefacts)
SEED_NAMES = ["42", "7", "1701"]


def load_pred(d: Path, name: str, tag: str) -> pd.DataFrame:
    return pd.read_csv(d / f"{name}-{tag}.tsv", sep="\t")[KEY + VALUES]


def aligned(d: Path, name: str, split: str, tag: str):
    gold = load_gold(split)
    merged = gold.merge(load_pred(d, name, tag), on=KEY, suffixes=("_g", "_p"))
    y = (merged[[f"{v}_g" for v in VALUES]].to_numpy() >= 0.5).astype(int)
    p = merged[[f"{v}_p" for v in VALUES]].to_numpy(dtype=float)
    return y, p


def macro_f1(y: np.ndarray, p: np.ndarray, t: float) -> float:
    pb = (p >= t).astype(int)
    tp = (y & pb).sum(0); fp = ((1 - y) & pb).sum(0); fn = (y & (1 - pb)).sum(0)
    den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros(len(tp), float), where=den > 0).mean())


def tuned_t(d: Path, name: str) -> float:
    y, p = aligned(d, name, "validation", "val")
    return max(GRID, key=lambda t: macro_f1(y, p, t))


def score(d: Path, name: str, t: float | None = None) -> tuple[float, float]:
    """Test macro-F1 at tuned t* (or at a fixed t if given)."""
    t_use = tuned_t(d, name) if t is None else t
    y, p = aligned(d, name, "test", "test")
    return macro_f1(y, p, t_use), t_use


def contrast(title: str, arm_a, arm_b, note: str = "") -> None:
    """arm_x = (label, dir, name_template, fixed_t or None)"""
    la, da, ta, fa = arm_a
    lb, db, tb, fb = arm_b
    a_scores, b_scores, ts = [], [], []
    for suf, sname in zip(SEEDS, SEED_NAMES):
        try:
            sa, t_a = score(da, ta.format(s=suf), fa)
            sb, t_b = score(db, tb.format(s=suf), fb)
        except FileNotFoundError as e:
            print(f"  seed {sname}: SKIP (missing {Path(str(e)).name})")
            continue
        a_scores.append(sa); b_scores.append(sb); ts.append((sname, t_a, t_b))
    if len(a_scores) < 2:
        print(f"\n{title}\n  insufficient seed-matched runs\n"); return

    a = np.array(a_scores); b = np.array(b_scores); d = b - a
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    for (sname, t_a, t_b), x, y_, dd in zip(ts, a, b, d):
        print(f"  seed {sname:>4s}:  {la} = {x:.4f} (t={t_a:.2f})   "
              f"{lb} = {y_:.4f} (t={t_b:.2f})   delta = {dd:+.4f}")
    print(f"  mean delta        {d.mean():+.4f}  (std {d.std(ddof=1):.4f})")
    print(f"  seed std          {la}: {a.std(ddof=1):.4f}   {lb}: {b.std(ddof=1):.4f}")
    print(f"  same sign in all {len(d)} seeds: {bool(np.all(np.sign(d) == np.sign(d[0])))}")
    if len(d) >= 2:
        t_stat, p_val = stats.ttest_rel(b, a)
        print(f"  paired t-test     t = {t_stat:+.3f}, p = {p_val:.4f}  (df = {len(d)-1})")
    print(f"  |mean delta| / mean seed std = "
          f"{abs(d.mean()) / max(np.mean([a.std(ddof=1), b.std(ddof=1)]), 1e-9):.2f}")


def main() -> None:
    print("=" * 78)
    print("PAIRED PER-SEED TESTS  (seeds 42, 7, 1701; each arm at its own tuned t*)")
    print("=" * 78)
    print("Three seeds -> 2 df. Read sign-consistency and effect-vs-seed-noise,")
    print("not the p-value alone.")

    print("\n" + "-" * 78)
    print("A. THRESHOLD CALIBRATION  (the paper's headline claim)")
    print("-" * 78)
    contrast("Baseline: default t=0.5  vs  validation-tuned t*",
             ("t=0.5", DIRECT, "Baseline{s}", 0.5),
             ("t*",    DIRECT, "Baseline{s}", None))

    print("\n" + "-" * 78)
    print("B. AUXILIARY FEATURES vs CALIBRATED BASELINE  (Reviewer #1's concern)")
    print("-" * 78)
    for label, tmpl in [("Prev-2", "Previous-Sentences-2{s}"),
                        ("LIWC-22", "Lex-LIWC-22{s}"),
                        ("BERTopic", "TD-BERTopic{s}")]:
        contrast(f"Calibrated baseline  vs  {label}",
                 ("baseline", DIRECT, "Baseline{s}", None),
                 (label,      DIRECT, tmpl,          None))

    print("\n" + "-" * 78)
    print("C. DIRECT vs PRESENCE-GATED, feature-matched (Table 4)")
    print("-" * 78)
    contrast("Direct + MJD  vs  Presence-gated (MJD)",
             ("direct", DIRECT, "Lex-MJD{s}", None),
             ("gated",  GATED,
              "1_Previous-Sentences-2-Lex-LIWC-22{s}_0.1_Lex-MJD{s}", None),
             note="gate = Prev-2 + LIWC-22 at t_gate = 0.10")


if __name__ == "__main__":
    main()
