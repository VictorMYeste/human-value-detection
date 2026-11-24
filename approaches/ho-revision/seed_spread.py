#!/usr/bin/env python3
"""
Seed spread of the Direct models (step 2.7 of the revision plan, limitations).

The paper reports one training run per system (seed 42). For the five Direct 19-value
models, Yeste2026 also saved runs with seeds 7 and 1701. Each run is scored on the values
of every HO slice with its own per-label F1 thresholds (fitted on validation), and the
spread of test Macro-F1 across the three seeds is compared with the effect sizes in the
paper.

Usage:
    python3 approaches/ho-revision/seed_spread.py
Writes results/seed-spread.csv and results/seed-spread.txt.
"""
import numpy as np
import pandas as pd

from common import RESULTS, SLICES, label_f1_thresholds, load, macro_f1
from rebuild_ensembles import DIRECT_DIR, DIRECT_MODELS, slice_labels

SEEDS = {"42": "", "7": "-s7", "1701": "-s1701"}


def main() -> None:
    rows = []
    for sl in SLICES:
        labs = slice_labels(sl)
        for m in DIRECT_MODELS:
            scores = {}
            for seed, suf in SEEDS.items():
                yv, pv = load(DIRECT_DIR / f"{m}{suf}-val.tsv", labs, "val")
                yt, pt = load(DIRECT_DIR / f"{m}{suf}-test.tsv", labs, "test")
                scores[seed] = macro_f1(yt, pt >= label_f1_thresholds(yv, pv), 0.5)
            v = np.array(list(scores.values()))
            rows.append(dict(slice=sl, model=m, **{f"s{k}": x for k, x in scores.items()},
                             sd=v.std(ddof=1), range=v.max() - v.min()))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seed-spread.csv", index=False)
    by_slice = df.groupby("slice", sort=False).agg(median_sd=("sd", "median"), max_range=("range", "max"))
    text = ("=== Test Macro-F1 of the five Direct models over three seeds (per-label F1 thresholds)\n"
            + df.round(4).to_string(index=False)
            + "\n\n--- By slice\n" + by_slice.round(4).to_string()
            + f"\n\nOverall: median SD {df.sd.median():.4f}, mean SD {df.sd.mean():.4f}, "
              f"median range {df.range.median():.4f}, max range {df.range.max():.4f}\n")
    (RESULTS / "seed-spread.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
