#!/usr/bin/env python3
"""
"Thr. tuning" column of main-text Table 7 (step 2.8 of the revision plan).

For each HO slice, the Direct model selected in rebuild_tables.py is scored on test at the
fixed threshold 0.5 and at its per-label F1 thresholds (fitted on validation). The gain is
tested with the one-sided paired bootstrap (B = 2000) in both directions, with
Benjamini-Hochberg across the eight slices, and per label with the exact McNemar test on
gold positives (BH within the slice).

Usage:
    python3 approaches/ho-revision/threshold_gain_direct.py
Writes results/threshold-gain-direct.csv and results/threshold-gain-direct.txt.
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS, SLICES, Bootstrap, bh, gold, label_f1_thresholds, load, macro_f1, mcnemar_positives
from rebuild_ensembles import DIRECT_DIR, slice_labels


def main() -> None:
    sel = json.loads((RESULTS / "rebuild-selections.json").read_text())
    boot = Bootstrap(len(gold("test")))
    rows = []
    for sl in SLICES:
        labs = slice_labels(sl)
        m = sel["direct"][sl]
        yv, pv = load(DIRECT_DIR / f"{m}-val.tsv", labs, "val")
        yt, pt = load(DIRECT_DIR / f"{m}-test.tsv", labs, "test")
        th = label_f1_thresholds(yv, pv)
        fixed, tuned = pt >= 0.5, pt >= th
        t = boot.both_ways(yt, fixed, tuned)
        mc = mcnemar_positives(yt, fixed, tuned)
        tested = [(l, p) for l, p in zip(labs, mc) if not np.isnan(p)]
        q = bh([p for _, p in tested]) if tested else []
        rows.append(dict(slice=sl, model=m, fixed_test=macro_f1(yt, fixed, 0.5), tuned_test=macro_f1(yt, tuned, 0.5),
                         median_threshold=float(np.median(th)), lower95=t["lower95"], p=t["p_2_better"],
                         p_worse=t["p_1_better"], n_values=len(labs),
                         mcnemar_sig=[l for (l, _), qq in zip(tested, q) if qq < 0.05]))
    df = pd.DataFrame(rows)
    df["diff"] = df.tuned_test - df.fixed_test
    df["p_bh"], df["p_worse_bh"] = bh(df.p.to_numpy()), bh(df.p_worse.to_numpy())
    df.to_csv(RESULTS / "threshold-gain-direct.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    text = ("=== Selected Direct model: per-label F1 thresholds vs fixed 0.5 (test; Table 7 'Thr. tuning')\n"
            + df[["slice", "model", "fixed_test", "tuned_test", "diff", "lower95", "p_bh", "p_worse_bh",
                  "median_threshold"]].round(4).to_string(index=False)
            + "\n\n--- Per-label McNemar on gold positives (BH within slice): significant values\n"
            + "\n".join(f"{r.slice}: {len(r.mcnemar_sig)} of {r.n_values}: {r.mcnemar_sig}" for _, r in df.iterrows())
            + "\n")
    (RESULTS / "threshold-gain-direct.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
