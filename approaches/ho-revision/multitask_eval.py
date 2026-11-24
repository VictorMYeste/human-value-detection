#!/usr/bin/env python3
"""
Multi-task training with an auxiliary HO objective (step 1.5 of the revision plan;
Reviewer #2, comment 2), evaluated under the main protocol.

The multi-task model (approaches/moral-values-ho-aux, labels 19 values + 8 HO) is
compared with the Direct Baseline trained with the same command, flags and seed
(approaches/moral-values/output/Baseline[-s<seed>]). Both use per-label F1 thresholds
fitted on validation (0.01-0.99) and are evaluated once on test, on all 19 values and
on the values of each HO slice. Differences are tested with the one-sided paired
sentence bootstrap (B = 2000) in both directions, with Benjamini-Hochberg correction
across the nine comparisons of a seed. With several seeds, the per-seed differences
are also summarised (mean, sd, sign consistency).

As a side result, the multi-task model's own HO outputs are scored as HO detectors and
set against the detectors selected in rebuild_tables.py (Table 1).

Usage (after the multi-task predictions are in approaches/moral-values-ho-aux/output/):
    python3 approaches/ho-revision/multitask_eval.py
Writes results/multitask.csv, results/multitask-ho.csv and results/multitask.txt.
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from common import (APPROACHES, PAIRS, RESULTS, SLICES, TABLE4_GATED, Bootstrap, bh, gold,
                    label_f1_thresholds, labels_of, load, macro_f1)

MT_DIR = APPROACHES / "moral-values-ho-aux" / "output"
DIRECT_DIR = APPROACHES / "moral-values" / "output"
SEEDS = {42: "", 7: "-s7", 1701: "-s1701"}


def slice_labels():
    out = {}
    for sl, (folder, _) in SLICES.items():
        out[sl] = labels_of(APPROACHES / f"{folder}_moral-values" / "output" / f"{TABLE4_GATED[sl]}-test.tsv")
    return out


def main() -> None:
    sl_labs = slice_labels()
    values = labels_of(DIRECT_DIR / "Baseline-test.tsv")
    values = [v for v in values if v in gold("test").columns][:19]
    units = {"All 19 values": values, **sl_labs}
    boot = Bootstrap(len(gold("test")))
    rows, ho_rows = [], []
    for seed, suf in SEEDS.items():
        mt_v, mt_t = MT_DIR / f"MT-Baseline{suf}-val.tsv", MT_DIR / f"MT-Baseline{suf}-test.tsv"
        if not (mt_v.exists() and mt_t.exists()):
            continue
        d_v, d_t = DIRECT_DIR / f"Baseline{suf}-val.tsv", DIRECT_DIR / f"Baseline{suf}-test.tsv"
        res = []
        for unit, labs in units.items():
            yv, dv = load(d_v, labs, "val"); yt, dt = load(d_t, labs, "test")
            _, mv = load(mt_v, labs, "val"); _, mt = load(mt_t, labs, "test")
            thd, thm = label_f1_thresholds(yv, dv), label_f1_thresholds(yv, mv)
            t = boot.both_ways(yt, dt >= thd[None, :], mt >= thm[None, :])
            res.append(dict(seed=seed, unit=unit, n_labels=len(labs),
                            direct_val=macro_f1(yv, dv, thd), direct_test=macro_f1(yt, dt, thd),
                            mt_val=macro_f1(yv, mv, thm), mt_test=macro_f1(yt, mt, thm),
                            lower95=t["lower95"], p_mt_better=t["p_2_better"], p_mt_worse=t["p_1_better"]))
        pb, pw = bh([r["p_mt_better"] for r in res]), bh([r["p_mt_worse"] for r in res])
        for r, a, b in zip(res, pb, pw):
            r.update(diff=r["mt_test"] - r["direct_test"], p_mt_better_bh=a, p_mt_worse_bh=b)
        rows += res

        sel = json.loads((RESULTS / "rebuild-selections.json").read_text())
        for pair, (folder, labs) in PAIRS.items():
            yv, mv = load(mt_v, labs, "val"); yt, mt = load(mt_t, labs, "test")
            th = label_f1_thresholds(yv, mv)
            det = sel["ho_detector"][pair]
            _, sv = load(APPROACHES / folder / "output" / f"{det}-val.tsv", labs, "val")
            _, st = load(APPROACHES / folder / "output" / f"{det}-test.tsv", labs, "test")
            ths = label_f1_thresholds(yv, sv)
            ho_rows.append(dict(seed=seed, pair=pair, mt_test=macro_f1(yt, mt, th),
                                detector=det, detector_test=macro_f1(yt, st, ths)))

    if not rows:
        sys.exit(f"No multi-task predictions in {MT_DIR}")
    df, ho = pd.DataFrame(rows), pd.DataFrame(ho_rows)
    df.to_csv(RESULTS / "multitask.csv", index=False)
    ho.to_csv(RESULTS / "multitask-ho.csv", index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pd.set_option("display.width", 200)
        for seed, g in df.groupby("seed", sort=False):
            print(f"=== Seed {seed}: multi-task (19 values + 8 HO) vs Direct Baseline, per-label F1 thresholds")
            print(g[["unit", "n_labels", "direct_val", "mt_val", "direct_test", "mt_test", "diff", "lower95",
                     "p_mt_better_bh", "p_mt_worse_bh"]].round(3).to_string(index=False))
        if df.seed.nunique() > 1:
            print(f"\n=== Across seeds {sorted(df.seed.unique())}: test difference (multi-task - Direct)")
            s = df.groupby("unit", sort=False)["diff"].agg(["mean", "std", "min", "max",
                                                            lambda d: int((d > 0).sum())])
            s.columns = ["mean", "sd", "min", "max", "seeds_positive"]
            print(s.round(4).to_string())
        print("\n=== The multi-task model's HO outputs as HO detectors (test macro-F1 over the pair)")
        print(ho.round(3).to_string(index=False))
    text = buf.getvalue()
    (RESULTS / "multitask.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
