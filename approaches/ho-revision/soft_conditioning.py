#!/usr/bin/env python3
"""
Soft use of the HO structure (step 1.3 of the revision plan; Reviewer #2, comment 2).

The reviewer noted that only hard gates and cascades were tested, and named soft
conditioning on HO probabilities and probabilistic hierarchy priors as missing. Three
soft variants are built on cached probabilities, without retraining. Each slice uses the
Direct model and the HO detectors selected on validation by rebuild_tables.py:

  HO prior      s_v = p_v * p_c^alpha, where c is the slice's HO category and alpha in
                {0.25, 0.5, 1, 2} is chosen on validation. It is the probabilistic
                counterpart of the hard mask, which multiplies by 1[p_c >= tau].
  parent stack  per value, a logistic regression on [logit p_v, logit p_c].
  HO stack      per value, a logistic regression on [logit p_v, logits of all eight HO
                probabilities]: HO probabilities used as features, a post-hoc analogue of
                concatenating them to the value head.

The stackers are fitted on validation. Their validation scores, used to choose
thresholds, are out-of-fold (5 folds grouped by text). The final models are refitted on
all of validation and applied once to test. Thresholds use the main rule (per-label F1,
0.01-0.99). Each variant is compared with the selected Direct model and with the
selected hard-gated pipeline: one-sided paired bootstrap (B = 2000) in both directions,
with Benjamini-Hochberg correction across the 8 slices x 3 variants of each family.

Usage (after rebuild_tables.py):
    python3 approaches/ho-revision/soft_conditioning.py
Writes results/soft-conditioning.csv and results/soft-conditioning.txt.
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from common import (APPROACHES, ID, PAIRS, RESULTS, SLICES, Bootstrap, bh, gold,
                    label_f1_thresholds, labels_of, load, macro_f1)

ALPHAS = [0.25, 0.5, 1.0, 2.0]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ho_probs(selections):
    """All eight HO probabilities from the detectors selected on validation."""
    out = {"val": {}, "test": {}}
    for pair, (folder, labs) in PAIRS.items():
        name = selections["ho_detector"][pair]
        for split in out:
            _, s = load(APPROACHES / folder / "output" / f"{name}-{split}.tsv", labs, split)
            for i, lab in enumerate(labs):
                out[split][lab] = s[:, i]
    return out


def stack(Xv, yv, Xt, groups):
    """Out-of-fold validation scores and test scores of a logistic stacker."""
    oof = np.zeros(len(yv))
    for tr, te in GroupKFold(n_splits=5).split(Xv, yv, groups):
        oof[te] = LogisticRegression(max_iter=2000).fit(Xv[tr], yv[tr]).predict_proba(Xv[te])[:, 1]
    test = LogisticRegression(max_iter=2000).fit(Xv, yv).predict_proba(Xt)[:, 1]
    return oof, test


def scored(yv, sv, yt, st):
    th = label_f1_thresholds(yv, sv)
    return dict(val_f1=macro_f1(yv, sv, th), test_f1=macro_f1(yt, st, th),
                pred=st >= th[None, :], thresholds=th)


def main() -> None:
    sel = json.loads((RESULTS / "rebuild-selections.json").read_text())
    ho = ho_probs(sel)
    ho_labels = list(ho["val"])
    groups = gold("val")["Text-ID"].to_numpy()
    boot = Bootstrap(len(gold("test")))
    direct_dir = APPROACHES / "moral-values" / "output"
    rows = []
    for sl, (folder, parent) in SLICES.items():
        out = APPROACHES / f"{folder}_moral-values" / "output"
        gname, dname = sel["ho_gated"][sl], sel["direct"][sl]
        labs = labels_of(out / f"{gname}-test.tsv")
        yv, dv = load(direct_dir / f"{dname}-val.tsv", labs, "val")
        yt, dt = load(direct_dir / f"{dname}-test.tsv", labs, "test")
        _, gv = load(out / f"{gname}-val.tsv", labs, "val")
        _, gt = load(out / f"{gname}-test.tsv", labs, "test")
        systems = {"Direct": scored(yv, dv, yt, dt), "hard gate": scored(yv, gv, yt, gt)}

        # HO prior: alpha chosen on validation
        pc_v, pc_t = ho["val"][parent][:, None], ho["test"][parent][:, None]
        best = max((scored(yv, dv * pc_v ** a, yt, dt * pc_t ** a) | {"alpha": a} for a in ALPHAS),
                   key=lambda r: r["val_f1"])
        systems["HO prior"] = best

        # Stackers
        Lv_all = np.column_stack([logit(ho["val"][l]) for l in ho_labels])
        Lt_all = np.column_stack([logit(ho["test"][l]) for l in ho_labels])
        for variant, cols in [("parent stack", [ho_labels.index(parent)]),
                              ("HO stack", list(range(len(ho_labels))))]:
            sv_, st_ = np.zeros_like(dv), np.zeros_like(dt)
            for i in range(len(labs)):
                Xv = np.column_stack([logit(dv[:, i]), Lv_all[:, cols]])
                Xt = np.column_stack([logit(dt[:, i]), Lt_all[:, cols]])
                sv_[:, i], st_[:, i] = stack(Xv, yv[:, i].astype(int), Xt, groups)
            systems[variant] = scored(yv, sv_, yt, st_)

        for name, r in systems.items():
            row = dict(slice=sl, system=name, val_f1=r["val_f1"], test_f1=r["test_f1"],
                       alpha=r.get("alpha"), direct_model=dname, gated_pipeline=gname)
            if name not in ("Direct", "hard gate"):
                for ref in ("Direct", "hard gate"):
                    t = boot.both_ways(yt, systems[ref]["pred"], r["pred"])
                    row[f"diff_vs_{ref}"] = r["test_f1"] - systems[ref]["test_f1"]
                    row[f"lower95_vs_{ref}"] = t["lower95"]
                    row[f"p_better_than_{ref}"] = t["p_2_better"]
                    row[f"p_worse_than_{ref}"] = t["p_1_better"]
            rows.append(row)

    df = pd.DataFrame(rows)
    soft = df.system.isin(["HO prior", "parent stack", "HO stack"])
    for ref in ("Direct", "hard gate"):
        for kind in ("better", "worse"):
            col = f"p_{kind}_than_{ref}"
            df.loc[soft, col + "_bh"] = bh(df.loc[soft, col].to_numpy())
    df.to_csv(RESULTS / "soft-conditioning.csv", index=False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(df)
    text = buf.getvalue()
    (RESULTS / "soft-conditioning.txt").write_text(text)
    sys.stdout.write(text)


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    order = ["Direct", "hard gate", "HO prior", "parent stack", "HO stack"]
    tab = df.pivot_table(index="slice", columns="system", values="test_f1", sort=False)[order]
    tab["alpha"] = df[df.system == "HO prior"].set_index("slice")["alpha"]
    print("=== Test macro-F1 (per-label F1 thresholds fitted on validation)")
    print(tab.round(3).to_string())
    print("\n=== Validation macro-F1 (stackers: out-of-fold)")
    print(df.pivot_table(index="slice", columns="system", values="val_f1", sort=False)[order].round(3).to_string())
    soft = df[df.system.isin(order[2:])]
    for ref in ("Direct", "hard gate"):
        print(f"\n=== Soft variants against {ref} (diff on test; BH over 24 tests)")
        t = soft.pivot_table(index="slice", columns="system", values=f"diff_vs_{ref}", sort=False)[order[2:]]
        print(t.round(3).to_string())
        for v in order[2:]:
            s = soft[soft.system == v]
            print(f"  {v:12s}: higher in {(s[f'diff_vs_{ref}'] > 0).sum()}/8, significantly better in "
                  f"{(s[f'p_better_than_{ref}_bh'] < 0.05).sum()}, significantly worse in "
                  f"{(s[f'p_worse_than_{ref}_bh'] < 0.05).sum()}; mean diff {s[f'diff_vs_{ref}'].mean():+.4f}")


if __name__ == "__main__":
    main()
