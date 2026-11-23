#!/usr/bin/env python3
"""
Threshold-rule sensitivity (step 1.2 of the revision plan; Reviewer #2, comment 3 c-d).

The reviewer asked (c) why thresholds maximise recall under a precision floor instead of
F1, and (d) whether the gains from tuning depend on the 0.40 floor. Every rule below is
fitted on validation and applied unchanged to test:

  fixed 0.5            the default;
  Eq. 8, floor f       per label, highest recall with precision >= f (f = 0.30, 0.40, 0.50),
                       on the implemented grid 0.10-0.89 with the 0.5 fallback;
  label F1             per label, the threshold on 0.01-0.99 that maximises that label's F1
                       (the revised main rule);
  global F1            one threshold for all labels that maximises macro-F1
                       (grid 0.05-1.00, as in eval-threshold.sh).

Systems:
  HO detection         every detector variant behind Table 1 (four bipolar pairs);
  value detection      per HO slice, the Direct model and the gated configuration of Table 4.

For each system, every rule is compared with fixed 0.5 and with the paper's rule
(Eq. 8, floor 0.40) by a one-sided paired bootstrap on the test sentences (B = 2000),
with Benjamini-Hochberg correction within each family of comparisons.

Usage:
    python3 approaches/ho-revision/threshold_sensitivity.py
Writes results/threshold-sensitivity.csv, results/threshold-comparisons.csv and
results/threshold-sensitivity.txt.
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from common import (APPROACHES, PAIRS, RESULTS, RULES, SLICES, TABLE4_GATED, Bootstrap, bh,
                    fit_rule, gold, label_f1s, labels_of, load, macro_f1)

PAPER_RULE = "Eq. 8, floor 0.40"


def systems():
    """(task, unit, system, labels, val path, test path)."""
    for pair, (folder, labs) in PAIRS.items():
        out = APPROACHES / folder / "output"
        for vf in sorted(out.glob("*-val.tsv")):
            name = vf.name[:-8]
            yield "HO detection", pair, name, labs, vf, out / f"{name}-test.tsv"
    direct = APPROACHES / "moral-values" / "output"
    for sl, (folder, _) in SLICES.items():
        out = APPROACHES / f"{folder}_moral-values" / "output"
        cfg = TABLE4_GATED[sl]
        labs = labels_of(out / f"{cfg}-test.tsv")
        yield "Value detection", sl, "Direct", labs, direct / "Baseline-val.tsv", direct / "Baseline-test.tsv"
        yield "Value detection", sl, f"Gated ({cfg})", labs, out / f"{cfg}-val.tsv", out / f"{cfg}-test.tsv"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    boot = Bootstrap(len(gold("test")))
    rows, comps = [], []
    for task, unit, system, labs, vf, tf in systems():
        yv, sv = load(vf, labs, "val")
        yt, st = load(tf, labs, "test")
        preds = {}
        for rule in RULES:
            th, n_fb = fit_rule(rule, yv, sv)
            preds[rule] = st >= th[None, :]
            rows.append(dict(task=task, unit=unit, system=system, rule=rule,
                             n_labels=len(labs), fallbacks=n_fb,
                             val_f1=macro_f1(yv, sv, th), test_f1=macro_f1(yt, st, th),
                             test_label_f1=json.dumps([round(x, 3) for x in label_f1s(yt, st, th)]),
                             thresholds=json.dumps([round(float(t), 2) for t in th])))
        for base in ["fixed 0.5", PAPER_RULE]:
            for rule in RULES:
                if rule == base:
                    continue
                lo, p = boot.compare(yt, preds[base], preds[rule])
                _, p_worse = boot.compare(yt, preds[rule], preds[base])
                f_base = macro_f1(yt, preds[base].astype(float), 0.5)
                f_rule = macro_f1(yt, preds[rule].astype(float), 0.5)
                comps.append(dict(task=task, unit=unit, system=system, baseline=base, rule=rule,
                                  delta=f_rule - f_base, lower95=lo, p=p, p_worse=p_worse))

    res = pd.DataFrame(rows)
    cmp_ = pd.DataFrame(comps)
    cmp_["p_bh"] = np.nan
    cmp_["p_worse_bh"] = np.nan
    for (_, base), idx in cmp_.groupby(["task", "baseline"]).groups.items():
        cmp_.loc[idx, "p_bh"] = bh(cmp_.loc[idx, "p"].to_numpy())
        cmp_.loc[idx, "p_worse_bh"] = bh(cmp_.loc[idx, "p_worse"].to_numpy())
    res.to_csv(RESULTS / "threshold-sensitivity.csv", index=False)
    cmp_.to_csv(RESULTS / "threshold-comparisons.csv", index=False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(res, cmp_)
    text = buf.getvalue()
    (RESULTS / "threshold-sensitivity.txt").write_text(text)
    sys.stdout.write(text)


def report(res: pd.DataFrame, cmp_: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    short = {"fixed 0.5": "0.5", "Eq. 8, floor 0.30": "f.30", "Eq. 8, floor 0.40": "f.40*",
             "Eq. 8, floor 0.50": "f.50", "label F1": "lblF1", "global F1": "glbF1"}
    for task in ["HO detection", "Value detection"]:
        r = res[res.task == task]
        tab = r.pivot_table(index=["unit", "system"], columns="rule", values="test_f1", sort=False)[RULES]
        tab.columns = [short[c] for c in tab.columns]
        print(f"\n=== {task}: test macro-F1 by rule (thresholds fitted on validation; * = paper's rule)")
        print(tab.round(3).to_string())
        fb = r[r.rule.str.startswith("Eq. 8")].pivot_table(index=["unit", "system"], columns="rule",
                                                           values="fallbacks", sort=False)
        nl = r.groupby(["unit", "system"], sort=False).n_labels.first()
        fb.columns = [short[c] for c in fb.columns]
        fb["labels"] = nl
        print(f"\n--- {task}: labels that fell back to 0.5 (no threshold reached the precision floor)")
        print(fb.to_string())

        c = cmp_[cmp_.task == task]
        for base in ["fixed 0.5", PAPER_RULE]:
            cc = c[c.baseline == base]
            print(f"\n--- {task}: each rule against '{base}' (one-sided paired bootstrap, BH within family)")
            summ = cc.groupby("rule", sort=False).agg(
                systems=("delta", "size"),
                mean_delta=("delta", "mean"),
                higher=("delta", lambda d: int((d > 0).sum())),
                sig_better=("p_bh", lambda p: int((p < 0.05).sum())),
                sig_worse=("p_worse_bh", lambda p: int((p < 0.05).sum())))
            print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
