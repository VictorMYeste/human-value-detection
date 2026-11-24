#!/usr/bin/env python3
"""
Diagnostics of the hard-gated pipelines quoted in main-text Sections 4.5 and 4.6 (steps 1.9
and 8.2 of the revision plan). Only per-label F1 thresholds are fitted here.

1. Tuned thresholds. The per-label F1 thresholds of the selected Direct models and of the
   selected gated value models, pooled over the eight slices (rebuild-selected.csv, part C).
2. Where the gated value models add false positives. On test, behind the gate of the selected
   pipeline, the false positives of the masked Direct model and of the gated value model,
   split by whether the sentence expresses the slice's category (any value of the slice).
3. The mask alone. Mean precision and recall over the values of the slice, masked Direct
   minus Direct (soft-conditioning.csv).

Usage (after rebuild_tables.py and soft_conditioning.py):
    python3 approaches/ho-revision/gate_diagnostics.py
Writes results/gate-diagnostics.txt and results/gate-false-positives.csv (part 2, used for a
table in Supplementary Section S8).
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from common import APPROACHES, RESULTS, SLICES, label_f1_thresholds, labels_of, load
from soft_conditioning import gate_mask


def thresholds() -> None:
    d = pd.read_csv(RESULTS / "rebuild-selected.csv")
    c = d[d.part == "C"]
    for name, col in (("Direct", "direct_thresholds"), ("gated value models", "gated_thresholds")):
        t = np.concatenate([json.loads(x) for x in c[col]])
        q1, q3 = np.percentile(t, [25, 75])
        print(f"{name:20s} n={len(t):3d}  median={np.median(t):.3f}  IQR=[{q1:.3f}, {q3:.3f}]  "
              f"share<0.5={np.mean(t < 0.5):.3f}")


def false_positives() -> None:
    sel = json.loads((RESULTS / "rebuild-selections.json").read_text())
    ddir = APPROACHES / "moral-values" / "output"
    rows = []
    for sl, (folder, parent) in SLICES.items():
        gname = sel["ho_gated"][sl]
        out = APPROACHES / f"{folder}_moral-values" / "output"
        labs = labels_of(out / f"{gname}-test.tsv")
        yv, dv = load(ddir / f"{sel['direct'][sl]}-val.tsv", labs, "val")
        yt, dt = load(ddir / f"{sel['direct'][sl]}-test.tsv", labs, "test")
        _, gv = load(out / f"{gname}-val.tsv", labs, "val")
        _, gt = load(out / f"{gname}-test.tsv", labs, "test")
        m = gate_mask(gname, parent)
        cat = yt.any(1)[:, None]
        admitted = m["test"][:, 0]
        res = {}
        for sys_, sv, st in (("masked Direct", dv * m["val"], dt * m["test"]), ("gated", gv, gt)):
            p = st >= label_f1_thresholds(yv, sv)[None, :]
            res[sys_] = dict(fp_out=int((p & ~cat).sum()), fp_in=int((p & ~yt & cat).sum()), tp=int((p & yt).sum()))
        rows.append(dict(slice=sl, pipeline=gname,
                         n_admitted_out=int((admitted & ~cat[:, 0]).sum()),
                         n_admitted_in=int((admitted & cat[:, 0]).sum()),
                         fp_out_masked=res["masked Direct"]["fp_out"], fp_out_gated=res["gated"]["fp_out"],
                         fp_in_masked=res["masked Direct"]["fp_in"], fp_in_gated=res["gated"]["fp_in"],
                         tp_masked=res["masked Direct"]["tp"], tp_gated=res["gated"]["tp"]))
    df = pd.DataFrame(rows)
    df["extra_fp_out"] = df.fp_out_gated / df.fp_out_masked - 1
    df.to_csv(RESULTS / "gate-false-positives.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"  extra false positives on sentences without the category: "
          f"{df.extra_fp_out.min():.1%} to {df.extra_fp_out.max():.1%}")


def mask_effect() -> None:
    d = pd.read_csv(RESULTS / "soft-conditioning.csv")
    out = {}
    for m in ("test_precision", "test_recall"):
        p = d.pivot_table(index="slice", columns="system", values=m, sort=False)
        out[m] = p["masked Direct"] - p["Direct"]
    df = pd.DataFrame(out)
    print(df.round(4).to_string())
    print(f"  mask lowers mean recall in {int((df.test_recall < 0).sum())}/8 slices, "
          f"raises mean precision in {int((df.test_precision > 0).sum())}/8")


def main() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pd.set_option("display.width", 200)
        print("=== 1. Tuned per-label F1 thresholds, pooled over the eight slices")
        thresholds()
        print("\n=== 2. False positives on test behind the gate of the selected pipeline")
        print("    fp_out: on sentences without the slice's category; fp_in: on sentences with it")
        false_positives()
        print("\n=== 3. The mask alone: masked Direct minus Direct (mean over the values of the slice)")
        mask_effect()
    text = buf.getvalue()
    (RESULTS / "gate-diagnostics.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
