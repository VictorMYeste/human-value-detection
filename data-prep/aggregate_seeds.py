#!/usr/bin/env python3
"""
A1 — Aggregate multi-seed runs into mean ± std (fills the [PROVISIONAL] spots).

Replicates eval-threshold.sh in Python: for each (config, seed) it sweeps a single
GLOBAL threshold on validation (max of the selection metric, ascending tie-break,
exactly as eval-threshold.sh) and reports the metric on val/test at the default 0.5
and at the tuned t*, then mean ± std across seeds.

Seed 42 == the original published outputs (unsuffixed names); seeds 7, 1701 are the
new runs (`-s7`, `-s1701`). Gold labels must be present under data/.

  python3 data-prep/aggregate_seeds.py --task value      # Table 4
  python3 data-prep/aggregate_seeds.py --task presence    # Table 2

VALUE task metric = macro-F1 over the 19 labels (== mean positive-class F1 per label;
reproduces the paper's 0.281/0.311/0.320/0.319 exactly).

PRESENCE is a single binary label, so two different "F1" numbers exist and the
manuscript currently conflates them — we print BOTH:
  * posF1   = F1 of the positive (presence) class only  (what the caption claims)
  * macroF1 = mean of the F1 of the present AND absent classes
The paper's headline presence number (0.74) is posF1 at the default 0.5. The tuned t* uses
the label-wise scheme (max recall s.t. precision >= 0.40 -> t*=0.10 in Table 2), not max-F1.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_fscore_support

ID = ["Text-ID", "Sentence-ID"]
VALUES = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    "Achievement", "Power: dominance", "Power: resources", "Face",
    "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
    "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance",
]
TASKS = {
    "value": dict(out="approaches/moral-values/output", cols=VALUES, grid_step=0.05,
                  configs=["Baseline", "Previous-Sentences-2", "Lex-LIWC-22", "TD-BERTopic"]),
    "presence": dict(out="approaches/presence/output", cols=["Presence"], grid_step=0.01,
                     configs=["Baseline", "Lex-LIWC-22_LingFeat",
                              "Previous-Sentences-2-Lex-LIWC-22",
                              "Previous-Sentences-2-Lex-EmoLex",
                              "Previous-Sentences-2-Lex-eMFD"]),
}


def fname(cfg, seed):
    return cfg if seed == 42 else f"{cfg}-s{seed}"


def load_gold(split, cols):
    g = pd.read_csv(Path("data") / f"{split}-english" / "labels-cat.tsv", sep="\t").set_index(ID)
    if cols == ["Presence"]:
        return pd.DataFrame({"Presence": (g[VALUES].astype(int).sum(axis=1) > 0).astype(int)},
                            index=g.index)
    return g[cols].astype(int)


def scores(gold, prob, th, task):
    """Return the metric dict at threshold th."""
    pred = (prob.loc[gold.index, gold.columns].values >= th).astype(int)
    y = gold.values
    if task == "presence":
        yt, yp = y[:, 0], pred[:, 0]
        return {"posF1": f1_score(yt, yp, average="binary", zero_division=0),
                "macroF1": f1_score(yt, yp, average="macro", zero_division=0)}
    return {"macroF1": f1_score(y, pred, average="macro", zero_division=0)}


# t* selection on validation, matching the code that produced the paper:
#   value    -> single global threshold maximising macro-F1 over 19 labels (eval-threshold.sh)
#   presence -> label-wise scheme = max positive-class recall s.t. precision >= 0.40,
#               searching [0.10, 0.90) (core.evaluation.find_best_threshold; t*=0.10 in Table 2)
SEL = {"value": "macroF1"}


def best_t(gold, prob, grid, task):
    if task == "presence":
        y = gold.values[:, 0]
        p = prob.loc[gold.index, gold.columns].values[:, 0]
        best, best_rec = 0.5, 0.0
        for t in np.round(np.arange(0.10, 0.90, 0.01), 2):
            pr, rc, _, _ = precision_recall_fscore_support(
                y, (p >= t).astype(int), average=None, labels=[0, 1], zero_division=0)
            if pr[1] >= 0.40 and rc[1] > best_rec:
                best_rec, best = rc[1], t
        return best
    vals = [scores(gold, prob, t, task)[SEL[task]] for t in grid]
    return grid[int(np.argmax(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="value")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1701])
    args = ap.parse_args()
    cfg = TASKS[args.task]
    out = Path(cfg["out"])
    grid = np.round(np.arange(cfg["grid_step"], 1.0 + 1e-9, cfg["grid_step"]), 2)
    keys = ["posF1", "macroF1"] if args.task == "presence" else ["macroF1"]

    gv, gt = load_gold("validation", cfg["cols"]), load_gold("test", cfg["cols"])
    sel_desc = "macro-F1" if args.task == "value" else "recall s.t. precision>=0.40 (label-wise)"
    print(f"Task={args.task}  seeds={args.seeds}  grid step={cfg['grid_step']}  "
          f"(t* maximises val {sel_desc})\n")

    for c in cfg["configs"]:
        rows = []
        for s in args.seeds:
            vp, tp = out / f"{fname(c, s)}-val.tsv", out / f"{fname(c, s)}-test.tsv"
            if not (vp.exists() and tp.exists()):
                print(f"  [skip] {c} seed {s}: missing prediction TSV"); continue
            pv = pd.read_csv(vp, sep="\t").set_index(ID)
            pt = pd.read_csv(tp, sep="\t").set_index(ID)
            t = best_t(gv, pv, grid, args.task)
            r = {"seed": s, "t*": t}
            for split, g, p in [("val", gv, pv), ("test", gt, pt)]:
                for half, th in [("0.5", 0.5), ("t*", t)]:
                    for k, v in scores(g, p, th, args.task).items():
                        r[f"{split}.{k}@{half}"] = v
            rows.append(r)

        print(f"### {c}")
        if not rows:
            print("  (no seeds found)\n"); continue
        for r in rows:
            cells = "  ".join(f"{m}@.5/t*={r[f'test.{m}@0.5']:.3f}/{r[f'test.{m}@t*']:.3f}" for m in keys)
            print(f"  seed {r['seed']:>4}: t*={r['t*']:.2f}  test {cells}")
        # mean +/- std of the headline test columns
        for m in keys:
            for half in ["0.5", "t*"]:
                col = f"test.{m}@{half}"
                a = np.array([r[col] for r in rows])
                sd = a.std(ddof=1) if len(a) > 1 else 0.0
                print(f"    test {m} @{half}: {a.mean():.3f} ± {sd:.3f}  (n={len(a)})")
        print()


if __name__ == "__main__":
    main()
