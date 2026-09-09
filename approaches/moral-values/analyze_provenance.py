#!/usr/bin/env python3
"""
Provenance-split analysis for the IP&M resubmission.

Reviewer #4 (W9, SC9), Reviewer #2, and the Associate Editor's decision summary
all ask about the effect of machine translation: only ~14% of the English
ValueEval'24 release is English-original, and the lexical/affective features are
word-frequency based, so they are the signals most exposed to translation
artefacts.

Rather than auditing translation quality in the abstract, this script measures
whether translation affects the *conclusions*, which is the question that
actually bears on the paper:

  1. Does macro-F1 differ between English-original and machine-translated
     sentences?
  2. Do the lexicon features help less on translated text than on
     English-original text?  (Reviewer #2's specific hypothesis.)

Source language is recoverable without any new data: ValueEval'24 Text-IDs carry
an ISO language prefix (EN_002, BG_001, ...), as already noted in Section 3.2 of
the manuscript.

The metric reproduces core/evaluation.py exactly: the macro average of
POSITIVE-CLASS F1 over the 19 value labels, binarised at a single global
threshold.  Thresholds are tuned on validation and applied unchanged to test,
per seed, exactly as in the published protocol.

Usage:
    python3 analyze_provenance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
DATA = HERE / ".." / ".." / "data"
OUT = HERE / "output"

VALUES = [
    "Self-direction: thought", "Self-direction: action", "Stimulation",
    "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face",
    "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
]

# The paper's global-threshold grid (eval-threshold.sh).
GRID = [round(0.05 * i, 2) for i in range(1, 21)]

KEY = ["Text-ID", "Sentence-ID"]


def load_gold(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{split}-english" / "labels-cat.tsv", sep="\t")
    return df[KEY + VALUES]


def load_pred(name: str, split_tag: str) -> pd.DataFrame:
    path = OUT / f"{name}-{split_tag}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")[KEY + VALUES]


def aligned(name: str, split: str, split_tag: str):
    """Merge predictions with gold on (Text-ID, Sentence-ID), as core/evaluation.py does."""
    gold = load_gold(split)
    pred = load_pred(name, split_tag)
    merged = gold.merge(pred, on=KEY, suffixes=("_gold", "_pred"))
    if len(merged) != len(gold):
        print(f"  ! {name}/{split}: merged {len(merged)} of {len(gold)} gold rows",
              file=sys.stderr)
    y = merged[[f"{v}_gold" for v in VALUES]].to_numpy() >= 0.5
    p = merged[[f"{v}_pred" for v in VALUES]].to_numpy(dtype=float)
    lang = merged["Text-ID"].str.split("_").str[0].to_numpy()
    return y.astype(int), p, lang


def macro_f1(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Macro average of positive-class F1 over the 19 labels — core/evaluation.py:103-107."""
    pb = (p >= t).astype(int)
    per_label = [
        precision_recall_fscore_support(
            y[:, j], pb[:, j], average=None, labels=[0, 1], zero_division=0
        )[2][1]
        for j in range(y.shape[1])
    ]
    return float(np.mean(per_label))


def tune(name: str) -> float:
    """Sweep the global threshold on validation, return the argmax."""
    y, p, _ = aligned(name, "validation", "val")
    scores = {t: macro_f1(y, p, t) for t in GRID}
    return max(scores, key=scores.get)


def evaluate(name: str):
    """Tune on validation, apply to test, and stratify by source language."""
    t_star = tune(name)
    y, p, lang = aligned(name, "test", "test")

    is_en = lang == "EN"
    res = {
        "t_star": t_star,
        "overall": macro_f1(y, p, t_star),
        "en_original": macro_f1(y[is_en], p[is_en], t_star),
        "translated": macro_f1(y[~is_en], p[~is_en], t_star),
        "n_en": int(is_en.sum()),
        "n_tr": int((~is_en).sum()),
        "per_lang": {
            L: (macro_f1(y[lang == L], p[lang == L], t_star), int((lang == L).sum()))
            for L in sorted(set(lang))
        },
    }
    return res


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def main() -> None:
    # Seed 42 carries no suffix; 7 and 1701 do.
    configs = {
        "Baseline": ["Baseline", "Baseline-s7", "Baseline-s1701"],
        "Lex-LIWC-22": ["Lex-LIWC-22", "Lex-LIWC-22-s7", "Lex-LIWC-22-s1701"],
    }

    results: dict[str, list] = {}
    for label, names in configs.items():
        print(f"\n=== {label} ===")
        runs = []
        for n in names:
            try:
                r = evaluate(n)
            except FileNotFoundError as e:
                print(f"  SKIP {n}: missing {e}")
                continue
            runs.append(r)
            print(f"  {n:24s} t*={r['t_star']:.2f}  "
                  f"overall={r['overall']:.4f}  "
                  f"EN-orig={r['en_original']:.4f}  "
                  f"translated={r['translated']:.4f}  "
                  f"delta={r['en_original'] - r['translated']:+.4f}")
        results[label] = runs

    # ---- aggregate over seeds -------------------------------------------------
    print("\n" + "=" * 72)
    print("MACRO-F1 BY SOURCE PROVENANCE  (mean +/- std over seeds 42, 7, 1701)")
    print("=" * 72)
    agg = {}
    for label, runs in results.items():
        if not runs:
            continue
        o = mean_std([r["overall"] for r in runs])
        e = mean_std([r["en_original"] for r in runs])
        t = mean_std([r["translated"] for r in runs])
        agg[label] = {"overall": o, "en": e, "tr": t}
        print(f"\n{label}")
        print(f"  overall           {o[0]:.4f} +/- {o[1]:.4f}")
        print(f"  English-original  {e[0]:.4f} +/- {e[1]:.4f}   (n={runs[0]['n_en']})")
        print(f"  machine-translated{t[0]:.4f} +/- {t[1]:.4f}   (n={runs[0]['n_tr']})")
        print(f"  EN-orig - translated: {e[0] - t[0]:+.4f}")

    # ---- Reviewer #2's hypothesis --------------------------------------------
    if "Baseline" in agg and "Lex-LIWC-22" in agg:
        print("\n" + "=" * 72)
        print("DO LEXICON FEATURES HELP LESS ON TRANSLATED TEXT?")
        print("(Reviewer #2's hypothesis: word-frequency features are the exposed signal)")
        print("=" * 72)
        d_en = agg["Lex-LIWC-22"]["en"][0] - agg["Baseline"]["en"][0]
        d_tr = agg["Lex-LIWC-22"]["tr"][0] - agg["Baseline"]["tr"][0]
        print(f"  LIWC-22 gain on English-original : {d_en:+.4f}")
        print(f"  LIWC-22 gain on translated       : {d_tr:+.4f}")
        print(f"  difference in gain               : {d_en - d_tr:+.4f}")

    # ---- per language ---------------------------------------------------------
    if results.get("Baseline"):
        print("\n" + "=" * 72)
        print("MACRO-F1 BY SOURCE LANGUAGE  (Baseline, mean over seeds)")
        print("=" * 72)
        langs = sorted(results["Baseline"][0]["per_lang"])
        for L in langs:
            vals = [r["per_lang"][L][0] for r in results["Baseline"]]
            n = results["Baseline"][0]["per_lang"][L][1]
            m, s = mean_std(vals)
            flag = "  <- English-original" if L == "EN" else ""
            print(f"  {L}  n={n:5d}   {m:.4f} +/- {s:.4f}{flag}")


if __name__ == "__main__":
    main()
