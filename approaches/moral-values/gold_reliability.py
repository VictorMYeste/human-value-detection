#!/usr/bin/env python3
"""
Gold-standard reliability: does the main system's per-language performance track how well
that language's annotators agreed?

The ValueEval'24 labels are the curated ValuesML annotations. Each language was annotated by
its own team and curated by that team's curators. Inter-annotator agreement was measured
before curation, per language group, as Krippendorff's alpha over text spans with at least
50% word overlap (Scharfbillig et al., ValuesML, Behavior Research Methods 58:276, 2026, doi 10.3758/s13428-026-03092-z,
Table 5; identical in the PsyArXiv preprint 10.31234/osf.io/y2up4_v1).

Per source language this script reports the refined-value alpha, the share of test sentences
with at least one gold value, the main system's macro- and micro-F1 under prevalence-matched
thresholds (CC@prev, three seeds), and its total-count error |predicted / gold - 1|. It then
gives Spearman correlations of each with alpha across the nine languages, with exact
permutation p-values (9! orderings). Nine points: descriptive only.

Usage:
    python3 gold_reliability.py
Writes results/gold-reliability.txt.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.stats import rankdata

from analyze_provenance import VALUES, load_gold
from measurement_analysis import RES, SEEDS, fit_rules, macro_f1, system_probs

SYSTEM = "Feature ensemble"
# Krippendorff's alpha for the 19 refined values, before curation (ValuesML Table 5).
ALPHA_REFINED = {"BG": 0.473, "DE": 0.340, "EL": 0.680, "EN": 0.371, "FR": 0.656,
                 "HE": 0.533, "IT": 0.609, "NL": 0.380, "TR": 0.408}
ALPHA_ALL = 0.486


def micro_f1(y: np.ndarray, pb: np.ndarray) -> float:
    tp = (y & pb).sum(); fp = ((1 - y) & pb).sum(); fn = (y & (1 - pb)).sum()
    return float(2 * tp / (2 * tp + fp + fn))


def spearman_exact(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and its exact two-sided permutation p-value."""
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    perms = np.array(list(permutations(range(len(x)))))
    null = (ry[perms] * rx).sum(1) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean())


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    lang = gt["Text-ID"].str.split("_").str[0].to_numpy()
    langs = sorted(ALPHA_REFINED)

    stats = {l: {"macro": [], "micro": [], "signed_err": []} for l in langs}
    for seed in SEEDS:
        pv, pt = system_probs(SYSTEM, seed, "val"), system_probs(SYSTEM, seed, "test")
        dec = (pt >= fit_rules(pv, yv)["tprev"]).astype(int)
        for l in langs:
            m = lang == l
            stats[l]["macro"].append(macro_f1(yt[m], dec[m]))
            stats[l]["micro"].append(micro_f1(yt[m], dec[m]))
            stats[l]["signed_err"].append((dec[m].sum() / yt[m].sum() - 1) * 100)
    for l in langs:
        stats[l]["total_err"] = [abs(e) for e in stats[l]["signed_err"]]

    rows = {l: {"alpha": ALPHA_REFINED[l], "n": int((lang == l).sum()),
                "valued": yt[lang == l].any(1).mean() * 100,
                "labels": yt[lang == l].sum(1).mean(),
                **{k: float(np.mean(v)) for k, v in stats[l].items()}} for l in langs}

    out("GOLD-STANDARD RELIABILITY BY SOURCE LANGUAGE -- ValueEval'24 English test split")
    out(f"Krippendorff's alpha (19 refined values, before curation; ValuesML Table 5): all languages {ALPHA_ALL}.")
    out(f"Main system: {SYSTEM}, prevalence-matched thresholds (CC@prev), mean of seeds {SEEDS}.")
    out()
    out(f"  {'lang':4s} {'alpha':>6s} {'sentences':>9s} {'valued %':>9s} {'labels/sent':>11s} {'macro-F1':>9s} "
        f"{'micro-F1':>9s} {'count err %':>11s}")
    for l in sorted(langs, key=lambda l: -ALPHA_REFINED[l]):
        r = rows[l]
        out(f"  {l:4s} {r['alpha']:6.3f} {r['n']:9d} {r['valued']:9.1f} {r['labels']:11.3f} {r['macro']:9.3f} "
            f"{r['micro']:9.3f} {r['signed_err']:+11.1f}")
    out(f"  all  {ALPHA_ALL:6.3f} {len(lang):9d} {yt.any(1).mean() * 100:9.1f} {yt.sum(1).mean():11.3f}")
    out("  count err: predicted / gold value mentions - 1 (positive = overcount).")
    out()
    a = np.array([rows[l]["alpha"] for l in langs])
    out("  Spearman correlations across the nine languages (exact permutation p, two-sided):")
    for k, label in [("macro", "macro-F1"), ("micro", "micro-F1"), ("total_err", "absolute count error"),
                     ("valued", "share of sentences with a gold value"), ("labels", "gold labels per sentence")]:
        rho, p = spearman_exact(a, np.array([rows[l][k] for l in langs]))
        out(f"    alpha vs {label:38s} rho = {rho:+.2f}  p = {p:.3f}")
    rho, p = spearman_exact(np.array([rows[l]["labels"] for l in langs]), np.array([rows[l]["signed_err"] for l in langs]))
    out(f"    gold labels per sentence vs signed count error  rho = {rho:+.2f}  p = {p:.3f}")

    RES.mkdir(exist_ok=True)
    (RES / "gold-reliability.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'gold-reliability.txt'}")


if __name__ == "__main__":
    main()
