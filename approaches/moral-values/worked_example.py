#!/usr/bin/env python3
"""
Worked example: which values do political manifestos emphasise relative to news?

A social scientist with a value classifier would answer this by estimating, for each of the
19 values, the share of manifesto sentences expressing it minus the share of news sentences,
with a confidence interval, and reading off the significant differences. This script gives
that answer with the gold labels and with each estimator, so the consequences of the
decision rule can be read as substantive conclusions:

  CC@0.5     default threshold
  CC@t*      macro-F1-optimal global threshold (shared-task practice)
  CC@prev    prevalence-matched per-label thresholds (the recommended rule)
  PCC-Platt  average of Platt-calibrated probabilities
  ACC@prev   adjusted count with pooled validation error rates (standard quantification)
  SC@prev    CC@prev contrast divided by the attenuation slope of the same contrast on validation
  LLM        Gemma-2-9B few-shot + presence gate labels, counted as they are (single run)

Two versions of the question:
  crude              all manifesto sentences vs all news sentences;
  language-adjusted  manifestos compared with news from the same source language, then averaged
                     with weights proportional to each language's manifesto sentences. Each
                     language was annotated by its own team, and annotation rates differ
                     threefold between teams (gold_reliability.py), so this version removes
                     differences in language composition between the genres.

Main system: the feature ensemble (three seeds). Its estimate is the mean of the three runs,
and every count and test is on that mean. Significance: stratified document bootstrap
(B = 2000, shared with measurement_analysis.py), normal-approximation p-values, BH across the
19 values; a value with no reference mentions in one genre is tested with Fisher's exact
test. Everything fitted on validation (thresholds, Platt maps, error rates, slopes) is refitted
on each validation resample.

Usage:
    python3 worked_example.py
Writes results/worked-example.txt, worked-example-values.csv and worked-example-summary.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analyze_provenance import VALUES, load_gold
from llm_measurement import LLM_SYSTEMS, load_labels
from measurement_analysis import (ALPHA, B, RES, RNG_SEED, SEEDS, RowFits, adjust, bh, boot_p, ci,
                                  document_design, genre_reference_p, slope, system_probs, wsum)
from shift_tests import Groups, safe_slope

SYSTEM = "Feature ensemble"
LLM = "Gemma-2-9B few-shot + gate"
ESTIMATORS = ["CC@0.5", "CC@t*", "CC@prev", "PCC-Platt", "ACC@prev", "SC@prev", "LLM"]
VERSIONS = ["crude", "language-adjusted"]
SHORT = {v: v.replace("Self-direction", "SD").replace("Security", "Sec.").replace("Conformity", "Conf.")
         .replace("Benevolence", "Benev.").replace("Universalism", "Univ.").replace("Power", "Pow.") for v in VALUES}


class GenreContrast:
    """Manifesto-news contrasts of one split, crude and adjusted for source language."""

    def __init__(self, G: Groups):
        self.G = G
        self.langs = sorted(set(G.lang))
        self.man, self.news = G.genre == "manifesto", G.genre == "news"
        man_sent = np.array([G.nsent[self.man & (G.lang == l)].sum() for l in self.langs], float)
        self.w = man_sent / man_sent.sum()

    def parts(self, cdoc: np.ndarray) -> dict[str, np.ndarray]:
        G = self.G
        return {"man": G.prev(cdoc, self.man), "news": G.prev(cdoc, self.news),
                "man_l": np.stack([G.prev(cdoc, self.man & (G.lang == l)) for l in self.langs], axis=1),
                "news_l": np.stack([G.prev(cdoc, self.news & (G.lang == l)) for l in self.langs], axis=1)}

    def contrasts(self, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"crude": p["man"] - p["news"],
                "language-adjusted": (self.w[None, :, None] * (p["man_l"] - p["news_l"])).sum(1)}

    def of(self, cdoc: np.ndarray) -> dict[str, np.ndarray]:
        return self.contrasts(self.parts(cdoc))


def summary(e: np.ndarray, g: np.ndarray, gsig: np.ndarray) -> dict:
    esig = bh(boot_p(e)) < ALPHA
    same = np.sign(e[0]) == np.sign(g[0])
    top = lambda x, k=3: set(np.argsort(-x)[:k])
    bottom = lambda x, k=3: set(np.argsort(x)[:k])
    return {"found": int((gsig & esig & same).sum()), "missed": int((gsig & ~esig).sum()),
            "reversed": int((gsig & esig & ~same).sum()), "spurious": int((~gsig & esig).sum()),
            "slope": slope(g, e), "rho": float(np.corrcoef(rankdata(e[0]), rankdata(g[0]))[0, 1]),
            "mae": float(np.abs(e[0] - g[0]).mean() * 100),
            "top3": len(top(e[0]) & top(g[0])), "bottom3": len(bottom(e[0]) & bottom(g[0]))}


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    T = GenreContrast(Groups(document_design(gt)))
    Vdesign = document_design(gv, seed=RNG_SEED + 1)
    V = GenreContrast(Groups(Vdesign))

    gold = T.of(T.G.S @ yt)
    gold_v = V.of(V.G.S @ yv)
    # a value with no reference mentions in one genre is tested with Fisher's exact test on the pooled
    # sentence counts, in both versions (the bootstrap cannot vary an empty group)
    gsig = {k: bh(genre_reference_p(gold[k], T.G.S @ yt, T.G.nsent, T.man, T.news)) < ALPHA for k in VERSIONS}

    runs: dict[str, dict[str, list[np.ndarray]]] = {k: {e: [] for e in ESTIMATORS} for k in VERSIONS}
    betas: dict[str, list[float]] = {k: [] for k in VERSIONS}
    for seed in SEEDS:
        pv, pt = system_probs(SYSTEM, seed, "val"), system_probs(SYSTEM, seed, "test")
        rf = RowFits(pv, yv, Vdesign)                     # every rule refitted on each validation row
        counts = {"CC@0.5": T.G.S @ (pt >= 0.5).astype(float),
                  **{r: rf.counts(r, pt, T.G.S) for r in ["CC@t*", "CC@prev", "PCC-Platt"]}}
        for r in ["CC@0.5", "CC@t*", "CC@prev", "PCC-Platt"]:
            for k, c in T.of(counts[r]).items():
                runs[k][r].append(c)
        tp_v, fp_v = rf.counts("CC@prev", pv, V.G.S, weight=yv), rf.counts("CC@prev", pv, V.G.S, weight=1 - yv)
        every = np.ones(V.G.S.shape[0], bool)
        tpr = wsum(V.G.W, tp_v, every) / wsum(V.G.W, V.G.S @ yv, every)
        fpr = wsum(V.G.W, fp_v, every) / wsum(V.G.W, V.G.S @ (1 - yv), every)
        p = T.parts(counts["CC@prev"])
        acc = {"man": adjust(p["man"], tpr, fpr), "news": adjust(p["news"], tpr, fpr),
               "man_l": adjust(p["man_l"], tpr[:, None], fpr[:, None]),
               "news_l": adjust(p["news_l"], tpr[:, None], fpr[:, None])}
        est_v = V.of(tp_v + fp_v)
        for k, c in T.contrasts(acc).items():
            runs[k]["ACC@prev"].append(c)
        for k in VERSIONS:
            beta = safe_slope(slope(gold_v[k], est_v[k]))
            betas[k].append(float(beta[0]))
            runs[k]["SC@prev"].append(runs[k]["CC@prev"][-1] / beta[:, None])
    for k, c in T.of(T.G.S @ load_labels(LLM_SYSTEMS[LLM][0], "test")).items():
        runs[k]["LLM"].append(c)

    avg = {k: {e: np.mean(runs[k][e], axis=0) for e in ESTIMATORS} for k in VERSIONS}

    out("WORKED EXAMPLE -- Which values do manifestos emphasise relative to news? (ValueEval'24 English test split)")
    out(f"Main system: {SYSTEM}, seeds {SEEDS}; per-value estimates averaged over runs. LLM: {LLM} (single run).")
    out(f"Difference = share of manifesto sentences - share of news sentences, percentage points. * = BH q < {ALPHA} "
        f"(bootstrap, B = {B}); ! = significant with the opposite sign to a significant gold difference.")
    out(f"Language-adjusted: within-language differences weighted by manifesto sentences "
        f"({', '.join(f'{l} {w:.2f}' for l, w in zip(T.langs, T.w))}).")
    out(f"SC validation slopes (mean over runs): crude {np.mean(betas['crude']):.2f}, "
        f"language-adjusted {np.mean(betas['language-adjusted']):.2f}.")

    rows, summ = [], []
    for k in VERSIONS:
        g = gold[k]
        out()
        out("=" * 124)
        out(f"{k.upper()}: {gsig[k].sum()} of 19 gold differences significant")
        out("=" * 124)
        out(f"  {'value':24s} {'gold [95% CI]':>22s} " + " ".join(f"{e:>9s}" for e in ESTIMATORS))
        sig = {e: bh(boot_p(avg[k][e])) < ALPHA for e in ESTIMATORS}
        for i in np.argsort(-g[0]):
            lo, hi = np.percentile(g[1:, i], [2.5, 97.5]) * 100
            gtxt = f"{g[0, i] * 100:+.1f}{'*' if gsig[k][i] else ' '} [{lo:+.1f}, {hi:+.1f}]"
            cells = []
            for e in ESTIMATORS:
                v = avg[k][e][0, i] * 100
                mark = ("!" if gsig[k][i] and np.sign(v) != np.sign(g[0, i]) else "*") if sig[e][i] else " "
                cells.append(f"{v:+8.1f}{mark}")
                rows.append({"version": k, "value": VALUES[i], "estimator": e, "diff_pp": v, "significant": bool(sig[e][i])})
            rows.append({"version": k, "value": VALUES[i], "estimator": "gold", "diff_pp": g[0, i] * 100,
                         "lo_pp": lo, "hi_pp": hi, "significant": bool(gsig[k][i])})
            out(f"  {SHORT[VALUES[i]]:24s} {gtxt:>22s} " + " ".join(cells))

        out()
        out(f"  Conclusions per estimator (on the mean of the main system's {len(SEEDS)} runs): of the {gsig[k].sum()} "
            "significant gold differences, how many are found with the right sign, missed or reversed; spurious = significant")
        out("  where gold is not; top/bottom 3 = overlap with gold's three most manifesto- / news-leaning values.")
        out(f"  {'estimator':10s} {'found':>6s} {'missed':>7s} {'reversed':>9s} {'spurious':>9s} {'slope [95% CI]':>20s} "
            f"{'rank rho':>9s} {'MAE pp':>7s} {'top 3':>6s} {'bottom 3':>9s}")
        for e in ESTIMATORS:
            m = summary(avg[k][e], g, gsig[k])
            ms = runs[k][e]
            out(f"  {e:10s} {m['found']:6d} {m['missed']:7d} {m['reversed']:9d} {m['spurious']:9d} "
                f"{ci(m['slope']):>20s} {m['rho']:9.2f} {m['mae']:7.2f} {m['top3']:6d} {m['bottom3']:9d}")
            summ.append({"version": k, "estimator": e, "runs": len(ms), **{key: float(np.ravel(m[key])[0]) for key in m}})

        out()
        for label, sign in [("more in manifestos", 1), ("more in news", -1)]:
            out(f"  Significantly {label}:")
            for name, arr, s in [("gold", g, gsig[k])] + [(e, avg[k][e], sig[e]) for e in ESTIMATORS]:
                idx = [i for i in np.argsort(-sign * arr[0]) if s[i] and np.sign(arr[0, i]) == sign]
                out(f"    {name:10s} ({len(idx):2d}) " + ", ".join(SHORT[VALUES[i]] for i in idx))

    ga, gc = gold["language-adjusted"][0], gold["crude"][0]
    out()
    out(f"Crude vs language-adjusted gold: rank rho {np.corrcoef(rankdata(gc), rankdata(ga))[0, 1]:.2f}; "
        f"significant in crude only: {', '.join(SHORT[VALUES[i]] for i in np.where(gsig['crude'] & ~gsig['language-adjusted'])[0]) or 'none'}; "
        f"in adjusted only: {', '.join(SHORT[VALUES[i]] for i in np.where(~gsig['crude'] & gsig['language-adjusted'])[0]) or 'none'}.")

    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "worked-example-values.csv", index=False, float_format="%.4f")
    pd.DataFrame(summ).to_csv(RES / "worked-example-summary.csv", index=False, float_format="%.4f")
    (RES / "worked-example.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'worked-example.txt'}, worked-example-values.csv and worked-example-summary.csv")


if __name__ == "__main__":
    main()
