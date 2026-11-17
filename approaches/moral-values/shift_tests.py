#!/usr/bin/env python3
"""
Shift tests: does the recommended measurement procedure hold for groups that were not in the fit?

measurement_analysis.py fits every rule on the full validation split, which covers the same
nine source languages and both genres as the test split. In practice an analyst may have no
annotated data for some of the groups being compared. Three scenarios test that, always on
the ValueEval'24 English test split:

  A. Leave one language out. For each source language L, every rule (thresholds, Platt maps,
     the regression-calibration slope) is fitted on the validation documents of the other
     eight languages. We then estimate L's value prevalence and L's contrasts with the eight
     other languages (8 x 19 = 152 contrasts per language; 1,368 pooled over the nine).
     Reference: the same quantities with everything fitted on the full validation split.
  B. Genre transfer. Every rule is fitted on news validation documents only, and we estimate
     manifesto prevalence and the manifesto-news contrast. With no manifesto validation data
     the genre slope cannot be estimated, so RC uses the language-pair slope measured within
     news validation as a proxy: a stress test of borrowing a slope across kinds of contrast.
     Reference: the full fit, with RC using the genre slope measured on validation.
  C. Language contrasts within news only (full fit): is the attenuation of language contrasts
     an artefact of languages differing in their share of manifestos? RC uses the language
     slope measured within news validation.

Rules: CC@t* (macro-F1 threshold, the shared-task practice), CC@prev (prevalence-matched
per-label thresholds), PCC-Platt (calibrated probability average), and RC@prev (CC@prev
with group deviations from the corpus total divided by the validation attenuation slope).

Metrics, as in measurement_analysis.py: contrast slope (OLS of estimated on gold contrasts;
1 = faithful), share of contrasts with differential bias (paired bootstrap, BH q < 0.05),
MAE in percentage points; for prevalence, the error in a group's total value mentions,
MAE over values, and the number of values significantly mis-estimated. Uncertainty: the
same stratified document bootstraps of test and validation as measurement_analysis.py;
metrics are computed per run and averaged over a system's runs.

Usage:
    python3 shift_tests.py
Writes results/shift-tests.txt and results/shift-tests.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_provenance import VALUES, load_gold
from measurement_analysis import (ALPHA, B, MIN_SLOPE, RES, RNG_SEED, SYSTEMS, apply_rules, bh, boot_p, ci,
                                  document_design, fit_rules, slope, system_probs, system_seeds)

RULES = ["CC@t*", "CC@prev", "PCC-Platt", "RC@prev"]
BASE = {"CC@t*": "CC@t*", "CC@prev": "CC@prev", "PCC-Platt": "PCC-Platt", "RC@prev": "CC@prev"}


class Groups:
    """Bootstrapped prevalence of any set of documents of one split (row 0 = original sample)."""

    def __init__(self, design: dict):
        self.S, self.W, self.nsent = design["S"], design["W"], design["nsent"]
        self.lang, self.genre = design["lang"], design["genre"]
        self.all = np.ones(len(self.lang), bool)

    def prev(self, cdoc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return (self.W[:, mask] @ cdoc[mask]) / (self.W[:, mask] @ self.nsent[mask])[:, None]

    def langs(self, cdoc: np.ndarray, langs: list[str], doc_mask: np.ndarray | None = None) -> np.ndarray:
        m = self.all if doc_mask is None else doc_mask
        return np.stack([self.prev(cdoc, (self.lang == l) & m) for l in langs], axis=1)


def pair_contrasts(lp: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([lp[:, i] - lp[:, j] for i, j in pairs], axis=1).reshape(lp.shape[0], -1)


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def safe_slope(beta: np.ndarray) -> np.ndarray:
    """Validation slopes at or below MIN_SLOPE (or undefined) mean no correction (slope 1)."""
    return np.where(np.isfinite(beta) & (beta > MIN_SLOPE), beta, 1.0)


def rc(group: np.ndarray, total: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Regression calibration: divide a group's deviation from the corpus total by the slope."""
    t = total.reshape(total.shape[0], *([1] * (group.ndim - 2)), total.shape[-1])
    b = beta.reshape(-1, *([1] * (group.ndim - 1)))
    return np.clip(t + (group - t) / b, 0.0, 1.0)


def contrast_metrics(est: np.ndarray, gold: np.ndarray) -> dict:
    return {"slope": slope(gold, est), "biased": (bh(boot_p(est - gold)) < ALPHA).mean() * 100,
            "mae": np.abs(est[0] - gold[0]).mean() * 100}


def prevalence_metrics(est: np.ndarray, gold: np.ndarray) -> dict:
    ratio = est.sum(1) / gold.sum(1)
    return {"total_err": abs(ratio[0] - 1) * 100, "prev_mae": np.abs(est[0] - gold[0]).mean() * 100,
            "prev_biased": int((bh(boot_p(est - gold)) < ALPHA).sum())}


def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    T = Groups(document_design(gt))
    V = Groups(document_design(gv, seed=RNG_SEED + 1))
    langs = sorted(set(T.lang))
    val_lang_sent = gv["Text-ID"].str.split("_").str[0].to_numpy()
    val_news_sent = ~gv["Text-ID"].str.contains("_M_").to_numpy()
    every_val_sent = np.ones(len(gv), bool)
    man_t, news_t = T.genre == "manifesto", T.genre == "news"
    man_v, news_v = V.genre == "manifesto", V.genre == "news"

    gold_t, gold_v = T.S @ yt, V.S @ yv
    gold_lang = T.langs(gold_t, langs)
    gold_lang_news = T.langs(gold_t, langs, news_t)
    gold_man = T.prev(gold_t, man_t)
    gold_genre = gold_man - T.prev(gold_t, news_t)
    lpairs = all_pairs(len(langs))
    n_gold_sig_news = int((bh(boot_p(pair_contrasts(gold_lang_news, lpairs))) < ALPHA).sum())
    n_gold_sig_all = int((bh(boot_p(pair_contrasts(gold_lang, lpairs))) < ALPHA).sum())

    def val_lang_slope(dec_v: np.ndarray, subset: list[str], doc_mask: np.ndarray | None = None) -> np.ndarray:
        pairs = all_pairs(len(subset))
        e = pair_contrasts(V.langs(V.S @ dec_v, subset, doc_mask), pairs)
        g = pair_contrasts(V.langs(gold_v, subset, doc_mask), pairs)
        return safe_slope(slope(g, e))

    def val_genre_slope(dec_v: np.ndarray) -> np.ndarray:
        c = V.S @ dec_v
        return safe_slope(slope(V.prev(gold_v, man_v) - V.prev(gold_v, news_v), V.prev(c, man_v) - V.prev(c, news_v)))

    def fitted(pv: np.ndarray, pt: np.ndarray, val_sent_mask: np.ndarray):
        """Fit every rule on the selected validation sentences; per-document test counts per rule,
        and the CC@prev decisions on the whole validation split (for slopes on any subset)."""
        fit = fit_rules(pv[val_sent_mask], yv[val_sent_mask])
        est = apply_rules(pt, fit)
        return {r: T.S @ est[r] for r in ["CC@t*", "CC@prev", "PCC-Platt"]}, (pv >= fit["tprev"]).astype(int)

    res: dict[tuple[str, str, str], list[dict]] = {}
    add = lambda key, m: res.setdefault(key, []).append(m)
    beta_log: dict[str, list] = {s: [] for s in SYSTEMS}

    for system in SYSTEMS:
        for seed in system_seeds(system):
            pv, pt = system_probs(system, seed, "val"), system_probs(system, seed, "test")
            full_counts, full_dec_v = fitted(pv, pt, every_val_sent)
            beta_full_lang = val_lang_slope(full_dec_v, langs)

            # ---- A. leave one language out, and the in-distribution reference
            for scen in ["full fit", "leave-one-language-out"]:
                est_c = {r: [] for r in RULES}; gold_c = {r: [] for r in RULES}; prevs = {r: [] for r in RULES}
                betas = []
                for li, held in enumerate(langs):
                    others = [l for l in langs if l != held]
                    oi = [langs.index(l) for l in others]
                    if scen == "full fit":
                        counts, beta = full_counts, beta_full_lang
                    else:
                        counts, dec_v = fitted(pv, pt, val_lang_sent != held)
                        beta = val_lang_slope(dec_v, others)
                    betas.append(beta[0])
                    for r in RULES:
                        c = counts[BASE[r]]
                        lp, total = T.langs(c, langs), T.prev(c, T.all)
                        if r == "RC@prev":
                            lp = rc(lp, total, beta)
                        est_c[r].append((lp[:, [li]] - lp[:, oi]).reshape(B + 1, -1))
                        gold_c[r].append((gold_lang[:, [li]] - gold_lang[:, oi]).reshape(B + 1, -1))
                        prevs[r].append(prevalence_metrics(lp[:, li], gold_lang[:, li]))
                if scen == "leave-one-language-out":
                    beta_log[system].append(betas)
                for r in RULES:
                    p = pd.DataFrame(prevs[r])
                    add((system, f"A {scen}", r), {
                        **contrast_metrics(np.concatenate(est_c[r], axis=1), np.concatenate(gold_c[r], axis=1)),
                        "total_err": p["total_err"].mean(), "total_err_max": p["total_err"].max(),
                        "prev_mae": p["prev_mae"].mean(), "prev_biased": p["prev_biased"].mean()})

            # ---- B. genre transfer: fit on news validation only, estimate manifestos
            news_counts, news_dec_v = fitted(pv, pt, val_news_sent)
            beta_proxy = val_lang_slope(news_dec_v, langs, news_v)
            beta_genre = val_genre_slope(full_dec_v)
            for scen, counts, beta in [("full fit", full_counts, beta_genre),
                                       ("news-only fit", news_counts, beta_proxy)]:
                for r in RULES:
                    c = counts[BASE[r]]
                    total, m, n = T.prev(c, T.all), T.prev(c, man_t), T.prev(c, news_t)
                    if r == "RC@prev":
                        m, n = rc(m, total, beta), rc(n, total, beta)
                    add((system, f"B {scen}", r), {**contrast_metrics(m - n, gold_genre),
                                                   **prevalence_metrics(m, gold_man), "beta": beta[0]})

            # ---- C. language contrasts within news only (full fit), next to all documents
            beta_news_lang = val_lang_slope(full_dec_v, langs, news_v)
            for scen, mask, gold, beta in [("all documents", None, gold_lang, beta_full_lang),
                                           ("news only", news_t, gold_lang_news, beta_news_lang)]:
                for r in RULES:
                    c = full_counts[BASE[r]]
                    lp = T.langs(c, langs, mask)
                    if r == "RC@prev":
                        lp = rc(lp, T.prev(c, T.all if mask is None else mask), beta)
                    add((system, f"C {scen}", r), {**contrast_metrics(pair_contrasts(lp, lpairs),
                                                                      pair_contrasts(gold, lpairs)), "beta": beta[0]})

    # ---- report
    def slope_ci(system: str, scen: str, r: str) -> str:
        return ci(np.mean([m["slope"] for m in res[(system, scen, r)]], axis=0))

    def avg(system: str, scen: str, r: str, k: str) -> float:
        return float(np.mean([m[k] for m in res[(system, scen, r)]]))

    out("SHIFT TESTS -- ValueEval'24 English test split; rules fitted on validation subsets")
    out(f"Stratified document bootstrap B = {B} (test and validation); metrics per run, averaged over runs.")
    out()

    out("=" * 112)
    out("A. LEAVE ONE LANGUAGE OUT: each language estimated with every rule fitted without its validation documents")
    out("   Contrasts: the held-out language vs each of the other eight (152 per language, 1,368 pooled).")
    out("   Total err: |predicted / gold value mentions - 1| for the held-out language, mean (max) over languages, %.")
    out("   Biased values: values significantly mis-estimated for the held-out language (of 19), mean over languages.")
    out("=" * 112)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'fit':24s} {'rule':10s} {'contrast slope':>20s} {'biased %':>9s} {'MAE pp':>7s} {'total err %':>16s} "
            f"{'prev MAE pp':>12s} {'biased values':>14s}")
        for scen in ["full fit", "leave-one-language-out"]:
            for r in RULES:
                k = f"A {scen}"
                out(f"  {scen:24s} {r:10s} {slope_ci(system, k, r):>20s} {avg(system, k, r, 'biased'):9.1f} "
                    f"{avg(system, k, r, 'mae'):7.2f} {avg(system, k, r, 'total_err'):8.1f} ({avg(system, k, r, 'total_err_max'):5.1f}) "
                    f"{avg(system, k, r, 'prev_mae'):12.2f} {avg(system, k, r, 'prev_biased'):14.1f}")
        b = np.array(beta_log[system])
        out(f"  validation language slope without the held-out language: {b.min():.2f}-{b.max():.2f} "
            f"(full fit: {np.mean([m['beta'] for m in res[(system, 'C all documents', 'RC@prev')]]):.2f})")

    out()
    out("=" * 112)
    out("B. GENRE TRANSFER: rules fitted on news validation only; manifesto prevalence and manifesto-news contrasts")
    out("   With no manifesto validation data, RC divides by the language-pair slope measured within news validation.")
    out("   Total err: |predicted / gold value mentions in manifestos - 1|, %.")
    out("=" * 112)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'fit':16s} {'rule':10s} {'genre slope':>20s} {'biased %':>9s} {'MAE pp':>7s} {'total err %':>12s} "
            f"{'prev MAE pp':>12s} {'biased values':>14s} {'RC slope':>9s}")
        for scen in ["full fit", "news-only fit"]:
            for r in RULES:
                k = f"B {scen}"
                beta = f"{avg(system, k, r, 'beta'):.2f}" if r == "RC@prev" else ""
                out(f"  {scen:16s} {r:10s} {slope_ci(system, k, r):>20s} {avg(system, k, r, 'biased'):9.1f} "
                    f"{avg(system, k, r, 'mae'):7.2f} {avg(system, k, r, 'total_err'):12.1f} "
                    f"{avg(system, k, r, 'prev_mae'):12.2f} {avg(system, k, r, 'prev_biased'):14.1f} {beta:>9s}")

    out()
    out("=" * 112)
    out("C. LANGUAGE CONTRASTS WITHIN NEWS ONLY (full fit), next to all documents")
    out(f"   Gold-significant language-pair contrasts: {n_gold_sig_all} of {len(lpairs) * 19} on all documents, "
        f"{n_gold_sig_news} within news.")
    out("   RC divides by the language slope measured on validation over the same documents (all / news only).")
    out("=" * 112)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'documents':14s} {'rule':10s} {'language slope':>20s} {'biased %':>9s} {'MAE pp':>7s} {'RC slope':>9s}")
        for scen in ["all documents", "news only"]:
            for r in RULES:
                k = f"C {scen}"
                beta = f"{avg(system, k, r, 'beta'):.2f}" if r == "RC@prev" else ""
                out(f"  {scen:14s} {r:10s} {slope_ci(system, k, r):>20s} {avg(system, k, r, 'biased'):9.1f} "
                    f"{avg(system, k, r, 'mae'):7.2f} {beta:>9s}")

    # ---- write
    RES.mkdir(exist_ok=True)
    rows = []
    for (system, scen, r), ms in res.items():
        sl = np.mean([m["slope"] for m in ms], axis=0)
        row = {"system": system, "scenario": scen, "rule": r, "slope": sl[0],
               "slope_lo": np.percentile(sl[1:], 2.5), "slope_hi": np.percentile(sl[1:], 97.5)}
        for k in ["biased", "mae", "total_err", "total_err_max", "prev_mae", "prev_biased", "beta"]:
            if k in ms[0]:
                row[k] = float(np.mean([m[k] for m in ms]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "shift-tests.csv", index=False, float_format="%.5f")
    (RES / "shift-tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'shift-tests.txt'} and shift-tests.csv")


if __name__ == "__main__":
    main()
