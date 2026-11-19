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
     the genre slope cannot be estimated, so SC uses the language-pair slope measured within
     news validation as a proxy: a stress test of borrowing a slope across kinds of contrast.
     Reference: the full fit, with SC using the genre slope measured on validation.
  C. Language contrasts within news only (full fit): is the attenuation of language contrasts
     an artefact of languages differing in their share of manifestos? SC uses the language
     slope measured within news validation.

Rules: every estimator of measurement_analysis.py that needs no annotated documents of the
group being estimated: CC@t* (macro-F1 threshold, the shared-task practice), CC@prev
(prevalence-matched per-label thresholds), PCC (raw probability average), PCC-Platt
(calibrated probability average), ACC@prev and PACC (adjusted counts with error rates from the
validation subset), EM (prior shift, with the subset's Platt maps and prevalence), and SC@prev
(CC@prev with group deviations from the corpus total divided by a validation attenuation slope).
The estimators that use each group's own annotated documents (ACC-G, Gold-only, PPI, Platt-G)
cannot be computed for a language left out of the fit or for manifestos under a news-only fit:
that is the situation these tests describe. measurement_analysis.py reports them with the full fit.

Metrics, as in measurement_analysis.py: contrast slope (OLS of estimated on gold contrasts;
1 = faithful), share of contrasts with differential bias (paired bootstrap, BH q < 0.05),
MAE in percentage points; for prevalence, the error in a group's total value mentions,
MAE over values, and the number of values significantly mis-estimated. Uncertainty: the
same stratified document bootstraps of test and validation as measurement_analysis.py, with
every rule refitted on each validation resample (RowFits). A system's estimate is the mean of
its runs, and every metric and test is on that mean.

Usage:
    python3 shift_tests.py
Writes results/shift-tests.txt and results/shift-tests.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_provenance import VALUES, load_gold
from measurement_analysis import (ALPHA, B, MIN_SLOPE, RES, RNG_SEED, SYSTEMS, EMGroups, RowFits, adjust, bh,
                                  boot_p, ci, document_design, lpair_reference_p, slope, system_probs, system_seeds,
                                  wsum)

RULES = ["CC@t*", "CC@prev", "PCC", "PCC-Platt", "ACC@prev", "PACC", "EM", "SC@prev"]


class Groups:
    """Bootstrapped prevalence of any set of documents of one split (row 0 = original sample)."""

    def __init__(self, design: dict):
        self.S, self.W, self.nsent = design["S"], design["W"], design["nsent"]
        self.lang, self.genre = design["lang"], design["genre"]
        self.all = np.ones(len(self.lang), bool)

    def prev(self, cdoc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Prevalence per row from per-document counts, fixed (docs, k) or per row (B+1, docs, k)."""
        return wsum(self.W, cdoc, mask) / (self.W[:, mask] @ self.nsent[mask])[:, None]

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


def sc(group: np.ndarray, total: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Slope correction: divide a group's deviation from the corpus total by the slope."""
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
    Vdesign = document_design(gv, seed=RNG_SEED + 1)
    V = Groups(Vdesign)
    langs = sorted(set(T.lang))
    val_lang_sent = gv["Text-ID"].str.split("_").str[0].to_numpy()
    val_news_sent = ~gv["Text-ID"].str.contains("_M_").to_numpy()
    every_val_sent = np.ones(len(gv), bool)
    man_t, news_t = T.genre == "manifesto", T.genre == "news"
    man_v, news_v = V.genre == "manifesto", V.genre == "news"

    gold_t, gold_v = T.S @ yt, V.S @ yv
    gold_lang = T.langs(gold_t, langs)
    gold_lang_news = T.langs(gold_t, langs, news_t)
    gold_man, gold_news = T.prev(gold_t, man_t), T.prev(gold_t, news_t)
    gold_genre = gold_man - gold_news
    lpairs = all_pairs(len(langs))
    lang_masks_t = [T.lang == l for l in langs]
    n_gold_sig_news = int((bh(lpair_reference_p(pair_contrasts(gold_lang_news, lpairs), gold_t, T.nsent,
                                                [m & news_t for m in lang_masks_t], lpairs)) < ALPHA).sum())
    n_gold_sig_all = int((bh(lpair_reference_p(pair_contrasts(gold_lang, lpairs), gold_t, T.nsent,
                                               lang_masks_t, lpairs)) < ALPHA).sum())

    def val_lang_slope(cv: np.ndarray, subset: list[str], doc_mask: np.ndarray | None = None) -> np.ndarray:
        """Validation language-pair slope from per-row validation counts of CC@prev decisions."""
        pairs = all_pairs(len(subset))
        e = pair_contrasts(V.langs(cv, subset, doc_mask), pairs)
        g = pair_contrasts(V.langs(gold_v, subset, doc_mask), pairs)
        return safe_slope(slope(g, e))

    def val_genre_slope(c: np.ndarray) -> np.ndarray:
        return safe_slope(slope(V.prev(gold_v, man_v) - V.prev(gold_v, news_v), V.prev(c, man_v) - V.prev(c, news_v)))

    class Fit:
        """Every rule refitted on each validation bootstrap row, using only the selected validation sentences.
        prev(rule, mask) gives a group's prevalence per row; cv holds the per-row CC@prev counts on the whole
        validation split (for slopes on any subset)."""

        def __init__(self, pv: np.ndarray, pt: np.ndarray, val_sent_mask: np.ndarray):
            rf = RowFits(pv, yv, Vdesign, sent_mask=val_sent_mask)
            self.counts = {r: rf.counts(r, pt, T.S) for r in ["CC@t*", "CC@prev", "PCC-Platt"]}
            self.counts["PCC"] = T.S @ pt
            self.cv = rf.counts("CC@prev", pv, V.S)
            w = val_sent_mask[:, None].astype(float)            # error rates on the selected sentences only
            pos, neg = V.S @ (yv * w), V.S @ ((1 - yv) * w)
            tp = rf.counts("CC@prev", pv, V.S, weight=yv * w)
            fp = rf.counts("CC@prev", pv, V.S, weight=(1 - yv) * w)
            with np.errstate(invalid="ignore", divide="ignore"):
                P, N = V.W @ pos, V.W @ neg
                self.rates = {"ACC@prev": (wsum(V.W, tp, V.all) / P, wsum(V.W, fp, V.all) / N),
                              "PACC": ((V.W @ (V.S @ (pv * yv * w))) / P, (V.W @ (V.S @ (pv * (1 - yv) * w))) / N)}
            self.em = EMGroups(T.S, T.W, pt, rf.platt, rf.prior)
            self._em: dict[bytes, np.ndarray] = {}

        def prev(self, rule: str, mask: np.ndarray) -> np.ndarray:
            if rule in self.counts:
                return T.prev(self.counts[rule], mask)
            if rule == "ACC@prev":
                return adjust(T.prev(self.counts["CC@prev"], mask), *self.rates[rule])
            if rule == "PACC":
                return adjust(T.prev(self.counts["PCC"], mask), *self.rates[rule])
            if rule == "EM":
                key = np.packbits(mask).tobytes()
                if key not in self._em:
                    self._em[key] = self.em.prevalence(mask)
                return self._em[key]
            raise KeyError(rule)

        def langs(self, rule: str, doc_mask: np.ndarray | None = None) -> np.ndarray:
            m = T.all if doc_mask is None else doc_mask
            return np.stack([self.prev(rule, (T.lang == l) & m) for l in langs], axis=1)

    res: dict[tuple[str, str, str], dict] = {}
    beta_log: dict[str, list] = {s: [] for s in SYSTEMS}
    lolo_rows: list[dict] = []          # one row per system, fit, rule and left-out language

    for system in SYSTEMS:
        # sums over the system's runs of every array a metric needs; the metrics are computed on the mean
        acc: dict[tuple[str, str], dict] = {}

        def add(scen: str, r: str, **arrays) -> None:
            slot = acc.setdefault((scen, r), {"n": 0})
            slot["n"] += 1
            for k, a in arrays.items():
                slot[k] = slot.get(k, 0) + np.asarray(a, float)

        for seed in system_seeds(system):
            pv, pt = system_probs(system, seed, "val"), system_probs(system, seed, "test")
            full = Fit(pv, pt, every_val_sent)
            beta_full_lang = val_lang_slope(full.cv, langs)

            # ---- A. leave one language out, and the in-distribution reference
            for scen in ["full fit", "leave-one-language-out"]:
                est_c = {r: [] for r in RULES}; held_prev = {r: [] for r in RULES}
                betas = []
                for li, held in enumerate(langs):
                    others = [l for l in langs if l != held]
                    oi = [langs.index(l) for l in others]
                    if scen == "full fit":
                        fit, beta = full, beta_full_lang
                    else:
                        fit = Fit(pv, pt, val_lang_sent != held)
                        beta = val_lang_slope(fit.cv, others)
                    betas.append(beta[0])
                    for r in RULES:
                        base = "CC@prev" if r == "SC@prev" else r
                        lp = fit.langs(base)
                        if r == "SC@prev":
                            lp = sc(lp, fit.prev(base, T.all), beta)
                        est_c[r].append((lp[:, [li]] - lp[:, oi]).reshape(B + 1, -1))
                        held_prev[r].append(lp[:, li])
                if scen == "leave-one-language-out":
                    beta_log[system].append(betas)
                for r in RULES:
                    add(f"A {scen}", r, est=np.concatenate(est_c[r], axis=1), prev=np.stack(held_prev[r], axis=1),
                        beta=np.mean(betas))

            # ---- B. genre transfer: fit on news validation only, estimate manifestos
            news_fit = Fit(pv, pt, val_news_sent)
            beta_proxy = val_lang_slope(news_fit.cv, langs, news_v)
            beta_genre = val_genre_slope(full.cv)
            for scen, fit, beta in [("full fit", full, beta_genre), ("news-only fit", news_fit, beta_proxy)]:
                for r in RULES:
                    base = "CC@prev" if r == "SC@prev" else r
                    m, n = fit.prev(base, man_t), fit.prev(base, news_t)
                    if r == "SC@prev":
                        total = fit.prev(base, T.all)
                        m, n = sc(m, total, beta), sc(n, total, beta)
                    add(f"B {scen}", r, est=m - n, man=m, news=n, beta=beta[0])

            # ---- C. language contrasts within news only (full fit), next to all documents
            beta_news_lang = val_lang_slope(full.cv, langs, news_v)
            for scen, mask, beta in [("all documents", None, beta_full_lang), ("news only", news_t, beta_news_lang)]:
                for r in RULES:
                    base = "CC@prev" if r == "SC@prev" else r
                    lp = full.langs(base, mask)
                    if r == "SC@prev":
                        lp = sc(lp, full.prev(base, T.all if mask is None else mask), beta)
                    add(f"C {scen}", r, est=pair_contrasts(lp, lpairs), beta=beta[0])

        # ---- metrics on the mean of the runs
        held_gold = (np.concatenate([(gold_lang[:, [li]] - gold_lang[:, [langs.index(l) for l in langs if l != h]])
                                     .reshape(B + 1, -1) for li, h in enumerate(langs)], axis=1))
        golds = {"A": held_gold, "B": gold_genre, "C all documents": pair_contrasts(gold_lang, lpairs),
                 "C news only": pair_contrasts(gold_lang_news, lpairs)}
        for (scen, r), slot in acc.items():
            mean = {k: v / slot["n"] for k, v in slot.items() if k != "n"}
            gold = golds.get(scen, golds.get(scen[0]))
            m = {**contrast_metrics(mean["est"], gold), "beta": float(mean["beta"])}
            if scen.startswith("A"):
                pm = pd.DataFrame([prevalence_metrics(mean["prev"][:, li], gold_lang[:, li]) for li in range(len(langs))])
                width = (len(langs) - 1) * len(VALUES)            # the left-out language against the other eight
                for li, held in enumerate(langs):
                    cols = slice(li * width, (li + 1) * width)
                    sl = slope(gold[:, cols], mean["est"][:, cols])
                    lo, hi = np.percentile(sl[1:], [2.5, 97.5])
                    fitted = (float(np.mean([b[li] for b in beta_log[system]])) if scen.endswith("out")
                              else float(mean["beta"]))
                    lolo_rows.append({"system": system, "fit": scen[2:], "rule": r, "language": held,
                                      "fitted_slope": fitted, "contrast_slope": sl[0], "contrast_slope_lo": lo,
                                      "contrast_slope_hi": hi,
                                      "mae": np.abs(mean["est"][0, cols] - gold[0, cols]).mean() * 100,
                                      "total_err": pm["total_err"][li], "prev_mae": pm["prev_mae"][li]})
                m.update({"total_err": pm["total_err"].mean(), "total_err_max": pm["total_err"].max(),
                          "prev_mae": pm["prev_mae"].mean(), "prev_biased": pm["prev_biased"].mean()})
            elif scen.startswith("B"):
                m.update(prevalence_metrics(mean["man"], gold_man))
                m.update({f"news_{k}": v for k, v in prevalence_metrics(mean["news"], gold_news).items()})
            res[(system, scen, r)] = m

    # ---- report
    def slope_ci(system: str, scen: str, r: str) -> str:
        return ci(res[(system, scen, r)]["slope"])

    def avg(system: str, scen: str, r: str, k: str) -> float:
        return float(res[(system, scen, r)][k])

    out("SHIFT TESTS -- ValueEval'24 English test split; rules fitted on validation subsets")
    out("Estimators that need annotated documents of the group itself (ACC-G, Gold-only, PPI, Platt-G) cannot be")
    out("computed for a language left out of the fit or for manifestos under a news-only fit; see measurement-analysis.txt.")
    out(f"Stratified document bootstrap B = {B} (test and validation); metrics on the mean of a system's runs.")
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
            f"(full fit: {res[(system, 'C all documents', 'SC@prev')]['beta']:.2f})")
    lolo = pd.DataFrame(lolo_rows)
    main_sys = SYSTEMS[0]
    out(f"\n  Each left-out language, {main_sys} (AE item 5): fitted = validation language slope without it;")
    out("  slopes and MAE over its 152 contrasts with the other eight; total err as above.")
    out(f"  {'language':8s} {'fitted':>6s} {'CC@prev slope':>21s} {'SC@prev slope':>21s} {'MAE CC':>7s} {'MAE SC':>7s} "
        f"{'total err CC':>13s} {'total err SC':>13s}")
    sel = lolo[(lolo["system"] == main_sys) & (lolo["fit"] == "leave-one-language-out")].set_index(["rule", "language"])
    for held in langs:
        cc, scr = sel.loc[("CC@prev", held)], sel.loc[("SC@prev", held)]
        cell = lambda x: f"{x['contrast_slope']:.2f} [{x['contrast_slope_lo']:.2f}, {x['contrast_slope_hi']:.2f}]"
        out(f"  {held:8s} {cc['fitted_slope']:6.2f} {cell(cc):>21s} {cell(scr):>21s} {cc['mae']:7.2f} {scr['mae']:7.2f} "
            f"{cc['total_err']:13.1f} {scr['total_err']:13.1f}")
    for r in ["CC@prev", "SC@prev"]:
        x = sel.loc[r]
        out(f"  {r}: contrast slope over the nine languages {x['contrast_slope'].min():.2f}-{x['contrast_slope'].max():.2f} "
            f"(median {x['contrast_slope'].median():.2f}); MAE {x['mae'].min():.2f}-{x['mae'].max():.2f}; "
            f"total err {x['total_err'].min():.1f}-{x['total_err'].max():.1f}%")

    out()
    out("=" * 112)
    out("B. GENRE TRANSFER: rules fitted on news validation only; manifesto prevalence and manifesto-news contrasts")
    out("   With no manifesto validation data, SC divides by the language-pair slope measured within news validation.")
    out("   Total err: |predicted / gold value mentions in manifestos - 1|, %; news: the same for news documents.")
    out("=" * 112)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'fit':16s} {'rule':10s} {'genre slope':>20s} {'biased %':>9s} {'MAE pp':>7s} {'total err %':>12s} "
            f"{'prev MAE pp':>12s} {'biased values':>14s} {'news err %':>11s} {'SC slope':>9s}")
        for scen in ["full fit", "news-only fit"]:
            for r in RULES:
                k = f"B {scen}"
                beta = f"{avg(system, k, r, 'beta'):.2f}" if r == "SC@prev" else ""
                out(f"  {scen:16s} {r:10s} {slope_ci(system, k, r):>20s} {avg(system, k, r, 'biased'):9.1f} "
                    f"{avg(system, k, r, 'mae'):7.2f} {avg(system, k, r, 'total_err'):12.1f} "
                    f"{avg(system, k, r, 'prev_mae'):12.2f} {avg(system, k, r, 'prev_biased'):14.1f} "
                    f"{avg(system, k, r, 'news_total_err'):11.1f} {beta:>9s}")

    out()
    out("=" * 112)
    out("C. LANGUAGE CONTRASTS WITHIN NEWS ONLY (full fit), next to all documents")
    out(f"   Gold-significant language-pair contrasts: {n_gold_sig_all} of {len(lpairs) * 19} on all documents, "
        f"{n_gold_sig_news} within news.")
    out("   SC divides by the language slope measured on validation over the same documents (all / news only).")
    out("=" * 112)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'documents':14s} {'rule':10s} {'language slope':>20s} {'biased %':>9s} {'MAE pp':>7s} {'SC slope':>9s}")
        for scen in ["all documents", "news only"]:
            for r in RULES:
                k = f"C {scen}"
                beta = f"{avg(system, k, r, 'beta'):.2f}" if r == "SC@prev" else ""
                out(f"  {scen:14s} {r:10s} {slope_ci(system, k, r):>20s} {avg(system, k, r, 'biased'):9.1f} "
                    f"{avg(system, k, r, 'mae'):7.2f} {beta:>9s}")

    # ---- write
    RES.mkdir(exist_ok=True)
    rows = []
    for (system, scen, r), m in res.items():
        sl = m["slope"]
        row = {"system": system, "scenario": scen, "rule": r, "slope": sl[0],
               "slope_lo": np.percentile(sl[1:], 2.5), "slope_hi": np.percentile(sl[1:], 97.5)}
        for k in ["biased", "mae", "total_err", "total_err_max", "prev_mae", "prev_biased", "news_total_err",
                  "news_prev_mae", "news_prev_biased", "beta"]:
            if k in m:
                row[k] = float(m[k])
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "shift-tests.csv", index=False, float_format="%.5f")
    lolo.to_csv(RES / "shift-lolo-languages.csv", index=False, float_format="%.5f")
    (RES / "shift-tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'shift-tests.txt'} and shift-tests.csv")


if __name__ == "__main__":
    main()
