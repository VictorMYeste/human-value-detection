#!/usr/bin/env python3
"""
Measurement error of LLM labels: can 7-9B instruction-tuned LLMs be used to count values?

LLM annotation is increasingly used in computational social science to measure how often
texts express a category. Here the LLM systems of the paper return a set of values per
sentence (0/1 labels, no scores), so no threshold can be tuned: their labels are classify-
and-count estimates as they stand. This script measures how far those counts and group
comparisons are from the gold standard, with the metrics of measurement_analysis.py, and
whether the two corrections that need only validation labels repair them:

  CC     the LLM labels as returned (classify and count);
  ACC    adjusted count, (CC - fpr) / (tpr - fpr) with tpr and fpr of the LLM on validation;
  RC     regression calibration: group deviations from the corpus total divided by the
         attenuation slope of the LLM's own validation contrasts (language pairs for
         languages, manifesto-news for genre).

LLM systems (single runs; the paper's Section on LLM baselines):
  Gemma-2-9B few-shot + gate  definition prompt, 20 examples, SBERT presence gate (best single LLM)
  Gemma-2-9B few-shot         the same without the gate
  Gemma-2-9B QLoRA            QLoRA direct configuration
  LLM ensemble                hard vote of the gated few-shot and the QLoRA Gemma models

Reference: the feature ensemble (the paper's main system, three seeds) with CC@t*, CC@prev
and RC@prev, computed here with the same code: rules refitted on every validation resample and
metrics on the mean of the three runs, as in measurement_analysis.py.

Uncertainty: the stratified document bootstraps of test and validation shared with
measurement_analysis.py (B = 2000). LLM results are single runs, so their intervals cover
sampling of documents only, not run-to-run variation.

Unparseable answers (step 5.2, AE item 6): the LLM runner turned an answer it could not parse
into an empty set, the same as "no value", and kept no raw generations. Section C therefore
reports empty answers by genre and language, an upper bound on unparseable ones, next to the
expert labels and the main system. Section D gives the Stimulation contrast of every system, on
test and validation, and the most that empty answers could move it.

Usage:
    python3 llm_measurement.py
Writes results/llm-measurement.txt, results/llm-measurement.csv and results/llm-empty-answers.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_provenance import KEY, VALUES, load_gold
from measurement_analysis import (ALPHA, B, RES, RNG_SEED, SEEDS, RowFits, adjust, bh, boot_p, boot_rates, ci,
                                  genre_reference_p, lpair_reference_p, doc_index,
                                  document_design, macro_f1, rank_corr, slope, system_probs)
from shift_tests import Groups, all_pairs, pair_contrasts, rc, safe_slope

HERE = Path(__file__).resolve().parent
LLM_OUT = HERE / ".." / "llm_moral-values" / "output"
LLM_ENS = HERE / ".." / "ensembles" / "llm" / "all"
LLM_SYSTEMS = {  # name -> (file pattern, published test macro-F1)
    "Gemma-2-9B few-shot + gate": (LLM_OUT / "definition-20" / "google-gemma-2-9b-it-gated_Presence-{}.tsv", 0.238),
    "Gemma-2-9B few-shot": (LLM_OUT / "definition-20" / "google-gemma-2-9b-it-{}.tsv", 0.233),
    "Gemma-2-9B QLoRA": (LLM_OUT / "qlora-direct-lr-2e-4" / "google-gemma-2-9b-it-{}.tsv", 0.236),
    "LLM ensemble (hard vote)": (LLM_ENS / "fixed-hard-champion-{}.tsv", 0.277),
}
REFERENCE = "Feature ensemble"


def load_labels(pattern: Path, split: str) -> np.ndarray:
    gold = load_gold("validation" if split == "val" else "test")
    pred = pd.read_csv(str(pattern).format(split), sep="\t")[KEY + VALUES]
    merged = gold[KEY].merge(pred, on=KEY, how="left", validate="1:1")
    if merged[VALUES].isna().any().any():
        raise ValueError(f"{pattern} {split}: labels missing for some sentences")
    labels = merged[VALUES].to_numpy(dtype=float)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"{pattern} {split}: expected 0/1 labels")
    return labels.astype(int)


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
    lpairs = all_pairs(len(langs))
    man_t, news_t = T.genre == "manifesto", T.genre == "news"
    man_v, news_v = V.genre == "manifesto", V.genre == "news"
    gold_v = V.S @ yv

    def views(total, man, news, lang) -> dict[str, np.ndarray]:
        return {"all": total, "genre": man - news, "lang": lang, "lpair": pair_contrasts(lang, lpairs)}

    def cc_views(c: np.ndarray) -> dict:
        """Views from per-document test counts, fixed (docs, k) or per bootstrap row (B+1, docs, k)."""
        return views(T.prev(c, T.all), T.prev(c, man_t), T.prev(c, news_t), T.langs(c, langs))

    def acc_views(dec_t: np.ndarray, dec_v: np.ndarray) -> dict:
        tpr, fpr = boot_rates(V.S, V.W, yv, dec_v)
        c = T.S @ dec_t
        return views(adjust(T.prev(c, T.all), tpr, fpr), adjust(T.prev(c, man_t), tpr, fpr),
                     adjust(T.prev(c, news_t), tpr, fpr), adjust(T.langs(c, langs), tpr[:, None], fpr[:, None]))

    def rc_views(c: np.ndarray, cv: np.ndarray) -> tuple[dict, float, float]:
        """Regression calibration from per-document test counts c and validation counts cv."""
        e_l, g_l = pair_contrasts(V.langs(cv, langs), lpairs), pair_contrasts(V.langs(gold_v, langs), lpairs)
        beta_l = safe_slope(slope(g_l, e_l))
        beta_g = safe_slope(slope(V.prev(gold_v, man_v) - V.prev(gold_v, news_v), V.prev(cv, man_v) - V.prev(cv, news_v)))
        total = T.prev(c, T.all)
        return (views(total, rc(T.prev(c, man_t), total, beta_g), rc(T.prev(c, news_t), total, beta_g),
                      rc(T.langs(c, langs), total, beta_l)), beta_l[0], beta_g[0])

    gold = cc_views(T.S @ yt)
    gold_doc = T.S @ yt
    gsig = {"genre": bh(genre_reference_p(gold["genre"], gold_doc, T.nsent, man_t, news_t)) < ALPHA,
            "lpair": bh(lpair_reference_p(gold["lpair"], gold_doc, T.nsent, [T.lang == l for l in langs], lpairs)) < ALPHA}

    def metrics(v: dict) -> dict:
        diff = v["all"] - gold["all"]
        sig = bh(boot_p(diff)) < ALPHA
        m = {"ratio": v["all"].sum(1) / gold["all"].sum(1), "mae": np.abs(diff[0]).mean() * 100,
             "under": int((sig & (diff[0] < 0)).sum()), "over": int((sig & (diff[0] > 0)).sum()),
             "rho": rank_corr(gold["lang"], v["lang"]), "cover": (boot_p(diff) >= ALPHA).mean() * 100}
        for key in ["genre", "lpair"]:
            e0, g0, gs = v[key][0], gold[key][0], gsig[key]
            esig = bh(boot_p(v[key])) < ALPHA
            same = np.sign(e0) == np.sign(g0)
            m.update({f"{key}_reproduced": (esig & same)[gs].mean() * 100,
                      f"{key}_detected": ((boot_p(v[key]) < ALPHA) & same)[gs].mean() * 100,
                      f"{key}_reversed": (esig & ~same)[gs].mean() * 100,
                      f"{key}_biased": (bh(boot_p(v[key] - gold[key])) < ALPHA).mean() * 100,
                      f"{key}_mae": np.abs(e0 - g0).mean() * 100,
                      f"{key}_cover": (boot_p(v[key] - gold[key]) >= ALPHA).mean() * 100,
                      f"{key}_slope": slope(gold[key], v[key])})
        return m

    results: dict[str, dict[str, list[dict]]] = {}
    f1s: dict[str, dict[str, list[float]]] = {}
    slopes_used: dict[str, list[tuple[float, float]]] = {}

    def record(system: str, est: str, v: dict, f1: float | None = None) -> None:
        results.setdefault(system, {}).setdefault(est, []).append(metrics(v))
        if f1 is not None:
            f1s.setdefault(system, {}).setdefault(est, []).append(f1)

    # ---- reference: the main supervised system, rules refitted on every validation row; metrics on the
    #      mean of its three runs, as in measurement_analysis.py
    ref_views: dict[str, list[dict]] = {"CC@t*": [], "CC@prev": [], "RC@prev": []}
    decisions = {"test": {}, "val": {}}           # system -> 0/1 labels (main system: one array per run)
    for seed in SEEDS:
        pv, pt = system_probs(REFERENCE, seed, "val"), system_probs(REFERENCE, seed, "test")
        rf = RowFits(pv, yv, Vdesign)
        f1s.setdefault(REFERENCE, {}).setdefault("CC@t*", []).append(macro_f1(yt, (pt >= rf.tstar[0]).astype(int)))
        f1s[REFERENCE].setdefault("CC@prev", []).append(macro_f1(yt, (pt >= rf.tprev[0]).astype(int)))
        c_prev = rf.counts("CC@prev", pt, T.S)
        decisions["test"].setdefault(REFERENCE, []).append((pt >= rf.tprev[0]).astype(int))
        decisions["val"].setdefault(REFERENCE, []).append((pv >= rf.tprev[0]).astype(int))
        ref_views["CC@t*"].append(cc_views(rf.counts("CC@t*", pt, T.S)))
        ref_views["CC@prev"].append(cc_views(c_prev))
        v, bl, bg = rc_views(c_prev, rf.counts("CC@prev", pv, V.S))
        ref_views["RC@prev"].append(v)
        slopes_used.setdefault(REFERENCE, []).append((bl, bg))
        del c_prev
    for est_name, vs in ref_views.items():
        record(REFERENCE, est_name, {key: np.mean([v[key] for v in vs], axis=0) for key in vs[0]})

    # ---- LLM systems, single runs
    for system, (pattern, published) in LLM_SYSTEMS.items():
        dec_v, dec_t = load_labels(pattern, "val"), load_labels(pattern, "test")
        decisions["test"][system], decisions["val"][system] = [dec_t], [dec_v]
        f1 = macro_f1(yt, dec_t)
        if abs(f1 - published) > 0.0006:
            raise ValueError(f"{system}: test macro-F1 {f1:.4f} differs from the published {published:.3f}")
        record(system, "CC", cc_views(T.S @ dec_t), f1)
        record(system, "ACC", acc_views(dec_t, dec_v))
        v, bl, bg = rc_views(T.S @ dec_t, V.S @ dec_v)
        record(system, "RC", v)
        slopes_used[system] = [(bl, bg)]

    # ---- report
    def mean_arr(system: str, est: str, k: str) -> np.ndarray:
        return np.mean([m[k] for m in results[system][est]], axis=0)

    def avg(system: str, est: str, k: str) -> float:
        return float(np.mean([m[k] for m in results[system][est]]))

    out("LLM LABELS AS A MEASUREMENT INSTRUMENT -- ValueEval'24 English test split")
    out(f"Stratified document bootstrap B = {B}; LLM systems are single runs; the reference is the mean of three seeds.")
    out(f"Gold-significant contrasts: {gsig['genre'].sum()} of 19 manifesto-news, {gsig['lpair'].sum()} of "
        f"{len(lpairs) * 19} language-pair. Test macro-F1 of every LLM system matches the published value.")
    out()
    out("=" * 118)
    out("A. CORPUS PREVALENCE AND MANIFESTO-NEWS CONTRASTS")
    out("   pred/gold: total estimated value mentions / gold mentions [95% CI]. Biased values: significantly under / over")
    out("   (BH q < 0.05, of 19). Genre: slope of estimated on gold contrasts [95% CI]; reproduced: gold-significant contrasts")
    out("   found significant with the right sign (BH over the family); single: the same when each contrast is tested on")
    out("   its own (p < 0.05, uncorrected); biased: contrasts with differential error (BH q < 0.05).")
    out("=" * 118)
    out(f"  {'system':28s} {'est.':8s} {'macro-F1':>8s} {'pred/gold':>18s} {'MAE pp':>7s} {'under/over':>11s} "
        f"{'genre slope':>20s} {'reprod. %':>9s} {'single %':>9s} {'biased %':>9s} {'MAE pp':>7s}")
    for system, ests in results.items():
        for est in ests:
            f1 = f"{np.mean(f1s[system][est]):.3f}" if est in f1s.get(system, {}) else "--"
            under_over = f"{avg(system, est, 'under'):.0f}/{avg(system, est, 'over'):.0f}"
            out(f"  {system:28s} {est:8s} {f1:>8s} {ci(mean_arr(system, est, 'ratio')):>18s} {avg(system, est, 'mae'):7.2f} "
                f"{under_over:>11s} {ci(mean_arr(system, est, 'genre_slope')):>20s} {avg(system, est, 'genre_reproduced'):9.1f} "
                f"{avg(system, est, 'genre_detected'):9.1f} {avg(system, est, 'genre_biased'):9.1f} {avg(system, est, 'genre_mae'):7.2f}")
    out()
    out("=" * 118)
    out("B. SOURCE-LANGUAGE PAIRWISE CONTRASTS (684)")
    out("=" * 118)
    out(f"  {'system':28s} {'est.':8s} {'language slope':>20s} {'reprod. %':>9s} {'single %':>9s} {'biased %':>9s} "
        f"{'MAE pp':>7s} {'lang. rank rho':>20s} {'cover % (total/genre/lpair)':>28s}")
    for system, ests in results.items():
        for est in ests:
            out(f"  {system:28s} {est:8s} {ci(mean_arr(system, est, 'lpair_slope')):>20s} "
                f"{avg(system, est, 'lpair_reproduced'):9.1f} {avg(system, est, 'lpair_detected'):9.1f} "
                f"{avg(system, est, 'lpair_biased'):9.1f} "
                f"{avg(system, est, 'lpair_mae'):7.2f} {ci(mean_arr(system, est, 'rho')):>20s} "
                f"{avg(system, est, 'cover'):14.0f}/{avg(system, est, 'genre_cover'):3.0f}/{avg(system, est, 'lpair_cover'):3.0f}")
    out("  Cover: share of totals / contrasts whose paired error (estimate - reference) is within sampling noise")
    out(f"  (p >= {ALPHA}), as in measurement-analysis.txt section H.")
    reversals = [f"{system} {est} {'genre' if k == 'genre' else 'language'} {avg(system, est, f'{k}_reversed'):.1f}%"
                 for system, ests in results.items() for est in ests for k in ["genre", "lpair"]
                 if avg(system, est, f"{k}_reversed") > 0]
    out("\n  Gold-significant contrasts found significant with the WRONG sign: " + ("; ".join(reversals) or "none"))
    out("  Validation slopes used by RC (language, genre): " + "; ".join(
        f"{s}: " + ", ".join(f"{bl:.2f}/{bg:.2f}" for bl, bg in v) for s, v in slopes_used.items()))

    # ---- C. empty answers by group (unparseable answers are among them)
    split_y = {"test": yt, "val": yv}
    split_groups = {}
    for sp, G in [("test", T), ("val", V)]:
        d = doc_index(G.S)
        split_groups[sp] = {"all": np.ones(len(d), bool), "manifestos": G.genre[d] == "manifesto",
                            "news": G.genre[d] == "news", **{l: G.lang[d] == l for l in langs}}
    out()
    out("=" * 118)
    out("C. EMPTY ANSWERS: share of sentences labeled with no value, %, by group (test and validation)")
    out("   The LLM runner stored an answer it could not parse as an empty set and kept no raw generations, so")
    out("   unparseable answers cannot be counted. They are among the empty answers, which bound them from above.")
    out(f"   Reference: the expert labels. {REFERENCE}: prevalence-matched decisions, mean over runs.")
    out("=" * 118)
    empty_rows = []
    for sp in ["test", "val"]:
        groups = split_groups[sp]
        out(f"\n  {'test' if sp == 'test' else 'validation'}")
        out(f"  {'labels':28s} " + " ".join(f"{g[:5]:>6s}" for g in groups))
        for name, decs in [("reference", [split_y[sp]])] + list(decisions[sp].items()):
            empty = np.mean([~(d.astype(bool).any(1)) for d in decs], axis=0)
            shares = {g: empty[m].mean() * 100 for g, m in groups.items()}
            out(f"  {name:28s} " + " ".join(f"{v:6.1f}" for v in shares.values()))
            empty_rows.append({"split": sp, "labels": name, **shares})

    # ---- D. the Stimulation contrast under every system
    j = VALUES.index("Stimulation")
    out()
    out("=" * 118)
    out("D. STIMULATION, MANIFESTOS MINUS NEWS (the reversal in the Introduction), pp")
    out("   Test: difference [95% CI] and p (bootstrap); validation: difference. Bound: the most the difference could")
    out("   move toward the reference if every empty manifesto answer had been an unparsed answer carrying Stimulation")
    out("   at the system's own rate among its non-empty manifesto answers.")
    out("=" * 118)
    for name, decs in [("reference", [yt])] + list(decisions["test"].items()):
        cnt = np.mean([T.S @ d for d in decs], axis=0)
        g = T.prev(cnt, man_t)[:, j] - T.prev(cnt, news_t)[:, j]
        lo, hi = np.percentile(g[1:] * 100, [2.5, 97.5])
        dv = [split_y["val"]] if name == "reference" else decisions["val"][name]
        cv = np.mean([V.S @ d for d in dv], axis=0)
        gv_ = (V.prev(cv, V.genre == "manifesto")[0, j] - V.prev(cv, V.genre == "news")[0, j]) * 100
        bound = "--"
        if name in LLM_SYSTEMS:
            dm = np.mean([d for d in decs], axis=0)
            m = split_groups["test"]["manifestos"]
            empty = ~(dm[m] > 0).any(1)
            rate = dm[m][~empty, j].mean()
            bound = f"{empty.mean() * rate * 100:.2f}"
        out(f"  {name:28s} test {g[0] * 100:+6.1f} [{lo:+.1f}, {hi:+.1f}]  p {boot_p(g[:, None])[0]:.3f}   "
            f"validation {gv_:+6.1f}   bound {bound}")

    # ---- write
    pd.DataFrame(empty_rows).to_csv(RES / "llm-empty-answers.csv", index=False, float_format="%.3f")
    rows = []
    for system, ests in results.items():
        for est, ms in ests.items():
            row = {"system": system, "estimator": est, "runs": len(ms),
                   "macro_f1": np.mean(f1s[system][est]) if est in f1s.get(system, {}) else np.nan}
            for k in ["ratio", "genre_slope", "lpair_slope", "rho"]:
                a = mean_arr(system, est, k)
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = a[0], np.percentile(a[1:], 2.5), np.percentile(a[1:], 97.5)
            for k in ["mae", "under", "over", "cover", "genre_reproduced", "genre_detected", "genre_reversed",
                      "genre_biased", "genre_mae", "genre_cover", "lpair_reproduced", "lpair_detected", "lpair_reversed",
                      "lpair_biased", "lpair_mae", "lpair_cover"]:
                row[k] = avg(system, est, k)
            rows.append(row)
    RES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "llm-measurement.csv", index=False, float_format="%.5f")
    (RES / "llm-measurement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'llm-measurement.txt'} and llm-measurement.csv")


if __name__ == "__main__":
    main()
