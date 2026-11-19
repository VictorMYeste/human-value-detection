#!/usr/bin/env python3
"""
How large must the annotated sample be? (step 5.2, AE item 7)

Every estimator in the paper fits something on the annotated validation split: a threshold, a
Platt map, error rates, a slope, or the group shares themselves. Here the validation split
(523 documents) is replaced by random subsamples of 25, 50, 100 and 200 documents, every
estimator is refitted on each subsample, and the test estimates are compared with the test
reference, exactly as in measurement_analysis.py.

Subsamples: manifestos and news are drawn separately, so every subsample has at least two
manifestos (about 11% of the documents, as in the split) and at least two news documents per
language; the rest is allocated to languages in proportion to their size. REPS subsamples per
size give the point errors and the spread of the fitted thresholds and slopes. The first
COVER_REPS of them also get B_IN bootstrap resamples (news within language, manifestos pooled,
Rao-Wu rescaled as in the main analysis), paired with the first B_IN test resamples, which give
interval coverage and detection exactly as in sections C and H of measurement_analysis.py.
The full split is the last size (its intervals from the first B_IN rows of the main design).

All of it is done in one pass per size and run: the subsamples and their resamples are rows of
the validation weight matrix, so RowFits refits every rule on each of them at once.

Main system (feature ensemble), mean of its three runs. Run from approaches/moral-values.
Outputs: results/subsample-validation.txt and .csv.
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd

from analyze_provenance import VALUES, load_gold
from measurement_analysis import (ALPHA, MIN_SLOPE, RES, RNG_SEED, RowFits, adjust, bh, boot_p, doc_index,
                                  document_design, genre_reference_p, lpair_reference_p, slope, system_probs,
                                  system_seeds, wsum)

SYSTEM = "Feature ensemble"
SIZES = [25, 50, 100, 200, None]      # None = the whole validation split
REPS = 100                            # subsamples per size
COVER_REPS = 20                       # of which get an inner bootstrap (intervals, coverage, detection)
B_IN = 50                             # bootstrap resamples per covered subsample (enough for a standard error)
SEED = RNG_SEED + 2
RULES = ["CC@t*", "CC@prev", "PCC-Platt", "ACC@prev", "ACC-G@prev", "SC@prev",
         "Gold-only", "PPI@prev", "PPI-Platt", "Platt-G"]
ANNOT_RULES = {"Gold-only", "PPI@prev", "PPI-Platt", "Platt-G"}   # no slope (see measurement_analysis, C)


def draw(n: int, lang: np.ndarray, genre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of one subsample of about n validation documents."""
    man = np.flatnonzero(genre == "manifesto")
    m = max(2, round(n * len(man) / len(lang)))
    langs = np.unique(lang)
    news = [np.flatnonzero((lang == l) & (genre == "news")) for l in langs]
    size = np.array([len(x) for x in news])
    expected = (n - m) * size / size.sum()
    k = np.floor(expected).astype(int)
    rest = (n - m) - k.sum()
    if rest > 0:                                          # randomised rounding keeps the expected allocation
        frac = expected - k
        k[rng.choice(len(langs), rest, replace=False, p=frac / frac.sum())] += 1
    k = np.minimum(np.maximum(k, 2), size)
    parts = [rng.choice(man, m, replace=False)] + [rng.choice(x, kk, replace=False) for x, kk in zip(news, k)]
    return np.sort(np.concatenate(parts))


def inner_bootstrap(idx: np.ndarray, lang: np.ndarray, genre: np.ndarray, n_docs: int,
                    rng: np.random.Generator) -> np.ndarray:
    """B_IN rescaling-bootstrap rows of one subsample: news within language, manifestos pooled."""
    stratum = np.where(genre[idx] == "manifesto", "manifesto", lang[idx])
    w = np.zeros((B_IN, n_docs))
    for s in np.unique(stratum):
        members = idx[stratum == s]
        n = len(members)
        draws = rng.integers(0, n, size=(B_IN, n - 1))
        counts = np.zeros((B_IN, n))
        np.add.at(counts, (np.repeat(np.arange(B_IN), n - 1), draws.ravel()), 1)
        w[:, members] = counts * n / (n - 1)
    return w


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    V, T = document_design(gv, seed=RNG_SEED + 1), document_design(gt)
    langs = sorted(set(T["lang"]))
    pairs = list(combinations(range(len(langs)), 2))
    lang_v, genre_v, lang_t, genre_t = V["lang"], V["genre"], T["lang"], T["genre"]
    Sv, St, nv, nt = V["S"], T["S"], V["nsent"], T["nsent"]
    doc_v, doc_t = doc_index(Sv), doc_index(St)
    groups_v = {"man": genre_v == "manifesto", "news": genre_v == "news", **{l: lang_v == l for l in langs}}
    groups_t = {"man": genre_t == "manifesto", "news": genre_t == "news", **{l: lang_t == l for l in langs}}
    all_v, all_t = np.ones(len(lang_v), bool), np.ones(len(lang_t), bool)
    gold_v_doc, gold_t_doc = Sv @ yv, St @ yt
    pos_v, neg_v = Sv @ yv, Sv @ (1 - yv)

    def level(Wr: np.ndarray, cdoc: np.ndarray, masks: dict, allm: np.ndarray, ns: np.ndarray) -> dict:
        prev = lambda m: wsum(Wr, cdoc, m) / (Wr[:, m] @ ns[m])[:, None]
        return {"all": prev(allm), "man": prev(masks["man"]), "news": prev(masks["news"]),
                "lang": np.stack([prev(masks[l]) for l in langs], axis=1)}

    def derive(gl: dict) -> dict:
        lp = gl["lang"]
        return {**gl, "genre": gl["man"] - gl["news"],
                "lpair": np.stack([lp[:, i] - lp[:, j] for i, j in pairs], axis=1).reshape(len(lp), -1)}

    # reference significance, as in the main analysis (all B test rows)
    gold_full = derive(level(T["W"], gold_t_doc, groups_t, all_t, nt))
    lang_masks_t = [groups_t[l] for l in langs]
    gsig = {"genre": bh(genre_reference_p(gold_full["genre"], gold_t_doc, nt, groups_t["man"], groups_t["news"])) < ALPHA,
            "lpair": bh(lpair_reference_p(gold_full["lpair"], gold_t_doc, nt, lang_masks_t, pairs)) < ALPHA}

    records, spread, sizes_seen = [], [], {}
    for n in SIZES:
        label = "all" if n is None else str(n)
        if n is None:
            subs = [np.arange(len(lang_v))]
            Wv = np.vstack([np.ones((1, len(lang_v))), V["W"][1:B_IN + 1]])
            tmap = np.arange(B_IN + 1)
            point_rows, cover = np.array([0]), [(0, np.arange(1, B_IN + 1))]
        else:
            subs = [draw(n, lang_v, genre_v, rng) for _ in range(REPS)]
            ind = np.zeros((REPS, len(lang_v)))
            for r, idx in enumerate(subs):
                ind[r, idx] = 1
            inner = [inner_bootstrap(subs[c], lang_v, genre_v, len(lang_v), rng) for c in range(COVER_REPS)]
            Wv = np.vstack([ind] + inner)
            tmap = np.concatenate([np.zeros(REPS, int)] + [np.arange(1, B_IN + 1)] * COVER_REPS)
            point_rows = np.arange(REPS)
            cover = [(c, REPS + c * B_IN + np.arange(B_IN)) for c in range(COVER_REPS)]
        sizes_seen[label] = (np.mean([len(s) for s in subs]), np.mean([nv[s].sum() for s in subs]))
        Wt = T["W"][tmap]
        design = {"W": Wv, "S": Sv}
        gold = derive(level(Wt, gold_t_doc, groups_t, all_t, nt))
        gold_val = level(Wv, gold_v_doc, groups_v, all_v, nv)

        def val_lpairs(c: np.ndarray) -> np.ndarray:
            return derive(level(Wv, c, groups_v, all_v, nv))

        per_run = {r: [] for r in RULES}
        fitted = {"t*": [], "t_prev": [], "beta_lang": [], "beta_genre": []}
        for seed in system_seeds(SYSTEM):
            pv, pt = system_probs(SYSTEM, seed, "val"), system_probs(SYSTEM, seed, "test")
            rf = RowFits(pv, yv, design)
            gl = {r: level(Wt, rf.counts(r, pt, St), groups_t, all_t, nt) for r in ["CC@t*", "CC@prev", "PCC-Platt"]}
            for r in ["CC@t*", "CC@prev", "PCC-Platt"]:
                per_run[r].append(derive(gl[r]))
            tp, fp = rf.counts("CC@prev", pv, Sv, weight=yv), rf.counts("CC@prev", pv, Sv, weight=1 - yv)

            def rates(mask: np.ndarray):
                with np.errstate(invalid="ignore", divide="ignore"):
                    return wsum(Wv, tp, mask) / wsum(Wv, pos_v, mask), wsum(Wv, fp, mask) / wsum(Wv, neg_v, mask)

            pooled = rates(all_v)
            cc = gl["CC@prev"]
            per_run["ACC@prev"].append(derive({g: adjust(cc[g], *pooled) if g != "lang" else
                                               adjust(cc[g], pooled[0][:, None], pooled[1][:, None]) for g in cc}))
            by_lang = [rates(groups_v[l]) for l in langs]
            per_run["ACC-G@prev"].append(derive({
                "all": adjust(cc["all"], *pooled), "man": adjust(cc["man"], *rates(groups_v["man"])),
                "news": adjust(cc["news"], *rates(groups_v["news"])),
                "lang": adjust(cc["lang"], np.stack([x[0] for x in by_lang], 1), np.stack([x[1] for x in by_lang], 1))}))
            # SC: slopes measured on the (sub)sample's own validation contrasts
            pred_v = tp + fp
            ev, gvv = val_lpairs(pred_v), derive(gold_val)
            b_lang, b_genre = slope(gvv["lpair"], ev["lpair"]), slope(gvv["genre"], ev["genre"])
            fix = lambda b: np.where(np.isfinite(b) & (b > MIN_SLOPE), b, 1.0)
            a = cc["all"]
            per_run["SC@prev"].append(derive({
                "all": a, "man": np.clip(a + (cc["man"] - a) / fix(b_genre)[:, None], 0, 1),
                "news": np.clip(a + (cc["news"] - a) / fix(b_genre)[:, None], 0, 1),
                "lang": np.clip(a[:, None] + (cc["lang"] - a[:, None]) / fix(b_lang)[:, None, None], 0, 1)}))
            fitted["t*"].append(rf.tstar); fitted["t_prev"].append(rf.tprev)
            fitted["beta_lang"].append(b_lang); fitted["beta_genre"].append(b_genre)
            # estimators that use each group's annotated documents
            per_run["Gold-only"].append(derive(dict(gold_val)))
            for name, base, pv_doc in [("PPI@prev", "CC@prev", pred_v), ("PPI-Platt", "PCC-Platt", rf.counts("PCC-Platt", pv, Sv))]:
                pl = level(Wv, pv_doc, groups_v, all_v, nv)
                per_run[name].append(derive({g: gl[base][g] + gold_val[g] - pl[g] for g in pl}))

            def platt_group(g: str) -> np.ndarray:
                rfg = RowFits(pv, yv, design, sent_mask=groups_v[g][doc_v], platt_only=True)
                cols = np.flatnonzero(groups_t[g][doc_t])
                c = rfg.counts("PCC-Platt", pt[cols], St[:, cols])
                return wsum(Wt, c, groups_t[g]) / (Wt[:, groups_t[g]] @ nt[groups_t[g]])[:, None]

            per_run["Platt-G"].append(derive({"all": gl["PCC-Platt"]["all"], "man": platt_group("man"),
                                              "news": platt_group("news"),
                                              "lang": np.stack([platt_group(l) for l in langs], axis=1)}))
            del rf, gl, tp, fp, pred_v
            print(f"  size {label:>3s}, seed {seed}: {time.time() - t0:5.0f} s", flush=True)

        # fitted quantities: spread over subsamples (run mean of each subsample's fit)
        fq = {k: np.mean(v, axis=0)[point_rows] for k, v in fitted.items()}
        spread.append({"size": label, "t*_sd": fq["t*"].std(), "t*_range": f"{fq['t*'].min():.2f}-{fq['t*'].max():.2f}",
                       "t_prev_sd": fq["t_prev"].std(0).mean(),
                       "beta_lang_median": np.median(fq["beta_lang"]), "beta_lang_sd": fq["beta_lang"].std(),
                       "beta_genre_median": np.median(fq["beta_genre"]), "beta_genre_sd": fq["beta_genre"].std()})

        for r in RULES:
            est = {k: np.mean([v[k] for v in per_run[r]], axis=0) for k in ["all", "lang", "genre", "lpair"]}
            g0 = {k: gold[k][0] for k in est}
            for i in point_rows:
                e = {k: est[k][i] for k in est}
                d_lp = e["lpair"] - g0["lpair"]
                lang_err = np.abs(e["lang"].sum(1) / g0["lang"].sum(1) - 1).mean() * 100
                rec = {"size": label, "rule": r, "rep": int(i), "total_err": abs(e["all"].sum() / g0["all"].sum() - 1) * 100,
                       "prev_mae": np.abs(e["all"] - g0["all"]).mean() * 100, "lang_total_err": lang_err,
                       "genre_mae": np.abs(e["genre"] - g0["genre"]).mean() * 100, "lpair_mae": np.abs(d_lp).mean() * 100,
                       "lpair_mae_refsig": np.abs(d_lp[gsig["lpair"]]).mean() * 100}
                if r not in ANNOT_RULES:
                    rec["genre_slope"] = float((g0["genre"] * e["genre"]).sum() / (g0["genre"] ** 2).sum())
                    rec["lpair_slope"] = float((g0["lpair"] * e["lpair"]).sum() / (g0["lpair"] ** 2).sum())
                records.append(rec)
            for c, inner_rows in cover:
                rows_ = np.concatenate([[c], inner_rows])
                rec = {"size": label, "rule": r, "rep": int(c), "covered": True}
                for k in ["all", "genre", "lpair"]:
                    e_r, g_r = est[k][rows_], gold[k][rows_]
                    rec[f"{k}_cover"] = (boot_p(e_r - g_r) >= ALPHA).mean() * 100
                    if k != "all":
                        same = np.sign(e_r[0]) == np.sign(g_r[0])
                        rec[f"{k}_detected"] = ((boot_p(e_r) < ALPHA) & same)[gsig[k]].mean() * 100
                records.append(rec)

    df = pd.DataFrame(records)
    point = df[df["covered"] != True].drop(columns=["covered"])
    cov = df[df["covered"] == True]
    order = [("all" if n is None else str(n)) for n in SIZES]
    summary = []
    for label in order:
        for r in RULES:
            p, c = point[(point["size"] == label) & (point["rule"] == r)], cov[(cov["size"] == label) & (cov["rule"] == r)]
            row = {"size": label, "rule": r, "docs": sizes_seen[label][0], "sentences": sizes_seen[label][1]}
            for m in ["total_err", "prev_mae", "lang_total_err", "genre_mae", "lpair_mae", "lpair_mae_refsig",
                      "genre_slope", "lpair_slope"]:
                if m in p and p[m].notna().any():
                    row[m], row[f"{m}_p10"], row[f"{m}_p90"] = p[m].median(), p[m].quantile(0.1), p[m].quantile(0.9)
            for m in ["all_cover", "genre_cover", "lpair_cover", "genre_detected", "lpair_detected"]:
                row[m] = c[m].mean()
            summary.append(row)
    summ = pd.DataFrame(summary)
    RES.mkdir(exist_ok=True)
    summ.to_csv(RES / "subsample-validation.csv", index=False)
    pd.DataFrame(spread).to_csv(RES / "subsample-validation-fits.csv", index=False)

    lines = []
    out = lines.append
    out("HOW LARGE MUST THE ANNOTATED SAMPLE BE? -- validation subsamples, every rule refitted on each")
    out(f"System: {SYSTEM}, mean of its runs. {REPS} subsamples per size (median and 10-90% range over them);")
    out(f"{COVER_REPS} of them with {B_IN} bootstrap resamples each for coverage and detection (mean over them).")
    out("Sizes (documents, sentences, mean over subsamples): "
        + "; ".join(f"{k}: {v[0]:.0f}, {v[1]:.0f}" for k, v in sizes_seen.items()))
    out("Total err: |estimated / reference value mentions - 1|, % (corpus); lang err: the same per language, mean of nine.")
    out("MAE: pp; lpair sig.: MAE of the 126 reference-significant language contrasts.")
    out("Cover: share whose paired error is within sampling noise (p >= 0.05), %; detected: share of the reference-")
    out("significant contrasts that are significant with the right sign (single test), %.")
    out("Slopes are not shown for the estimators that use each group's annotated documents (see measurement-analysis C).")
    out("=" * 132)
    fmt = lambda row, m, d=2: (f"{row[m]:.{d}f} [{row[m + '_p10']:.{d}f}-{row[m + '_p90']:.{d}f}]"
                               if m in row and pd.notna(row.get(m)) else "--")
    for label in order:
        out(f"\nsize {label} ({sizes_seen[label][0]:.0f} documents)")
        out(f"  {'rule':11s} {'total err %':>17s} {'lang err %':>18s} {'genre MAE':>17s} {'lpair MAE':>17s} "
            f"{'lpair sig.':>17s} {'lpair slope':>17s} {'cover tot/gen/lp':>17s} {'det. gen/lp':>12s}")
        for _, row in summ[summ["size"] == label].iterrows():
            out(f"  {row['rule']:11s} {fmt(row, 'total_err', 1):>17s} {fmt(row, 'lang_total_err', 1):>18s} "
                f"{fmt(row, 'genre_mae'):>17s} {fmt(row, 'lpair_mae'):>17s} {fmt(row, 'lpair_mae_refsig'):>17s} "
                f"{fmt(row, 'lpair_slope'):>17s} "
                f"{row['all_cover']:5.0f}/{row['genre_cover']:3.0f}/{row['lpair_cover']:3.0f}     "
                f"{row['genre_detected']:4.0f}/{row['lpair_detected']:3.0f}")
    out("\nFitted quantities, spread over subsamples (run mean of each subsample's fit):")
    out(f"  {'size':5s} {'t* sd':>6s} {'t* range':>10s} {'t_prev sd':>9s} {'lang slope':>16s} {'genre slope':>16s}")
    for s in spread:
        out(f"  {s['size']:5s} {s['t*_sd']:6.3f} {s['t*_range']:>10s} {s['t_prev_sd']:9.3f} "
            f"{s['beta_lang_median']:6.2f} (sd {s['beta_lang_sd']:.2f}) {s['beta_genre_median']:6.2f} (sd {s['beta_genre_sd']:.2f})")
    text = "\n".join(lines)
    (RES / "subsample-validation.txt").write_text(text + "\n")
    print(text)
    print(f"\nRuntime {time.time() - t0:.0f} s.")      # not in the file, so reruns reproduce it exactly


if __name__ == "__main__":
    main()
