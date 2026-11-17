#!/usr/bin/env python3
"""
Measurement analysis: does the decision rule change what a social scientist would conclude?

In computational social science, value classifiers are used to *measure* how often
values are invoked and to *compare* groups of texts, not to maximise macro-F1. This
script asks how well each decision rule recovers the gold-standard answers to four
such questions on the ValueEval'24 English test split:

  A. corpus-level prevalence of each of the 19 values;
  B. the manifesto-vs-news contrast for each value;
  C. pairwise contrasts between the nine source languages for each value;
  D. document-level value shares (one score per document and value).

Decision rules, all fitted on validation and applied unchanged to test. Every threshold
is chosen on the same grid (0.01-0.99) for every system:
  CC@0.5       classify and count at the default threshold;
  CC@t*        classify and count at the global threshold that maximises macro-F1;
  CC@label-t*  classify and count at per-label thresholds that maximise each label's F1 (SCut);
  CC@prev      classify and count at per-label thresholds that reproduce validation prevalence;
  PCC          average the raw predicted probabilities;
  PCC-Platt    average the probabilities after per-label Platt scaling fitted on validation.

Standard quantification estimators (Forman 2008; Bella et al. 2010; Saerens et al. 2002),
applied to every group (all, manifestos, news, each language) and, where defined, document:
  ACC@t*       adjusted classify and count, (CC - fpr) / (tpr - fpr), at the macro-F1 threshold,
               with tpr and fpr estimated on validation;
  ACC@prev     the same adjustment at the prevalence-matched thresholds;
  PACC         probabilistic adjusted count on the raw probabilities;
  EM           maximum-likelihood prior under prior shift (the fixed point of Saerens et al.'s
               EM), computed per group from the Platt-calibrated probabilities.
Adjusted estimates are clipped to [0, 1]; where tpr - fpr <= 0 the adjustment falls back to CC.

Corrections for group comparisons (step 1.3b: do they undo the attenuation of contrasts?),
both built on CC@prev:
  ACC-G@prev   adjusted count with group-specific error rates: rates estimated on the
               validation documents of each language (language contrasts) or genre (genre
               contrasts); pooled rates for the corpus total;
  RC@prev      regression calibration of contrasts: group deviations from the corpus total are
               divided by the attenuation slope measured on validation (language-pair contrasts
               for languages, manifesto-news contrasts for genre).

Systems:
  Feature ensemble  soft vote of DeBERTa-base Baseline + LIWC-22 + Prev-2, rebuilt within
                    each seed (the paper's main system);
  DeBERTa           text-only DeBERTa-base;
  RoBERTa           text-only RoBERTa-base;
  Seed ensemble     soft vote of the three text-only DeBERTa seeds. It is a single run and a
                    control: it separates the effect of averaging models from that of the features.
The other three systems have one run per seed (42, 7, 1701).

Uncertainty: stratified cluster bootstrap over test *documents* (sentences of one
document are not independent), resampling documents within each source-language x
genre stratum, B = 2000. The same resamples are used for every system, seed and rule,
so differences between rules are paired. Point estimates are means over a system's
runs; intervals are 95% bootstrap percentile intervals of that mean. Contrasts
are tested with bootstrap p-values and Benjamini-Hochberg correction within each
family (19 genre contrasts; 36 x 19 = 684 language-pair contrasts).
Estimators that divide by quantities estimated on validation (the error rates of ACC,
PACC and ACC-G; the attenuation slopes of RC) also propagate validation sampling error:
validation documents are resampled in the same way (independent resamples, paired by
index). Thresholds and the Platt maps are held fixed, as for every other rule.

Usage:
    python3 measurement_analysis.py
Writes results/measurement-analysis.txt and four CSVs (results/measurement-*.csv; measurement-summary.csv
holds the summary metrics per system and rule, with 95% bootstrap intervals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

from analyze_provenance import VALUES, KEY, load_gold

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
RES = HERE / "results"

SEEDS = [42, 7, 1701]
B = 2000
RNG_SEED = 20261116
THRESHOLD_GRID = np.round(np.arange(1, 100) * 0.01, 2)   # one grid for every system and rule
ENS_MEMBERS = ["Baseline", "Lex-LIWC-22", "Previous-Sentences-2"]
SYSTEMS = ["Feature ensemble", "DeBERTa", "RoBERTa", "Seed ensemble"]
CLASSIFY_RULES = ["CC@0.5", "CC@t*", "CC@label-t*", "CC@prev", "PCC", "PCC-Platt"]
QUANT_RULES = ["ACC@t*", "ACC@prev", "PACC", "EM"]
GROUP_RULES = ["ACC-G@prev", "RC@prev"]
RULES = CLASSIFY_RULES + QUANT_RULES + GROUP_RULES
CC_RULES = ["CC@0.5", "CC@t*", "CC@label-t*", "CC@prev"]
EM_LOGIT_CLIP = 12.0  # EM works on per-group histograms of calibrated logits in [-12, 12] ...
EM_BIN_WIDTH = 0.1    # ... with bins of 0.1 logit, using each bin's mean probability
EM_STEPS = 40         # bisection steps for the maximum-likelihood prior (precision ~1e-12)
MIN_SLOPE = 0.05      # RC falls back to no correction if the validation slope is below this
RARE = 0.02          # values below 2% gold test prevalence count as rare
MIN_DOC_SENTS = 10   # documents shorter than this are excluded from the document-level analysis
ALPHA = 0.05


# ----------------------------------------------------------------------------- loading
def _tag(seed: int, roberta: bool) -> str:
    if roberta:
        return f"-s{seed}"                 # RoBERTa artefacts are tagged for every seed
    return "" if seed == 42 else f"-s{seed}"  # "" == seed 42 (published artefacts)


def load_probs(name: str, seed: int, split: str, roberta: bool = False) -> np.ndarray:
    gold = load_gold("validation" if split == "val" else "test")
    pred = pd.read_csv(OUT / f"{name}{_tag(seed, roberta)}-{split}.tsv", sep="\t")[KEY + VALUES]
    merged = gold[KEY].merge(pred, on=KEY, how="left", validate="1:1")
    if merged[VALUES].isna().any().any():
        raise ValueError(f"{name} seed {seed} {split}: predictions missing for some sentences")
    return merged[VALUES].to_numpy(dtype=float)


def system_seeds(system: str) -> list[int | None]:
    """Runs of a system: one per seed, except the seed ensemble, which is a single run (None)."""
    return [None] if system == "Seed ensemble" else list(SEEDS)


def system_probs(system: str, seed: int | None, split: str) -> np.ndarray:
    if system == "DeBERTa":
        return load_probs("Baseline", seed, split)
    if system == "Feature ensemble":
        return np.mean([load_probs(m, seed, split) for m in ENS_MEMBERS], axis=0)
    if system == "RoBERTa":
        return load_probs("RoBERTa-Baseline", seed, split, roberta=True)
    if system == "Seed ensemble":
        return np.mean([load_probs("Baseline", s, split) for s in SEEDS], axis=0)
    raise KeyError(system)


# ----------------------------------------------------------------------------- rules
def macro_f1(y: np.ndarray, pb: np.ndarray) -> float:
    tp = (y & pb).sum(0); fp = ((1 - y) & pb).sum(0); fn = (y & (1 - pb)).sum(0)
    den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros(len(tp), float), where=den > 0).mean())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def global_threshold(pv: np.ndarray, yv: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> float:
    """The single threshold that maximises validation macro-F1 (ties -> lowest)."""
    scores = [macro_f1(yv, (pv >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def label_f1_thresholds(pv: np.ndarray, yv: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """Per-label thresholds that maximise each label's validation F1 (SCut, Yang 2001; ties -> lowest)."""
    pos = yv.astype(bool)[:, :, None]
    pb = pv[:, :, None] >= grid[None, None, :]
    tp = (pb & pos).sum(0)
    fp = (pb & ~pos).sum(0)
    fn = pos.sum(0) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(den.shape), where=den > 0)
    return grid[np.argmax(f1, axis=1)]


def prevalence_thresholds(pv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Per-label thresholds that reproduce each label's validation prevalence (PCut-style)."""
    prev = yv.mean(0)
    return np.array([np.quantile(pv[:, j], 1 - prev[j]) for j in range(pv.shape[1])])


def fit_rules(pv: np.ndarray, yv: np.ndarray) -> dict:
    platt = np.zeros((len(VALUES), 2))
    for j in range(len(VALUES)):
        lr = LogisticRegression(C=1e4, max_iter=2000).fit(logit(pv[:, j]).reshape(-1, 1), yv[:, j])
        platt[j] = lr.coef_[0, 0], lr.intercept_[0]
    return {"tstar": global_threshold(pv, yv), "tlabel": label_f1_thresholds(pv, yv),
            "tprev": prevalence_thresholds(pv, yv), "platt": platt}


def apply_rules(p: np.ndarray, fit: dict) -> dict[str, np.ndarray]:
    calibrated = 1 / (1 + np.exp(-(logit(p) * fit["platt"][:, 0] + fit["platt"][:, 1])))
    return {
        "CC@0.5": (p >= 0.5).astype(float),
        "CC@t*": (p >= fit["tstar"]).astype(float),
        "CC@label-t*": (p >= fit["tlabel"]).astype(float),
        "CC@prev": (p >= fit["tprev"]).astype(float),
        "PCC": p,
        "PCC-Platt": calibrated,
    }


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error pooled over all sentence-value pairs (equal-width bins)."""
    p, y = p.ravel(), y.ravel()
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return err


# ----------------------------------------------------------------------------- statistics
def bootstrap_weights(stratum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row 0 is the original sample; rows 1..B resample documents within strata."""
    w = np.zeros((B + 1, len(stratum)))
    w[0] = 1
    for s in np.unique(stratum):
        idx = np.where(stratum == s)[0]
        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        counts = np.zeros((B, len(idx)))
        np.add.at(counts, (np.repeat(np.arange(B), len(idx)), draws.ravel()), 1)
        w[1:, idx] = counts
    return w


def boot_p(d: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p-value that a contrast differs from zero (rows 1..B)."""
    r = d[1:]
    le, ge = (r <= 0).sum(0), (r >= 0).sum(0)
    return np.minimum(1.0, 2 * np.minimum(le + 1, ge + 1) / (B + 1))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1.0)
    return out


def slope(g: np.ndarray, e: np.ndarray) -> np.ndarray:
    """OLS slope of estimated on gold contrasts, per bootstrap row (1 = no attenuation)."""
    gc = g - g.mean(1, keepdims=True); ec = e - e.mean(1, keepdims=True)
    return (gc * ec).sum(1) / (gc ** 2).sum(1)


def rank_corr(g: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Spearman correlation across groups (axis 1), averaged over values; per bootstrap row."""
    rg, re = rankdata(g, axis=1), rankdata(e, axis=1)
    rg = rg - rg.mean(1, keepdims=True); re = re - re.mean(1, keepdims=True)
    den = np.sqrt((rg ** 2).sum(1) * (re ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rg * re).sum(1) / den
    return np.nanmean(rho, axis=1)


def ci(x: np.ndarray) -> str:
    lo, hi = np.percentile(x[1:], [2.5, 97.5])
    return f"{x[0]:.2f} [{lo:.2f}, {hi:.2f}]"


def mean_sd(x, nd: int = 3) -> str:
    """Mean ± sd over runs; just the value for a single-run system."""
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f}" if x.size == 1 else f"{x.mean():.{nd}f}±{x.std(ddof=1):.{nd}f}"


def document_design(gt: pd.DataFrame, seed: int = RNG_SEED) -> dict:
    """Documents of a split, their strata and the bootstrap weights shared by every analysis script.

    S maps sentences to documents (docs x sentences); W holds the original sample (row 0)
    and B stratified document resamples. The fixed RNG seed makes W identical across scripts,
    so results from different scripts are paired. The validation split uses its own seed.
    """
    codes, docs = pd.factorize(gt["Text-ID"])
    n_docs, n_sent = len(docs), len(gt)
    S = csr_matrix((np.ones(n_sent), (codes, np.arange(n_sent))), shape=(n_docs, n_sent))
    lang = np.array([d.split("_")[0] for d in docs])
    genre = np.array(["manifesto" if "_M_" in d else "news" for d in docs])
    W = bootstrap_weights(np.char.add(np.char.add(lang, "|"), genre), np.random.default_rng(seed))
    return {"docs": docs, "S": S, "nsent": np.asarray(S.sum(1)).ravel(), "lang": lang, "genre": genre, "W": W}


# ----------------------------------------------------------------------------- quantification
def boot_rates(Sv, Wv: np.ndarray, yv: np.ndarray, pred_v: np.ndarray, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Validation true- and false-positive rates per bootstrap row, (B+1, labels) each.

    pred_v holds 0/1 decisions (adjusted count) or probabilities (probabilistic adjusted
    count); mask restricts the estimate to the validation documents of one group.
    """
    m = np.ones(Sv.shape[0], bool) if mask is None else mask
    Sm, Wm = Sv[m], Wv[:, m]
    tp = Wm @ np.asarray(Sm @ (yv * pred_v))
    fp = Wm @ np.asarray(Sm @ ((1 - yv) * pred_v))
    pos = Wm @ np.asarray(Sm @ yv)
    neg = Wm @ np.asarray(Sm @ (1 - yv))
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / pos, fp / neg


def adjust(cc: np.ndarray, tpr: np.ndarray, fpr: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """(cc - fpr) / (tpr - fpr), clipped to [0, 1]; falls back to cc where tpr - fpr is not
    positive or not finite. Tallies fallbacks and clipping on the original sample (row 0)."""
    den = tpr - fpr
    good = np.isfinite(den) & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(good, (cc - fpr) / np.where(good, den, 1.0), cc)
    if stats is not None:
        g0 = np.broadcast_to(good, raw.shape)[0]
        stats["cells"] += g0.size
        stats["fallback"] += int((~g0).sum())
        stats["clip0"] += int((raw[0] < 0).sum())
        stats["clip1"] += int((raw[0] > 1).sum())
    return np.clip(raw, 0.0, 1.0)


def em_prior_exact(pc: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Maximum-likelihood prevalence per label under prior shift, over individual sentences.

    The log-likelihood sum_i log(pi * p_i / prior + (1 - pi) * (1 - p_i) / (1 - prior)) is
    concave in pi, so its derivative is decreasing and bisection finds the maximum. This is
    the fixed point of the EM procedure of Saerens et al. (2002). Used to check the binned version.
    """
    a = pc / prior - (1 - pc) / (1 - prior)
    b = (1 - pc) / (1 - prior)
    lo, hi = np.zeros(pc.shape[1]), np.ones(pc.shape[1])
    d = lambda x: (a / (b + x * a)).sum(0)
    for _ in range(EM_STEPS):
        mid = (lo + hi) / 2
        up = d(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    est = (lo + hi) / 2
    return np.where(d(np.zeros_like(est)) <= 0, 0.0, np.where(d(np.ones_like(est)) >= 0, 1.0, est))


class EMGroups:
    """Bootstrapped maximum-likelihood prevalence for any group of test documents.

    Calibrated probabilities are binned on the logit scale per label; per-document bin
    counts and probability sums make every bootstrap row a weighted histogram, on which
    the concave log-likelihood is maximised by bisection (vectorised over rows and labels).
    """

    def __init__(self, S, W: np.ndarray, pc: np.ndarray, prior: np.ndarray):
        n, k = pc.shape
        self.W, self.prior, self.k = W, prior, k
        self.nb = int(round(2 * EM_LOGIT_CLIP / EM_BIN_WIDTH))
        z = np.clip(logit(pc), -EM_LOGIT_CLIP, EM_LOGIT_CLIP)
        idx = np.minimum(((z + EM_LOGIT_CLIP) / EM_BIN_WIDTH).astype(int), self.nb - 1)
        cols = (np.arange(k)[None, :] * self.nb + idx).ravel()
        rws = np.repeat(np.arange(n), k)
        shape = (n, k * self.nb)
        self.H = np.asarray((S @ csr_matrix((np.ones(n * k), (rws, cols)), shape=shape)).todense())
        self.P = np.asarray((S @ csr_matrix((pc.ravel(), (rws, cols)), shape=shape)).todense())
        centres = -EM_LOGIT_CLIP + EM_BIN_WIDTH * (np.arange(self.nb) + 0.5)
        self.centre_p = 1 / (1 + np.exp(-centres))

    def prevalence(self, mask: np.ndarray) -> np.ndarray:
        rows = self.W.shape[0]
        h = (self.W[:, mask] @ self.H[mask]).reshape(rows, self.k, self.nb)
        ps = (self.W[:, mask] @ self.P[mask]).reshape(rows, self.k, self.nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            pbar = np.where(h > 0, ps / h, self.centre_p[None, None, :])
        prior = self.prior[None, :, None]
        a = pbar / prior - (1 - pbar) / (1 - prior)
        b = (1 - pbar) / (1 - prior)
        d = lambda x: (h * a / (b + x[:, :, None] * a)).sum(2)
        lo, hi = np.zeros((rows, self.k)), np.ones((rows, self.k))
        for _ in range(EM_STEPS):
            mid = (lo + hi) / 2
            up = d(mid) > 0
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        est = (lo + hi) / 2
        est = np.where(d(np.zeros_like(est)) <= 0, 0.0, est)
        return np.where(d(np.ones_like(est)) >= 0, 1.0, est)


# ----------------------------------------------------------------------------- main
def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    design = document_design(gt)
    docs, S, nsent, lang, genre, W = (design[k] for k in ["docs", "S", "nsent", "lang", "genre", "W"])
    n_docs, n_sent_total = len(docs), len(gt)
    langs = sorted(set(lang))
    pairs = [(i, j) for i in range(len(langs)) for j in range(i + 1, len(langs))]
    allm, man, news = np.ones(n_docs, bool), genre == "manifesto", genre == "news"
    long_docs = nsent >= MIN_DOC_SENTS

    lang_masks = [lang == l for l in langs]
    doc_lang = np.array([langs.index(l) for l in lang[long_docs]])

    def group_prev(cdoc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return (W[:, mask] @ cdoc[mask]) / (W[:, mask] @ nsent[mask])[:, None]

    def group_level(cdoc: np.ndarray) -> dict[str, np.ndarray]:
        """Group prevalences per bootstrap row from per-document counts (or probability sums)."""
        return {
            "all": group_prev(cdoc, allm), "man": group_prev(cdoc, man), "news": group_prev(cdoc, news),
            "lang": np.stack([group_prev(cdoc, m) for m in lang_masks], axis=1),        # (B+1, 9, 19)
            "doc": (cdoc / nsent[:, None])[long_docs],
        }

    def derive(gl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The quantities every section evaluates: totals, contrasts, document shares."""
        lp = gl["lang"]
        return {
            "all": gl["all"],
            "genre": gl["man"] - gl["news"],
            "lang": lp,
            "lpair": np.stack([lp[:, i] - lp[:, j] for i, j in pairs], axis=1).reshape(B + 1, -1),
            "doc": gl["doc"],
        }

    def views(cdoc: np.ndarray) -> dict[str, np.ndarray]:
        return derive(group_level(cdoc))

    # ---- validation design for the estimators that use validation error rates or slopes
    vdesign = document_design(gv, seed=RNG_SEED + 1)
    Sv, Wv, nsent_v, lang_v, genre_v = (vdesign[k] for k in ["S", "W", "nsent", "lang", "genre"])
    man_v, news_v = genre_v == "manifesto", genre_v == "news"
    lang_masks_v = [lang_v == l for l in langs]

    def val_group_prev(cdoc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return (Wv[:, mask] @ cdoc[mask]) / (Wv[:, mask] @ nsent_v[mask])[:, None]

    def val_slopes(cdoc_est: np.ndarray, cdoc_gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Attenuation slopes on validation: language-pair contrasts and manifesto-news contrasts."""
        def lpairs(c):
            lp = np.stack([val_group_prev(c, m) for m in lang_masks_v], axis=1)
            return np.stack([lp[:, i] - lp[:, j] for i, j in pairs], axis=1).reshape(B + 1, -1)
        genre_c = lambda c: val_group_prev(c, man_v) - val_group_prev(c, news_v)
        return slope(lpairs(cdoc_gold), lpairs(cdoc_est)), slope(genre_c(cdoc_gold), genre_c(cdoc_est))

    def adjusted(gl: dict, pooled: tuple, stats: dict, by_genre: tuple | None = None,
                 by_lang: tuple | None = None) -> dict[str, np.ndarray]:
        """Adjusted count for every group; pooled rates unless group-specific ones are given."""
        tp, fp = pooled
        g_man, g_news = by_genre if by_genre is not None else (pooled, pooled)
        g_lang = by_lang if by_lang is not None else (tp[:, None, :], fp[:, None, :])
        doc_rates = (by_lang[0][0][doc_lang], by_lang[1][0][doc_lang]) if by_lang is not None else (tp[0], fp[0])
        return derive({
            "all": adjust(gl["all"], tp, fp, stats),
            "man": adjust(gl["man"], *g_man, stats),
            "news": adjust(gl["news"], *g_news, stats),
            "lang": adjust(gl["lang"], *g_lang, stats),
            "doc": adjust(gl["doc"], *doc_rates),
        })

    def regression_calibrated(gl: dict, beta_lang: np.ndarray, beta_genre: np.ndarray, stats: dict) -> dict:
        """Divide group deviations from the corpus total by the validation attenuation slope."""
        fixed = {}
        for name, beta in [("lang", beta_lang), ("genre", beta_genre)]:
            good = np.isfinite(beta) & (beta > MIN_SLOPE)
            stats["fallback"] += int(~good[0]); stats["cells"] += 1
            fixed[name] = np.where(good, beta, 1.0)
        a = gl["all"]
        raw = {"man": a + (gl["man"] - a) / fixed["genre"][:, None],
               "news": a + (gl["news"] - a) / fixed["genre"][:, None],
               "lang": a[:, None, :] + (gl["lang"] - a[:, None, :]) / fixed["lang"][:, None, None]}
        for v in raw.values():
            stats["clip0"] += int((v[0] < 0).sum()); stats["clip1"] += int((v[0] > 1).sum())
        return derive({"all": a, **{k: np.clip(v, 0.0, 1.0) for k, v in raw.items()},
                       "doc": np.full_like(gl["doc"], np.nan)})

    gold = views(S @ yt)
    gold_v_doc = Sv @ yv
    rare = gold["all"][0] < RARE
    gsig = {k: bh(boot_p(gold[k])) < ALPHA for k in ["genre", "lpair"]}

    out("MEASUREMENT ANALYSIS -- ValueEval'24 English test split")
    out(f"{n_sent_total} sentences in {n_docs} documents ({man.sum()} manifestos, {news.sum()} news); "
        f"{len(langs)} source languages; stratified document bootstrap B = {B}")
    out(f"Rare values (< {RARE:.0%} gold prevalence): {rare.sum()} of 19")
    out()

    # ---- run every system x seed x rule
    rows = {s: {r: [] for r in RULES} for s in SYSTEMS}
    diag = {s: [] for s in SYSTEMS}
    qstats = {s: {r: {"cells": 0, "fallback": 0, "clip0": 0, "clip1": 0} for r in QUANT_RULES + GROUP_RULES}
              for s in SYSTEMS}
    em_check, slopes_v = {s: [] for s in SYSTEMS}, {s: [] for s in SYSTEMS}
    mechanism = {s: [] for s in SYSTEMS}
    man_sent = gt["Text-ID"].str.contains("_M_").to_numpy()

    def genre_rate_gap(dec: np.ndarray, positive: int) -> np.ndarray:
        """Manifesto minus news detection rate among gold positives (tpr) or gold negatives (fpr)."""
        rate = lambda m: (dec[m] * (yt[m] == positive)).sum(0) / np.maximum((yt[m] == positive).sum(0), 1)
        return rate(man_sent) - rate(~man_sent)

    for system in SYSTEMS:
        for seed in system_seeds(system):
            pv, pt = system_probs(system, seed, "val"), system_probs(system, seed, "test")
            fit = fit_rules(pv, yv)
            est = apply_rules(pt, fit)
            diag[system].append({
                "seed": seed if seed is not None else "+".join(map(str, SEEDS)), "tstar": fit["tstar"],
                **{f"F1 {r}": macro_f1(yt, est[r].astype(int)) for r in CC_RULES},
                "ECE raw": ece(pt, yt), "ECE Platt": ece(est["PCC-Platt"], yt),
                "CITL raw": pt.mean() / yt.mean(), "CITL Platt": est["PCC-Platt"].mean() / yt.mean(),
            })
            gl = {r: group_level(S @ est[r]) for r in CLASSIFY_RULES}
            for r in CLASSIFY_RULES:
                rows[system][r].append(derive(gl[r]))
            g0 = gold["genre"][0]
            gap_t, gap_f = genre_rate_gap(est["CC@prev"], 1), genre_rate_gap(est["CC@prev"], 0)
            mechanism[system].append((spearmanr(g0, gap_t)[0], spearmanr(g0, gap_f)[0],
                                      (np.sign(gap_f) == np.sign(g0)).mean()))

            # standard quantification estimators
            qs = qstats[system]
            dec_t = (pv >= fit["tstar"]).astype(int)
            dec_p = (pv >= fit["tprev"]).astype(int)
            rows[system]["ACC@t*"].append(adjusted(gl["CC@t*"], boot_rates(Sv, Wv, yv, dec_t), qs["ACC@t*"]))
            rates_prev = boot_rates(Sv, Wv, yv, dec_p)
            rows[system]["ACC@prev"].append(adjusted(gl["CC@prev"], rates_prev, qs["ACC@prev"]))
            rows[system]["PACC"].append(adjusted(gl["PCC"], boot_rates(Sv, Wv, yv, pv), qs["PACC"]))
            pc, prior = est["PCC-Platt"], yv.mean(0)
            em = EMGroups(S, W, pc, prior)
            em_gl = {"all": em.prevalence(allm), "man": em.prevalence(man), "news": em.prevalence(news),
                     "lang": np.stack([em.prevalence(m) for m in lang_masks], axis=1),
                     "doc": np.full_like(gl["PCC"]["doc"], np.nan)}
            for k in ["all", "man", "news", "lang"]:
                qs["EM"]["cells"] += em_gl[k][0].size
                qs["EM"]["clip0"] += int((em_gl[k][0] <= 0).sum()); qs["EM"]["clip1"] += int((em_gl[k][0] >= 1).sum())
            em_check[system].append(np.abs(em_gl["all"][0] - em_prior_exact(pc, prior)).max())
            rows[system]["EM"].append(derive(em_gl))

            # corrections for group comparisons (step 1.3b), on the prevalence-matched rule
            rows[system]["ACC-G@prev"].append(adjusted(
                gl["CC@prev"], rates_prev, qs["ACC-G@prev"],
                by_genre=(boot_rates(Sv, Wv, yv, dec_p, man_v), boot_rates(Sv, Wv, yv, dec_p, news_v)),
                by_lang=tuple(np.stack(x, axis=1) for x in zip(*[boot_rates(Sv, Wv, yv, dec_p, m)
                                                                  for m in lang_masks_v]))))
            beta_lang, beta_genre = val_slopes(Sv @ dec_p, gold_v_doc)
            slopes_v[system].append((beta_lang[0], beta_genre[0]))
            rows[system]["RC@prev"].append(regression_calibrated(gl["CC@prev"], beta_lang, beta_genre,
                                                                 qs["RC@prev"]))

    for system in SYSTEMS:
        for r in RULES:
            for v in rows[system][r]:
                for key in ["all", "genre", "lpair"]:
                    assert np.isfinite(v[key]).all(), f"{system} {r} {key}: non-finite estimates"

    # ---- sanity check against published numbers
    d42 = diag["DeBERTa"][0]; e42 = diag["Feature ensemble"][0]
    out(f"Sanity: DeBERTa seed 42 t*={d42['tstar']:.2f}, test macro-F1 {d42['F1 CC@t*']:.4f} (published 0.3115); "
        f"feature ensemble seed 42 t*={e42['tstar']:.2f}, {e42['F1 CC@t*']:.4f} (published 0.332)")
    out()

    def seed_mean(system: str, rule: str, fn) -> np.ndarray:
        return np.mean([fn(v) for v in rows[system][rule]], axis=0)

    summary: dict[tuple[str, str], dict] = {}  # system x rule -> metrics, written to measurement-summary.csv

    def put(system: str, rule: str, name: str, value, arr: np.ndarray | None = None) -> None:
        row = summary.setdefault((system, rule), {"system": system, "rule": rule, "runs": len(rows[system][rule])})
        row[name] = value
        if arr is not None:
            row[f"{name}_lo"], row[f"{name}_hi"] = np.percentile(arr[1:], [2.5, 97.5])

    # ---- A. corpus-level prevalence
    out("=" * 108)
    out("A. CORPUS-LEVEL PREVALENCE OF EACH VALUE")
    out("   MAE: mean absolute error over the 19 values, percentage points (point estimate, mean over runs; no interval,")
    out("        because percentile intervals of an absolute error near zero are biased upwards).")
    out("   Signed rel. error: mean of (est - gold) / gold over rare / frequent values, mean over runs, 95% bootstrap CI.")
    out(f"   Biased values: values whose (est - gold) differs from 0 (BH q < {ALPHA}), as under / over, mean over runs.")
    out("   Ratio: total estimated value mentions / gold mentions, mean over runs, 95% bootstrap CI.")
    out("=" * 108)
    srel = lambda m: (lambda v: ((v["all"] - gold["all"]) / gold["all"])[:, m].mean(1) * 100)
    ratio = lambda v: v["all"].sum(1) / gold["all"].sum(1)
    for system in SYSTEMS:
        out(f"\n{system}  ({len(rows[system][RULES[0]])} run(s))")
        out(f"  {'rule':12s} {'MAE (pp)':>9s} {'signed rel. err rare %':>26s} {'signed rel. err freq. %':>26s} {'biased under/over':>18s} {'ratio':>20s} {'macro-F1':>14s}")
        for r in RULES:
            mae_pt = np.mean([np.abs(v["all"][0] - gold["all"][0]).mean() * 100 for v in rows[system][r]])
            under, over = [], []
            for v in rows[system][r]:
                diff = v["all"] - gold["all"]
                sig = bh(boot_p(diff)) < ALPHA
                under.append((sig & (diff[0] < 0)).sum()); over.append((sig & (diff[0] > 0)).sum())
            f1 = [d[f"F1 {r}"] for d in diag[system]] if r in CC_RULES else None
            f1s = mean_sd(f1) if f1 else "--"
            out(f"  {r:12s} {mae_pt:9.2f} {ci(seed_mean(system, r, srel(rare))):>26s} {ci(seed_mean(system, r, srel(~rare))):>26s} "
                f"{f'{np.mean(under):.1f} / {np.mean(over):.1f}':>18s} {ci(seed_mean(system, r, ratio)):>20s} {f1s:>14s}")
            rat = seed_mean(system, r, ratio)
            put(system, r, "macro_f1", float(np.mean(f1)) if f1 else np.nan)
            put(system, r, "ratio", rat[0], rat)
            put(system, r, "prev_mae", mae_pt)
            put(system, r, "biased_under", float(np.mean(under)))
            put(system, r, "biased_over", float(np.mean(over)))

    # ---- B/C. contrasts
    def contrast_block(key: str, title: str, family: int) -> None:
        out()
        out("=" * 108)
        out(title)
        out(f"   Gold-significant contrasts (BH q < {ALPHA}): {gsig[key].sum()} of {family}.")
        out("   Sign ok: estimate points the same way as gold (gold-significant contrasts).")
        out("   Reproduced: estimate is itself significant with the right sign (gold-significant contrasts).")
        out(f"   Biased: estimated contrast differs from the gold contrast beyond sampling noise (paired, BH q < {ALPHA}),")
        out("           over ALL contrasts in the family -- i.e. differential measurement error between the groups.")
        out("   Slope: OLS of estimated on gold contrasts, all contrasts (1 = faithful, < 1 = differences shrunk).")
        out("   Percentages are per run, averaged over a system's runs.")
        out("=" * 108)
        gs = gsig[key]
        reversed_max = 0.0
        for system in SYSTEMS:
            out(f"\n{system}")
            hdr = f"  {'rule':12s} {'sign ok %':>10s} {'reproduced %':>13s} {'biased %':>9s} {'MAE (pp)':>9s} {'slope':>20s}"
            if key == "lpair":
                hdr += f" {'lang. rank rho':>20s}"
            out(hdr)
            for r in RULES:
                stats = []
                for v in rows[system][r]:
                    e0, g0 = v[key][0], gold[key][0]
                    esig = bh(boot_p(v[key])) < ALPHA
                    same = np.sign(e0) == np.sign(g0)
                    reversed_max = max(reversed_max, (esig & ~same)[gs].mean() * 100)
                    stats.append([
                        same[gs].mean() * 100,
                        (esig & same)[gs].mean() * 100,
                        (bh(boot_p(v[key] - gold[key])) < ALPHA).mean() * 100,
                        np.abs(e0 - g0).mean() * 100,
                    ])
                st = np.mean(stats, axis=0)
                sl = seed_mean(system, r, lambda v: slope(gold[key], v[key]))
                line = f"  {r:12s} {st[0]:10.1f} {st[1]:13.1f} {st[2]:9.1f} {st[3]:9.2f} {ci(sl):>20s}"
                for name, value in zip(["sign_ok", "reproduced", "biased", "mae"], st):
                    put(system, r, f"{key}_{name}", value)
                put(system, r, f"{key}_slope", sl[0], sl)
                if key == "lpair":
                    rho = seed_mean(system, r, lambda v: rank_corr(gold['lang'], v['lang']))
                    line += f" {ci(rho):>20s}"
                    put(system, r, "lang_rank_rho", rho[0], rho)
                out(line)
        out(f"\n   Largest share of gold-significant contrasts reproduced with the WRONG sign, any system/rule/run: {reversed_max:.1f}%")

    contrast_block("genre", "B. MANIFESTO vs NEWS CONTRAST PER VALUE (19 contrasts)", 19)
    out("\n   Gold-significant genre contrasts (manifesto - news, pp, 95% CI):")
    for j in np.where(gsig["genre"])[0]:
        lo, hi = np.percentile(gold["genre"][1:, j] * 100, [2.5, 97.5])
        out(f"     {VALUES[j]:28s} {gold['genre'][0, j] * 100:+6.2f} [{lo:+.2f}, {hi:+.2f}]")
    contrast_block("lpair", f"C. SOURCE-LANGUAGE PAIRWISE CONTRASTS ({len(pairs)} pairs x 19 values = {len(pairs) * 19})",
                   len(pairs) * 19)

    # ---- D. document level
    out()
    out("=" * 108)
    out(f"D. DOCUMENT-LEVEL VALUE SHARES (documents with >= {MIN_DOC_SENTS} sentences: {long_docs.sum()} of {n_docs})")
    out("   Pearson r between estimated and gold share of sentences expressing each value, averaged over values;")
    out("   MAE in percentage points; mean +- sd over runs.")
    out("=" * 108)
    for system in SYSTEMS:
        out(f"\n{system}")
        for r in RULES:
            if np.isnan(rows[system][r][0]["doc"]).all():
                out(f"  {r:12s} --  (group-level estimator; no document-level estimate)")
                continue
            rs, ms = [], []
            for v in rows[system][r]:
                e, g = v["doc"], gold["doc"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rs.append(np.nanmean([np.corrcoef(e[:, j], g[:, j])[0, 1] for j in range(len(VALUES))]))
                ms.append(np.abs(e - g).mean() * 100)
            out(f"  {r:12s} r = {mean_sd(rs)}   MAE = {mean_sd(ms, 2)} pp")

    # ---- E. calibration
    out()
    out("=" * 108)
    out("E. PROBABILITY CALIBRATION ON TEST (pooled over sentence-value pairs; mean over runs)")
    out("   ECE: expected calibration error, 15 equal-width bins. CITL: mean predicted probability / gold rate.")
    out("=" * 108)
    for system in SYSTEMS:
        d = pd.DataFrame(diag[system])
        out(f"  {system:17s} ECE raw {d['ECE raw'].mean():.4f}  ECE Platt {d['ECE Platt'].mean():.4f}   "
            f"CITL raw {d['CITL raw'].mean():.3f}  CITL Platt {d['CITL Platt'].mean():.3f}   "
            f"t* by run {', '.join(f'{t:.2f}' for t in d['tstar'])}")

    # ---- F. quantification diagnostics
    out()
    out("=" * 108)
    out("F. QUANTIFICATION DIAGNOSTICS (original sample, summed over runs)")
    out("   Cells: group x value estimates (corpus, manifestos, news, 9 languages = 12 groups x 19 values per run).")
    out("   Fallback: tpr - fpr <= 0 or undefined, so the adjusted count reverts to CC (for RC: slope <= "
        f"{MIN_SLOPE}, no correction).")
    out("   At 0 / at 1: raw estimate below 0 or above 1 before clipping (EM: solution on the boundary).")
    out("=" * 108)
    for system in SYSTEMS:
        out(f"\n{system}")
        for r in QUANT_RULES + GROUP_RULES:
            q = qstats[system][r]
            if r == "RC@prev":
                line = f"  {r:12s} slopes {q['cells']:4d}   fallback {q['fallback']:d}"
            else:
                line = f"  {r:12s} cells {q['cells']:5d}   fallback {100 * q['fallback'] / max(q['cells'], 1):5.1f}%"
            line += f"   at 0: {q['clip0']:4d}   at 1: {q['clip1']:4d}"
            out(line)
        sl = np.array(slopes_v[system])
        out(f"  validation attenuation slopes (CC@prev): language {', '.join(f'{x:.2f}' for x in sl[:, 0])}; "
            f"genre {', '.join(f'{x:.2f}' for x in sl[:, 1])}")
        out(f"  EM binning check: max |binned - exact| corpus prevalence over values and runs = "
            f"{max(em_check[system]):.2e}")

    # ---- G. why adjusted counts with pooled error rates fail between genres
    out()
    out("=" * 108)
    out("G. DO ERROR RATES DIFFER BETWEEN GENRES? (CC@prev decisions on test; mean over runs)")
    out("   Adjusted counts assume the same tpr and fpr in every group (prior shift only). Here: Spearman")
    out("   correlation, across the 19 values, between the gold manifesto-news contrast and the manifesto-news")
    out("   gap in tpr / in fpr. Positive = the classifier fires more, correctly and falsely, in the genre where")
    out("   the value is common, which inflates raw contrasts and breaks the prior-shift assumption.")
    out("=" * 108)
    for system in SYSTEMS:
        m = np.array(mechanism[system])
        out(f"  {system:17s} rho(contrast, tpr gap) {m[:, 0].mean():+.2f}   rho(contrast, fpr gap) {m[:, 1].mean():+.2f}   "
            f"fpr gap has the contrast's sign for {m[:, 2].mean() * 100:.0f}% of values")

    # ---- CSVs
    RES.mkdir(exist_ok=True)
    per_value, genre_rows, lang_rows = [], [], []
    for system in SYSTEMS:
        for j, val in enumerate(VALUES):
            row = {"system": system, "value": val, "gold_prev": gold["all"][0, j],
                   "gold_lo": np.percentile(gold["all"][1:, j], 2.5), "gold_hi": np.percentile(gold["all"][1:, j], 97.5)}
            grow = {"system": system, "value": val, "gold_diff": gold["genre"][0, j],
                    "gold_lo": np.percentile(gold["genre"][1:, j], 2.5), "gold_hi": np.percentile(gold["genre"][1:, j], 97.5),
                    "gold_sig": bool(gsig["genre"][j])}
            for r in RULES:
                m = np.mean([v["all"][:, j] for v in rows[system][r]], axis=0)
                row[r], row[f"{r}_lo"], row[f"{r}_hi"] = m[0], *np.percentile(m[1:], [2.5, 97.5])
                gm = np.mean([v["genre"][0, j] for v in rows[system][r]])
                grow[r] = gm
                grow[f"{r}_sig_seeds"] = int(sum(bh(boot_p(v["genre"]))[j] < ALPHA for v in rows[system][r]))
            per_value.append(row); genre_rows.append(grow)
        lp_sig = {r: np.sum([bh(boot_p(v["lpair"])) < ALPHA for v in rows[system][r]], axis=0) for r in RULES}
        for k, (i, jl) in enumerate(pairs):
            for j, val in enumerate(VALUES):
                idx = k * len(VALUES) + j
                lrow = {"system": system, "lang_a": langs[i], "lang_b": langs[jl], "value": val,
                        "gold_diff": gold["lpair"][0, idx], "gold_sig": bool(gsig["lpair"][idx])}
                for r in RULES:
                    lrow[r] = np.mean([v["lpair"][0, idx] for v in rows[system][r]])
                    lrow[f"{r}_sig_seeds"] = int(lp_sig[r][idx])
                lang_rows.append(lrow)
    pd.DataFrame(per_value).to_csv(RES / "measurement-per-value.csv", index=False, float_format="%.5f")
    pd.DataFrame(genre_rows).to_csv(RES / "measurement-genre-contrasts.csv", index=False, float_format="%.5f")
    pd.DataFrame(lang_rows).to_csv(RES / "measurement-language-contrasts.csv", index=False, float_format="%.5f")
    pd.DataFrame(list(summary.values())).to_csv(RES / "measurement-summary.csv", index=False, float_format="%.5f")
    (RES / "measurement-analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'measurement-analysis.txt'} and measurement-*.csv")


if __name__ == "__main__":
    main()
