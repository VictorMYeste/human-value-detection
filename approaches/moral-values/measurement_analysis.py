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

Further sections (step 5.2): G adds error rates per value by genre and language, on validation and
test (AE item 8; measurement-error-rates.csv); H accuracy and coverage; I totals per group; J language
contrasts on value composition, which removes the teams' differences in labeling rate (AE item 5).

Estimators that use the annotated validation documents of each group directly (step 5.2, AE
item 4). Like ACC-G and RC, they need annotated documents from every group being compared:
  Gold-only    the share of validation sentences with the value in the group: no classifier;
  PPI@prev     prediction-powered (difference) estimate per group: the CC@prev share on test
               plus the mean of (label - CC@prev decision) over the group's validation sentences
               (Angelopoulos et al. 2023); not clipped, so it can fall below 0;
  PPI-Platt    the same with Platt-calibrated probabilities as the predictions;
  Platt-G      calibrated probability average with Platt maps fitted on each group's own
               validation documents (each genre, each language); pooled maps for the corpus total.

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
genre stratum, B = 2000, with the rescaling bootstrap of Rao and Wu (n_h - 1 draws per
stratum, weights scaled by n_h / (n_h - 1)) so that strata with few documents keep their
variance. The same resamples are used for every system, seed and rule, so differences
between rules are paired. A system's estimate is the mean of its runs; every interval
and test is for that mean (95% percentile intervals). Tests use normal-approximation
p-values from the bootstrap standard error, with Benjamini-Hochberg correction within
each family (19 genre contrasts; 36 x 19 = 684 language-pair contrasts). Reference
contrasts in which one group has no mentions of a value use Fisher's exact test.
Slopes are least-squares fits through the origin.
Everything fitted on validation is refitted on every validation resample (documents
resampled in the same way; independent resamples, paired by index): the thresholds, the
Platt maps, the error rates of ACC, PACC and ACC-G, the prior of EM and the attenuation
slopes of RC (class RowFits). Validation sampling error therefore enters every interval.

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
from scipy.stats import fisher_exact, norm, rankdata, spearmanr
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
ANNOT_RULES = ["Gold-only", "PPI@prev", "PPI-Platt", "Platt-G"]   # use each group's annotated documents
RULES = CLASSIFY_RULES + QUANT_RULES + GROUP_RULES + ANNOT_RULES
CC_RULES = ["CC@0.5", "CC@t*", "CC@label-t*", "CC@prev"]
EM_BIN_WIDTH = 0.1    # EM works on per-group histograms of raw logits, with bins of 0.1 logit
EM_STEPS = 40         # at most this many Newton (or bisection) steps for the maximum-likelihood prior
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
    """Row 0 is the original sample; rows 1..B resample documents within strata.

    Rescaling bootstrap (Rao and Wu 1988): each row draws n_h - 1 of a stratum's n_h documents
    with replacement and scales their weights by n_h / (n_h - 1). Drawing all n_h would understate
    a stratum's variance by the factor (n_h - 1) / n_h, a quarter for the smallest strata here
    (4 manifestos).
    """
    w = np.zeros((B + 1, len(stratum)))
    w[0] = 1
    for s in np.unique(stratum):
        idx = np.where(stratum == s)[0]
        n = len(idx)
        assert n >= 2, f"stratum {s} has a single document"
        draws = rng.integers(0, n, size=(B, n - 1))
        counts = np.zeros((B, n))
        np.add.at(counts, (np.repeat(np.arange(B), n - 1), draws.ravel()), 1)
        w[1:, idx] = counts * n / (n - 1)
    return w


def boot_p(d: np.ndarray) -> np.ndarray:
    """Two-sided p-value that a quantity differs from zero: normal approximation with the bootstrap
    standard error (rows 1..B). Unlike a p-value counted from the resamples, it has no floor at
    2 / (B + 1), which would block Benjamini-Hochberg in the family of 684 language contrasts."""
    se = d[1:].std(0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(se > 0, np.abs(d[0]) / se, np.where(d[0] == 0, 0.0, np.inf))
    return 2 * norm.sf(z)


def empty_cell_p(p: np.ndarray, k_a: np.ndarray, n_a, k_b: np.ndarray, n_b) -> np.ndarray:
    """Reference contrasts in which one group has no mentions of a value: the bootstrap cannot vary
    that group, so the p-value comes from Fisher's exact test on sentence counts instead.
    k_a, k_b: sentences with the value in each group (row 0); n_a, n_b: sentences per group."""
    k_a, k_b = np.broadcast_arrays(np.asarray(k_a), np.asarray(k_b))
    n_a, n_b = np.broadcast_to(n_a, k_a.shape), np.broadcast_to(n_b, k_b.shape)
    p = np.array(p, float, copy=True)
    for i in zip(*np.where((k_a == 0) | (k_b == 0))):
        table = [[int(k_a[i]), int(n_a[i] - k_a[i])], [int(k_b[i]), int(n_b[i] - k_b[i])]]
        p[i] = fisher_exact(table)[1]
    return p


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1.0)
    return out


def genre_reference_p(d: np.ndarray, gold_doc: np.ndarray, nsent: np.ndarray, man: np.ndarray,
                      news: np.ndarray) -> np.ndarray:
    """p-values of reference manifesto-news contrasts d (B+1, 19), with empty cells handled."""
    return empty_cell_p(boot_p(d), gold_doc[man].sum(0), nsent[man].sum(), gold_doc[news].sum(0), nsent[news].sum())


def lpair_reference_p(d: np.ndarray, gold_doc: np.ndarray, nsent: np.ndarray, lang_masks: list[np.ndarray],
                      pairs: list[tuple[int, int]]) -> np.ndarray:
    """p-values of reference language-pair contrasts d (B+1, pairs x 19), with empty cells handled."""
    k = np.stack([gold_doc[m].sum(0) for m in lang_masks]); n = np.array([nsent[m].sum() for m in lang_masks])
    k_a = np.concatenate([k[i] for i, _ in pairs]); k_b = np.concatenate([k[j] for _, j in pairs])
    n_a = np.repeat([n[i] for i, _ in pairs], k.shape[1]); n_b = np.repeat([n[j] for _, j in pairs], k.shape[1])
    return empty_cell_p(boot_p(d), k_a, n_a, k_b, n_b)


def slope(g: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Least-squares slope of estimated on gold contrasts through the origin, per bootstrap row
    (1 = no attenuation). A contrast changes sign with the order of its two groups; a fit through
    the origin does not depend on that order, and it matches RC, which divides by the slope alone."""
    return (g * e).sum(1) / (g ** 2).sum(1)


def slope_intercept(g: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares with an intercept, original sample only (a diagnostic)."""
    b = np.polyfit(g[0], e[0], 1)
    return float(b[0]), float(b[1])


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


# ----------------------------------------------------------------------------- rules refitted per bootstrap row
PLATT_C = 1e4         # inverse penalty of the Platt maps (as LogisticRegression(C=1e4) for the original sample)
PLATT_BIN = 0.05      # per-row Platt maps are fitted on histograms of raw logits with this bin width
LOGIT_MAX = float(np.log((1 - 1e-6) / 1e-6))   # the clip of logit()


def doc_index(S) -> np.ndarray:
    """Document of each sentence, from the documents x sentences matrix."""
    return np.asarray(S.argmax(0)).ravel()


def counts_at_or_above(doc: np.ndarray, p: np.ndarray, n_docs: int, cands: np.ndarray,
                       weight: np.ndarray | None = None) -> np.ndarray:
    """Per document, the (weighted) number of sentences with p >= each candidate threshold,
    (n_docs, len(cands)); cands must be sorted."""
    pos = np.searchsorted(cands, p, side="right")      # number of candidates <= p
    h = doc_hist(doc, pos, n_docs, len(cands) + 1, weight)
    return np.cumsum(h[:, ::-1], axis=1)[:, ::-1][:, 1:]   # p >= cands[c]  <=>  pos > c


def doc_hist(doc: np.ndarray, col: np.ndarray, n_docs: int, n_cols: int, weight=None) -> np.ndarray:
    """(n_docs, n_cols) sums of weight (default 1) over sentences, by document and column."""
    return np.bincount(doc * n_cols + col, weights=weight, minlength=n_docs * n_cols).reshape(n_docs, n_cols).astype(float)


def wsum(W: np.ndarray, c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bootstrap-weighted sum over the documents in mask of per-document counts: (docs, k) for counts
    that are the same in every row, (B+1, docs, k) for counts that depend on the row."""
    if c.ndim == 3:
        return np.einsum("bd,bdk->bk", W[:, mask], c[:, mask], optimize=True)
    return W[:, mask] @ c[mask]


def platt_newton(n: np.ndarray, ypos: np.ndarray, zbar: np.ndarray, a: np.ndarray, c: np.ndarray,
                 steps: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Platt map per row from histograms (rows x bins: counts n, positives ypos, mean logit zbar):
    minimises the logistic loss of a * z + c plus a^2 / (2 PLATT_C), as LogisticRegression(C=PLATT_C)
    does. Damped Newton: each row's step is halved until its loss does not increase. A row stops when
    its step or its loss decrease becomes negligible; a group with no positive sentence has no finite
    optimum (the intercept drifts down), and stops once its calibrated probabilities are about 0."""
    a, c = a.astype(float).copy(), c.astype(float).copy()

    def loss(rows, a_, c_):
        eta = a_[:, None] * zbar[rows] + c_[:, None]
        return (n[rows] * np.logaddexp(0, eta) - ypos[rows] * eta).sum(1) + a_ ** 2 / (2 * PLATT_C)

    act = np.arange(len(a))                       # rows still iterating
    cur = loss(act, a, c)
    for _ in range(steps):
        aa, cc, nn, yy, zz = a[act], c[act], n[act], ypos[act], zbar[act]
        mu = 0.5 * (1 + np.tanh(0.5 * (aa[:, None] * zz + cc[:, None])))   # overflow-free logistic
        r, v = yy - nn * mu, nn * mu * (1 - mu)
        ga, gc = (r * zz).sum(1) - aa / PLATT_C, r.sum(1)                     # minus the gradient
        haa, hac, hcc = (v * zz ** 2).sum(1) + 1 / PLATT_C, (v * zz).sum(1), v.sum(1)
        det = haa * hcc - hac ** 2
        ok = det > 1e-12 * np.maximum(haa * hcc, 1e-300)
        safe = np.where(ok, det, 1.0)
        da = np.where(ok, (hcc * ga - hac * gc) / safe, ga / np.maximum(haa, 1e-12))
        dc = np.where(ok, (haa * gc - hac * ga) / safe, gc / np.maximum(hcc, 1e-12))
        t = np.ones_like(aa)
        new = loss(act, aa + da, cc + dc)
        for _ in range(40):                     # halve the step only where the loss went up
            worse = np.flatnonzero(~(new <= cur + 1e-12 * np.abs(cur)))
            if not len(worse):
                break
            t[worse] /= 2
            new[worse] = loss(act[worse], aa[worse] + t[worse] * da[worse], cc[worse] + t[worse] * dc[worse])
        keep = new <= cur + 1e-12 * np.abs(cur)
        a[act] = np.where(keep, aa + t * da, aa)
        c[act] = np.where(keep, cc + t * dc, cc)
        gain = np.where(keep, cur - new, 0.0)
        cur = np.where(keep, new, cur)
        done = (np.maximum(np.abs(t * da), np.abs(t * dc)) < 1e-9) | (gain < 1e-9) | ~keep
        act, cur = act[~done], cur[~done]
        if not len(act):
            break
    return a, c


class RowFits:
    """Decision rules refitted on every validation bootstrap row (row 0 = the original sample).

    Thresholds and Platt maps are refitted on each resample of validation documents, so their
    fitting error enters every interval, as the error rates of ACC and the slopes of RC do.
      t*      the 0.01-grid threshold that maximises the row's weighted macro-F1 (ties -> lowest);
      t_j     per-label grid thresholds that maximise each label's weighted F1;
      prev    per-label thresholds at the weighted (1 - prevalence) quantile of the row's
              validation probabilities (inverted empirical distribution);
      Platt   per-label logistic map of the logit, fitted by Newton's method on per-row
              histograms of the logit (bins of PLATT_BIN).
    sent_mask restricts the fit to some validation sentences (the shift tests, per-group fits);
    platt_only skips the thresholds (Platt-G).
    """

    def __init__(self, pv: np.ndarray, yv: np.ndarray, design_v: dict, sent_mask: np.ndarray | None = None,
                 platt_only: bool = False):
        W = design_v["W"]
        n_docs = W.shape[1]
        m = np.ones(len(pv), bool) if sent_mask is None else sent_mask
        p, y, d = pv[m], yv[m].astype(float), doc_index(design_v["S"])[m]
        k, G = p.shape[1], THRESHOLD_GRID
        self.k = k

        rows = W.shape[0]
        if not platt_only:
            # grid rules: weighted true and false positives at every grid threshold
            tp = np.stack([counts_at_or_above(d, p[:, j], n_docs, G, y[:, j]) for j in range(k)], 1)
            fp = np.stack([counts_at_or_above(d, p[:, j], n_docs, G, 1 - y[:, j]) for j in range(k)], 1)
            pos_doc = np.stack([np.bincount(d, weights=y[:, j], minlength=n_docs) for j in range(k)], 1)
            TP = (W @ tp.reshape(n_docs, -1)).reshape(-1, k, len(G))
            FP = (W @ fp.reshape(n_docs, -1)).reshape(-1, k, len(G))
            POS = (W @ pos_doc)[:, :, None]
            den = TP + FP + POS                       # 2 TP + FP + FN, with FN = POS - TP
            f1 = np.divide(2 * TP, den, out=np.zeros_like(den), where=den > 0)
            self.g_label = f1.argmax(2)               # (B+1, k) grid indices
            self.g_star = f1.mean(1).argmax(1)        # (B+1,)
            self.tlabel, self.tstar = G[self.g_label], G[self.g_star]

            # prevalence-matched thresholds: per row, the first sentence (in order of probability) at which
            # the row's cumulative weight reaches (1 - prevalence) of its total, found by bisection on a
            # per-document table of how many of each document's sentences come first
            n_doc_sent = np.bincount(d, minlength=n_docs).astype(float)
            tot = W @ n_doc_sent
            prev = (W @ pos_doc) / tot[:, None]
            self.prior = prev                         # each row's validation prevalence (the prior of EM)
            self.tprev = np.zeros((rows, k))
            for j in range(k):
                order = np.argsort(p[:, j], kind="stable")
                first = np.zeros((n_docs, len(order) + 1), np.float32)     # first[d, i]: doc d's sentences among the first i
                first[d[order], np.arange(1, len(order) + 1)] = 1
                np.cumsum(first, axis=1, out=first)
                target = (1 - prev[:, j]) * tot
                lo, hi = np.zeros(rows, int), np.full(rows, len(order) - 1)
                while (lo < hi).any():
                    mid = (lo + hi) // 2
                    reach = np.einsum("bd,db->b", W, first[:, mid + 1]) >= target
                    act = lo < hi                     # finished rows stay put (a value with no positive in a
                    hi = np.where(act & reach, mid, hi)   # row targets all its sentences, which the float32
                    lo = np.where(act & ~reach, mid + 1, lo)   # table may miss by a rounding error)
                self.tprev[:, j] = p[order[lo], j]

        # Platt maps, fitted per row on histograms of the logit over its observed range
        z = logit(p)
        self.platt = np.zeros((rows, k, 2))
        for j in range(k):
            z0 = z[:, j].min()
            nb = int((z[:, j].max() - z0) / PLATT_BIN) + 1
            col = np.minimum(((z[:, j] - z0) / PLATT_BIN).astype(int), nb - 1)
            hist = lambda wt: doc_hist(d, col, n_docs, nb, wt)
            n, ypos, zs = W @ hist(None), W @ hist(y[:, j]), W @ hist(z[:, j])
            zbar = np.divide(zs, n, out=np.zeros_like(zs), where=n > 0)
            # row 0 first, from the identity map; every row then starts from row 0's fit
            a0, c0 = platt_newton(n[:1], ypos[:1], zbar[:1], np.ones(1), np.zeros(1))
            a, c = platt_newton(n, ypos, zbar, np.repeat(a0, rows), np.repeat(c0, rows))
            assert np.isfinite(a).all() and np.isfinite(c).all(), f"Platt refit failed for label {j}"
            self.platt[:, j, 0], self.platt[:, j, 1] = a, c

    def counts(self, rule: str, p: np.ndarray, S, weight: np.ndarray | None = None) -> np.ndarray:
        """Per-row, per-document counts of a fitted rule's decisions, (B+1, docs, k): sentences at or
        above the row's threshold (times weight, e.g. the gold label, if given), or for PCC-Platt the
        sum of the row's calibrated probabilities."""
        doc, n_docs, k = doc_index(S), S.shape[0], self.k
        rows = self.platt.shape[0]
        wt = (lambda j: None) if weight is None else (lambda j: weight[:, j])
        out = np.zeros((rows, n_docs, k), np.float32)
        if rule in ("CC@t*", "CC@label-t*"):
            g = np.stack([counts_at_or_above(doc, p[:, j], n_docs, THRESHOLD_GRID, wt(j)) for j in range(k)], 1)
            idx = np.broadcast_to(self.g_star[:, None], (rows, k)) if rule == "CC@t*" else self.g_label
            for j in range(k):
                out[:, :, j] = g[:, j, idx[:, j]].T
        elif rule == "CC@prev":
            for j in range(k):
                cands = np.unique(self.tprev[:, j])
                c = counts_at_or_above(doc, p[:, j], n_docs, cands, wt(j))
                out[:, :, j] = c[:, np.searchsorted(cands, self.tprev[:, j])].T
        elif rule == "PCC-Platt":
            # Logits binned at PLATT_BIN; per bin, the calibrated probability at the bin's mean logit plus
            # a first-order correction for each sentence's deviation from that mean. The error is of
            # second order (below 1e-3 mentions per document; checked against the exact sum).
            assert weight is None
            z = logit(p)
            for j in range(k):
                z0 = z[:, j].min()
                nb = int((z[:, j].max() - z0) / PLATT_BIN) + 1
                col = np.minimum(((z[:, j] - z0) / PLATT_BIN).astype(int), nb - 1)
                zbar = np.bincount(col, z[:, j], nb) / np.maximum(np.bincount(col, minlength=nb), 1)
                H = doc_hist(doc, col, n_docs, nb)
                D = doc_hist(doc, col, n_docs, nb, z[:, j] - zbar[col])
                a, c = self.platt[:, j, 0, None], self.platt[:, j, 1, None]
                with np.errstate(over="ignore"):
                    cal = 1 / (1 + np.exp(-(a * zbar[None, :] + c)))
                out[:, :, j] = cal @ H.T + (a * cal * (1 - cal)) @ D.T
        else:
            raise KeyError(rule)
        return out


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

    Raw probabilities are binned on the logit scale per label (bins of EM_BIN_WIDTH); per-document
    bin counts and logit sums make every bootstrap row a weighted histogram. Each row maps its bins'
    mean logit through that row's Platt map and uses that row's validation prevalence as the prior,
    so the calibration is refitted with the row. The concave log-likelihood is then maximised by
    Newton steps kept inside a bisection bracket (vectorised over rows and labels).
    """

    def __init__(self, S, W: np.ndarray, p_raw: np.ndarray, platt: np.ndarray, prior: np.ndarray):
        n, k = p_raw.shape
        self.W, self.k = W, k
        self.a, self.c, self.prior = platt[:, :, 0, None], platt[:, :, 1, None], prior[:, :, None]
        self.nb = int(np.ceil(2 * LOGIT_MAX / EM_BIN_WIDTH))
        z = logit(p_raw)
        idx = np.minimum(((z + LOGIT_MAX) / EM_BIN_WIDTH).astype(int), self.nb - 1)
        cols = (np.arange(k)[None, :] * self.nb + idx).ravel()
        rws = np.repeat(np.arange(n), k)
        shape = (n, k * self.nb)
        self.H = np.asarray((S @ csr_matrix((np.ones(n * k), (rws, cols)), shape=shape)).todense())
        self.Z = np.asarray((S @ csr_matrix((z.ravel(), (rws, cols)), shape=shape)).todense())
        self.centres = -LOGIT_MAX + EM_BIN_WIDTH * (np.arange(self.nb) + 0.5)

    def prevalence(self, mask: np.ndarray) -> np.ndarray:
        rows = self.W.shape[0]
        h = (self.W[:, mask] @ self.H[mask]).reshape(rows, self.k, self.nb)
        zs = (self.W[:, mask] @ self.Z[mask]).reshape(rows, self.k, self.nb)
        occ = h.any(axis=(0, 1))                  # bins the group never reaches add nothing
        h, zs, centres = h[:, :, occ], zs[:, :, occ], self.centres[occ]
        with np.errstate(invalid="ignore", divide="ignore"):
            zbar = np.where(h > 0, zs / h, centres[None, None, :])
        pbar = 1 / (1 + np.exp(-(self.a * zbar + self.c)))
        prior = self.prior
        a = pbar / prior - (1 - pbar) / (1 - prior)
        b = (1 - pbar) / (1 - prior)
        d = lambda x: (h * a / (b + x[:, :, None] * a)).sum(2)   # derivative of the log-likelihood, decreasing
        # Newton steps on the concave log-likelihood, kept inside a shrinking bracket (bisection if a step leaves it)
        with np.errstate(invalid="ignore", divide="ignore"):
            g_lo, g_hi = d(np.zeros((rows, self.k))).ravel(), d(np.ones((rows, self.k))).ravel()
            at0, at1 = g_lo <= 0, g_hi >= 0                       # maxima on the boundary
            A, Bv, Hh = (m.reshape(rows * self.k, -1) for m in (a, b, h))
            lo, hi = np.zeros(rows * self.k), np.ones(rows * self.k)
            x = np.broadcast_to(prior[:, :, 0], (rows, self.k)).ravel().copy()
            act = np.flatnonzero(~at0 & ~at1)                     # estimates still iterating
            for _ in range(EM_STEPS):
                if not len(act):
                    break
                xa, aa = x[act], A[act]
                q = aa / (Bv[act] + xa[:, None] * aa)
                g, hess = (Hh[act] * q).sum(1), -(Hh[act] * q * q).sum(1)
                up = g > 0
                lo[act], g_lo[act] = np.where(up, xa, lo[act]), np.where(up, g, g_lo[act])
                hi[act], g_hi[act] = np.where(up, hi[act], xa), np.where(up, g_hi[act], g)
                l_, h_, gl, gh = lo[act], hi[act], g_lo[act], g_hi[act]
                xn = xa - g / hess
                # a Newton step that leaves the bracket is replaced by the secant through its two ends
                sec = (l_ * gh - h_ * gl) / (gh - gl)
                sec = np.where(np.isfinite(sec) & (sec > l_) & (sec < h_), sec, (l_ + h_) / 2)
                xn = np.where(~np.isfinite(xn) | (xn <= l_) | (xn >= h_), sec, xn)
                x[act] = xn
                act = act[np.abs(xn - xa) >= 1e-12]
            x = x.reshape(rows, self.k)
            at0, at1 = at0.reshape(rows, self.k), at1.reshape(rows, self.k)
            return np.where(at0, 0.0, np.where(at1, 1.0, x))


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
        """Group prevalence per bootstrap row from per-document counts, fixed (docs, k) or per row (B+1, docs, k)."""
        return wsum(W, cdoc, mask) / (W[:, mask] @ nsent[mask])[:, None]

    def group_level(cdoc: np.ndarray) -> dict[str, np.ndarray]:
        """Group prevalences per bootstrap row from per-document counts (or probability sums)."""
        return {
            "all": group_prev(cdoc, allm), "man": group_prev(cdoc, man), "news": group_prev(cdoc, news),
            "lang": np.stack([group_prev(cdoc, m) for m in lang_masks], axis=1),        # (B+1, 9, 19)
            "doc": ((cdoc[0] if cdoc.ndim == 3 else cdoc) / nsent[:, None])[long_docs],
        }

    def derive(gl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The quantities every section evaluates: totals, contrasts, document shares."""
        lp = gl["lang"]
        return {
            "all": gl["all"], "man": gl["man"], "news": gl["news"],
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
        return wsum(Wv, cdoc, mask) / (Wv[:, mask] @ nsent_v[mask])[:, None]

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

    gold_doc = S @ yt
    gold = views(gold_doc)
    gold_v_doc = Sv @ yv
    all_v = np.ones(Sv.shape[0], bool)
    doc_t, doc_v = doc_index(S), doc_index(Sv)
    nodoc = np.full((int(long_docs.sum()), len(VALUES)), np.nan)

    def val_level(cdoc: np.ndarray) -> dict[str, np.ndarray]:
        """Group prevalences on the validation resamples (Wv rows, paired with the test rows by index)."""
        return {"all": val_group_prev(cdoc, all_v), "man": val_group_prev(cdoc, man_v), "news": val_group_prev(cdoc, news_v),
                "lang": np.stack([val_group_prev(cdoc, m) for m in lang_masks_v], axis=1)}

    gold_vgl = val_level(gold_v_doc)
    v_groups = [("manifestos", man_v), ("news", news_v)] + list(zip(langs, lang_masks_v))
    no_pos = {g: [VALUES[j] for j in np.flatnonzero(gold_v_doc[m].sum(0) == 0)] for g, m in v_groups}
    rare = gold["all"][0] < RARE
    gold_p = {"genre": genre_reference_p(gold["genre"], gold_doc, nsent, man, news),
              "lpair": lpair_reference_p(gold["lpair"], gold_doc, nsent, lang_masks, pairs)}
    gsig = {k: bh(p) < ALPHA for k, p in gold_p.items()}

    out("MEASUREMENT ANALYSIS -- ValueEval'24 English test split")
    out(f"{n_sent_total} sentences in {n_docs} documents ({man.sum()} manifestos, {news.sum()} news); "
        f"{len(langs)} source languages; stratified document bootstrap B = {B}")
    out(f"Rare values (< {RARE:.0%} gold prevalence): {rare.sum()} of 19")
    out()

    # ---- run every system x seed x rule
    rows = {s: {r: [] for r in RULES} for s in SYSTEMS}
    diag = {s: [] for s in SYSTEMS}
    qstats = {s: {r: {"cells": 0, "fallback": 0, "clip0": 0, "clip1": 0}
                  for r in QUANT_RULES + GROUP_RULES + ["PPI@prev", "PPI-Platt"]} for s in SYSTEMS}
    em_check, slopes_v = {s: [] for s in SYSTEMS}, {s: [] for s in SYSTEMS}
    platt_check = {s: [] for s in SYSTEMS}
    pos_v, neg_v = Sv @ yv, Sv @ (1 - yv)
    mechanism = {s: [] for s in SYSTEMS}
    man_sent = gt["Text-ID"].str.contains("_M_").to_numpy()
    err_rates = {s: [] for s in SYSTEMS}
    sent_groups = {
        "validation": {"all": all_v[doc_v], "manifestos": man_v[doc_v], "news": news_v[doc_v],
                       **{l: m[doc_v] for l, m in zip(langs, lang_masks_v)}},
        "test": {"all": allm[doc_t], "manifestos": man[doc_t], "news": news[doc_t],
                 **{l: m[doc_t] for l, m in zip(langs, lang_masks)}}}

    def group_rates(dec: np.ndarray, y: np.ndarray, groups: dict) -> dict:
        """tpr, fpr and the counts behind them, per group of sentences, (4, 19) each."""
        res = {}
        for g, m in groups.items():
            pos, neg = y[m].sum(0), (1 - y[m]).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                res[g] = np.stack([(dec[m] * y[m]).sum(0) / pos, (dec[m] * (1 - y[m])).sum(0) / neg, pos, neg])
        return res

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
            # thresholds and Platt maps refitted on every validation bootstrap row
            rf = RowFits(pv, yv, vdesign)
            fixed_counts = {"CC@0.5": S @ est["CC@0.5"], "PCC": S @ pt}
            row_counts = {r: rf.counts(r, pt, S) for r in CLASSIFY_RULES if r not in fixed_counts}
            exact0 = S @ (1 / (1 + np.exp(-(logit(pt) * rf.platt[0, :, 0] + rf.platt[0, :, 1]))))
            platt_check[system].append((np.abs(rf.platt[0] - fit["platt"]).max(),
                                        np.abs(row_counts["PCC-Platt"][0] - exact0).max()))
            gl = {r: group_level(fixed_counts[r] if r in fixed_counts else row_counts.pop(r)) for r in CLASSIFY_RULES}
            del row_counts
            for r in CLASSIFY_RULES:
                rows[system][r].append(derive(gl[r]))
            g0 = gold["genre"][0]
            gap_t, gap_f = genre_rate_gap(est["CC@prev"], 1), genre_rate_gap(est["CC@prev"], 0)
            mechanism[system].append((spearmanr(g0, gap_t)[0], spearmanr(g0, gap_f)[0],
                                      (np.sign(gap_f) == np.sign(g0)).mean()))
            dec_v = apply_rules(pv, fit)["CC@prev"]
            err_rates[system].append({"validation": group_rates(dec_v, yv, sent_groups["validation"]),
                                      "test": group_rates(est["CC@prev"], yt, sent_groups["test"])})

            # validation true and false positives of each row's own decisions, per document
            vcounts = {r: (rf.counts(r, pv, Sv, weight=yv), rf.counts(r, pv, Sv, weight=1 - yv))
                       for r in ["CC@t*", "CC@prev"]}

            def rates(rule: str, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
                m = np.ones(Sv.shape[0], bool) if mask is None else mask
                tp, fp = vcounts[rule]
                with np.errstate(invalid="ignore", divide="ignore"):
                    return wsum(Wv, tp, m) / wsum(Wv, pos_v, m), wsum(Wv, fp, m) / wsum(Wv, neg_v, m)

            # standard quantification estimators
            qs = qstats[system]
            rows[system]["ACC@t*"].append(adjusted(gl["CC@t*"], rates("CC@t*"), qs["ACC@t*"]))
            rates_prev = rates("CC@prev")
            rows[system]["ACC@prev"].append(adjusted(gl["CC@prev"], rates_prev, qs["ACC@prev"]))
            rows[system]["PACC"].append(adjusted(gl["PCC"], boot_rates(Sv, Wv, yv, pv), qs["PACC"]))
            em = EMGroups(S, W, pt, rf.platt, rf.prior)
            em_gl = {"all": em.prevalence(allm), "man": em.prevalence(man), "news": em.prevalence(news),
                     "lang": np.stack([em.prevalence(m) for m in lang_masks], axis=1),
                     "doc": np.full_like(gl["PCC"]["doc"], np.nan)}
            for k in ["all", "man", "news", "lang"]:
                qs["EM"]["cells"] += em_gl[k][0].size
                qs["EM"]["clip0"] += int((em_gl[k][0] <= 0).sum()); qs["EM"]["clip1"] += int((em_gl[k][0] >= 1).sum())
            pc0 = 1 / (1 + np.exp(-(logit(pt) * rf.platt[0, :, 0] + rf.platt[0, :, 1])))
            em_check[system].append(np.abs(em_gl["all"][0] - em_prior_exact(pc0, rf.prior[0])).max())
            rows[system]["EM"].append(derive(em_gl))

            # corrections for group comparisons (step 1.3b), on the prevalence-matched rule
            rows[system]["ACC-G@prev"].append(adjusted(
                gl["CC@prev"], rates_prev, qs["ACC-G@prev"],
                by_genre=(rates("CC@prev", man_v), rates("CC@prev", news_v)),
                by_lang=tuple(np.stack(x, axis=1) for x in zip(*[rates("CC@prev", m) for m in lang_masks_v]))))
            beta_lang, beta_genre = val_slopes(vcounts["CC@prev"][0] + vcounts["CC@prev"][1], gold_v_doc)
            slopes_v[system].append((beta_lang[0], beta_genre[0]))
            rows[system]["RC@prev"].append(regression_calibrated(gl["CC@prev"], beta_lang, beta_genre,
                                                                 qs["RC@prev"]))

            # estimators that use each group's annotated validation documents (step 5.2, AE item 4)
            rows[system]["Gold-only"].append(derive({**gold_vgl, "doc": nodoc}))
            pred_v = {"CC@prev": vcounts["CC@prev"][0] + vcounts["CC@prev"][1], "PCC-Platt": rf.counts("PCC-Platt", pv, Sv)}
            for name, base in [("PPI@prev", "CC@prev"), ("PPI-Platt", "PCC-Platt")]:
                pgl = val_level(pred_v[base])
                ppi = {g: gl[base][g] + gold_vgl[g] - pgl[g] for g in ["all", "man", "news", "lang"]}
                for g in ppi:
                    qs[name]["cells"] += ppi[g][0].size
                    qs[name]["clip0"] += int((ppi[g][0] < 0).sum()); qs[name]["clip1"] += int((ppi[g][0] > 1).sum())
                rows[system][name].append(derive({**ppi, "doc": nodoc}))
            del pred_v

            def platt_group(mask_v: np.ndarray, mask_t: np.ndarray) -> np.ndarray:
                """Calibrated average in one group, with Platt maps fitted on that group's validation documents."""
                rfg = RowFits(pv, yv, vdesign, sent_mask=mask_v[doc_v], platt_only=True)
                cols = np.flatnonzero(mask_t[doc_t])
                return group_prev(rfg.counts("PCC-Platt", pt[cols], S[:, cols]), mask_t)

            rows[system]["Platt-G"].append(derive({
                "all": gl["PCC-Platt"]["all"], "man": platt_group(man_v, man), "news": platt_group(news_v, news),
                "lang": np.stack([platt_group(mv, mt) for mv, mt in zip(lang_masks_v, lang_masks)], axis=1),
                "doc": nodoc}))
            del vcounts, gl

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

    _means: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def run_mean(system: str, rule: str) -> dict[str, np.ndarray]:
        """A system's estimate: the mean of its runs, row by row. Every test is on this mean."""
        if (system, rule) not in _means:
            vs = rows[system][rule]
            _means[(system, rule)] = {k: np.mean([v[k] for v in vs], axis=0)
                                      for k in ["all", "man", "news", "genre", "lang", "lpair"]}
        return _means[(system, rule)]

    summary: dict[tuple[str, str], dict] = {}  # system x rule -> metrics, written to measurement-summary.csv

    def put(system: str, rule: str, name: str, value, arr: np.ndarray | None = None) -> None:
        row = summary.setdefault((system, rule), {"system": system, "rule": rule, "runs": len(rows[system][rule])})
        row[name] = value
        if arr is not None:
            row[f"{name}_lo"], row[f"{name}_hi"] = np.percentile(arr[1:], [2.5, 97.5])

    def put_spread(system: str, rule: str, name: str, per_run: list[float]) -> None:
        """Range over a system's runs of a metric computed on each run alone (what one model gives)."""
        if len(per_run) > 1:
            put(system, rule, f"{name}_run_min", float(np.min(per_run)))
            put(system, rule, f"{name}_run_max", float(np.max(per_run)))

    # ---- A. corpus-level prevalence
    out("=" * 108)
    out("A. CORPUS-LEVEL PREVALENCE OF EACH VALUE")
    out("   Every metric is computed on the mean of a system's runs (the system's estimate).")
    out("   MAE: mean absolute error over the 19 values, percentage points (no interval, because percentile")
    out("        intervals of an absolute error near zero are biased upwards).")
    out("   Signed rel. error: mean of (est - gold) / gold over rare / frequent values, 95% bootstrap CI.")
    out(f"   Biased values: values whose (est - gold) differs from 0 (BH q < {ALPHA}), as under / over.")
    out("   Ratio: total estimated value mentions / gold mentions, 95% bootstrap CI.")
    out("=" * 108)
    srel = lambda m: (lambda v: ((v["all"] - gold["all"]) / gold["all"])[:, m].mean(1) * 100)
    ratio = lambda v: v["all"].sum(1) / gold["all"].sum(1)
    for system in SYSTEMS:
        out(f"\n{system}  ({len(rows[system][RULES[0]])} run(s))")
        out(f"  {'rule':12s} {'MAE (pp)':>9s} {'signed rel. err rare %':>26s} {'signed rel. err freq. %':>26s} {'biased under/over':>18s} {'ratio':>20s} {'macro-F1':>14s}")
        for r in RULES:
            vm = run_mean(system, r)
            diff = vm["all"] - gold["all"]
            mae_pt = np.abs(diff[0]).mean() * 100
            sig = bh(boot_p(diff)) < ALPHA
            under, over = int((sig & (diff[0] < 0)).sum()), int((sig & (diff[0] > 0)).sum())
            f1 = [d[f"F1 {r}"] for d in diag[system]] if r in CC_RULES else None
            f1s = mean_sd(f1) if f1 else "--"
            rat = ratio(vm)
            out(f"  {r:12s} {mae_pt:9.2f} {ci(srel(rare)(vm)):>26s} {ci(srel(~rare)(vm)):>26s} "
                f"{f'{under} / {over}':>18s} {ci(rat):>20s} {f1s:>14s}")
            put(system, r, "macro_f1", float(np.mean(f1)) if f1 else np.nan)
            put(system, r, "ratio", rat[0], rat)
            put(system, r, "prev_mae", mae_pt)
            put(system, r, "biased_under", under)
            put(system, r, "biased_over", over)
            put_spread(system, r, "ratio", [ratio(v)[0] for v in rows[system][r]])
            put_spread(system, r, "prev_mae", [np.abs(v["all"][0] - gold["all"][0]).mean() * 100 for v in rows[system][r]])

    # ---- B/C. contrasts
    def contrast_block(key: str, title: str, family: int) -> None:
        out()
        out("=" * 108)
        out(title)
        out(f"   Gold-significant contrasts (BH q < {ALPHA}): {gsig[key].sum()} of {family}.")
        out("   Sign ok: estimate points the same way as gold (gold-significant contrasts).")
        out("   Reproduced: estimate is itself significant with the right sign (gold-significant contrasts; BH over the family).")
        out(f"   Single test: estimate significant with the right sign when tested on its own (p < {ALPHA}, uncorrected).")
        out("           Unlike 'reproduced', it does not depend on how many other estimated contrasts pass BH.")
        out(f"   Biased: estimated contrast differs from the gold contrast beyond sampling noise (paired, BH q < {ALPHA}),")
        out("           over ALL contrasts in the family -- i.e. differential measurement error between the groups.")
        out("   Slope: least squares of estimated on gold contrasts through the origin (1 = faithful, < 1 = shrunk).")
        out("   All metrics on the mean of a system's runs; 'runs' gives the range of the slope over single runs.")
        rel = 1 - gold[key][1:].var(0).sum() / (gold[key][0] ** 2).sum()
        out(f"   Slope and rank rho are not shown for {', '.join(ANNOT_RULES)}: they add annotated documents other than")
        out("           the test documents, so they do not share the reference's sampling noise and regress to its mean.")
        out(f"           Reliability of these reference contrasts (1 - sampling variance / mean square): {rel:.2f}; an")
        out("           unbiased estimator from other documents would have a slope near it, not near 1.")
        out("=" * 108)
        gs = gsig[key]
        reversed_max = 0.0
        for system in SYSTEMS:
            out(f"\n{system}")
            hdr = f"  {'rule':12s} {'sign ok %':>10s} {'reproduced %':>13s} {'single %':>9s} {'biased %':>9s} {'MAE (pp)':>9s} {'slope':>20s} {'runs':>11s}"
            if key == "lpair":
                hdr += f" {'lang. rank rho':>20s}"
            out(hdr)
            for r in RULES:
                vm = run_mean(system, r)
                e0, g0 = vm[key][0], gold[key][0]
                p_est = boot_p(vm[key])
                esig = bh(p_est) < ALPHA
                same = np.sign(e0) == np.sign(g0)
                reversed_max = max(reversed_max, (esig & ~same)[gs].mean() * 100)
                st = [same[gs].mean() * 100, (esig & same)[gs].mean() * 100, ((p_est < ALPHA) & same)[gs].mean() * 100,
                      (bh(boot_p(vm[key] - gold[key])) < ALPHA).mean() * 100, np.abs(e0 - g0).mean() * 100]
                annot = r in ANNOT_RULES
                sl = slope(gold[key], vm[key])
                per_run = [slope(gold[key], v[key])[0] for v in rows[system][r]]
                rng_txt = f"{min(per_run):.2f}-{max(per_run):.2f}" if len(per_run) > 1 and not annot else "--"
                line = (f"  {r:12s} {st[0]:10.1f} {st[1]:13.1f} {st[2]:9.1f} {st[3]:9.1f} {st[4]:9.2f} "
                        f"{'--' if annot else ci(sl):>20s} {rng_txt:>11s}")
                for name, value in zip(["sign_ok", "reproduced", "detected", "biased", "mae"], st):
                    put(system, r, f"{key}_{name}", value)
                if not annot:
                    put(system, r, f"{key}_slope", sl[0], sl)
                    put_spread(system, r, f"{key}_slope", per_run)
                put_spread(system, r, f"{key}_mae", [np.abs(v[key][0] - g0).mean() * 100 for v in rows[system][r]])
                put_spread(system, r, f"{key}_biased", [(bh(boot_p(v[key] - gold[key])) < ALPHA).mean() * 100
                                                        for v in rows[system][r]])
                if key == "lpair":
                    rho = rank_corr(gold["lang"], vm["lang"])
                    line += f" {'--' if annot else ci(rho):>20s}"
                    if not annot:
                        put(system, r, "lang_rank_rho", rho[0], rho)
                out(line)
        out(f"\n   Largest share of gold-significant contrasts reproduced with the WRONG sign, any system/rule: {reversed_max:.1f}%")

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
    out("   At 0 / at 1: raw estimate below 0 or above 1 before clipping (EM: solution on the boundary;")
    out("   PPI: estimates below 0 or above 1, which are kept, because clipping would bias them).")
    out("=" * 108)
    for system in SYSTEMS:
        out(f"\n{system}")
        for r in QUANT_RULES + GROUP_RULES + ["PPI@prev", "PPI-Platt"]:
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
        pc = np.array(platt_check[system])
        out(f"  Platt refit check (original sample): max |histogram fit - exact fit| of the parameters = "
            f"{pc[:, 0].max():.1e}; max |binned - exact| PCC-Platt count per document = {pc[:, 1].max():.1e}")
        vm = run_mean(system, "CC@prev")
        fits = {k: slope_intercept(gold[k], vm[k]) for k in ["genre", "lpair"]}
        out(f"  CC@prev slope with an intercept (diagnostic): genre {fits['genre'][0]:.2f} (intercept "
            f"{fits['genre'][1] * 100:+.2f} pp), language pairs {fits['lpair'][0]:.2f} (intercept {fits['lpair'][1] * 100:+.2f} pp)")
    empty = [(k, j) for k in ["genre", "lpair"] for j in range(len(gold_p[k]))
             if gold_p[k][j] != boot_p(gold[k])[j]]
    out(f"\n  Reference contrasts with no mentions in one group, tested with Fisher's exact test instead of the "
        f"bootstrap: {sum(k == 'genre' for k, _ in empty)} genre, {sum(k == 'lpair' for k, _ in empty)} language-pair; "
        f"significant after BH: genre {[VALUES[j] for k, j in empty if k == 'genre' and gsig['genre'][j]]}, "
        f"language pairs {sum(1 for k, j in empty if k == 'lpair' and gsig['lpair'][j])}")
    out("  Platt-G: values with no positive validation sentence in a group (that group's map then predicts about 0): "
        + ("; ".join(f"{g}: {', '.join(v)}" for g, v in no_pos.items() if v) or "none"))

    # ---- G. why adjusted counts with pooled error rates fail between genres
    out()
    out("=" * 108)
    out("G. DO ERROR RATES DIFFER BETWEEN GENRES? (CC@prev decisions on test; mean over runs)")
    out("   Adjusted counts assume the same tpr and fpr in every group (prior shift only). Here: Spearman")
    out("   correlation, across the 19 values, between the gold manifesto-news contrast and the manifesto-news")
    out("   gap in tpr / in fpr. Positive = the classifier fires more, correctly and falsely, in the genre where")
    out("   the value is common, which inflates raw contrasts and breaks the prior-shift assumption.")
    out("=" * 108)
    def mean_rates(system: str) -> dict:
        """Error rates per split and group, mean over the system's runs (NaN where a group has no positive; the same in every run)."""
        runs = err_rates[system]
        return {sp: {g: np.mean([r[sp][g] for r in runs], axis=0) for g in runs[0][sp]} for sp in runs[0]}

    gold_genre_v = (gold_v_doc[man_v].sum(0) / nsent_v[man_v].sum() - gold_v_doc[news_v].sum(0) / nsent_v[news_v].sum())
    for system in SYSTEMS:
        m = np.array(mechanism[system])
        out(f"  {system:17s} rho(contrast, tpr gap) {m[:, 0].mean():+.2f}   rho(contrast, fpr gap) {m[:, 1].mean():+.2f}   "
            f"fpr gap has the contrast's sign for {m[:, 2].mean() * 100:.0f}% of values")
        mr = mean_rates(system)["validation"]
        gt_, gf_ = mr["manifestos"][0] - mr["news"][0], mr["manifestos"][1] - mr["news"][1]
        ok = np.isfinite(gt_)
        out(f"  {'':17s} on validation:  rho {spearmanr(gold_genre_v[ok], gt_[ok])[0]:+.2f} (tpr gap)   "
            f"{spearmanr(gold_genre_v, gf_)[0]:+.2f} (fpr gap)   ({ok.sum()} values with manifesto positives)")

    # error-rate table (AE item 8): the main system, per value
    mr = mean_rates(SYSTEMS[0])
    out(f"\n  Error rates of the prevalence-matched decisions per value, {SYSTEMS[0]}, mean over runs, % (tpr / fpr):")
    out(f"  {'value':28s} {'validation: manifestos':>23s} {'news':>13s} {'test: manifestos':>18s} {'news':>13s} "
        f"{'test, languages: tpr range':>27s} {'fpr range':>13s}")
    pct = lambda x: "--" if not np.isfinite(x) else f"{x * 100:.0f}"
    for j, val in enumerate(VALUES):
        cells = [f"{pct(mr[sp][g][0, j]):>4s} / {pct(mr[sp][g][1, j]):>4s}" for sp in ["validation", "test"]
                 for g in ["manifestos", "news"]]
        lt = np.array([mr["test"][l][0, j] for l in langs]); lf = np.array([mr["test"][l][1, j] for l in langs])
        out(f"  {val:28s} {cells[0]:>23s} {cells[1]:>13s} {cells[2]:>18s} {cells[3]:>13s} "
            f"{np.nanmin(lt) * 100:13.0f}-{np.nanmax(lt) * 100:<4.0f} {np.nanmin(lf) * 100:9.1f}-{np.nanmax(lf) * 100:<4.1f}")
    out("  '--': no positive sentence in the group. Full table, every system and language: measurement-error-rates.csv")

    # ---- H. accuracy and coverage (step 5.2, AE item 2)
    out()
    out("=" * 108)
    out("H. ACCURACY AND COVERAGE OF TOTALS AND CONTRASTS (mean of a system's runs)")
    out("   Signed: totals, mean of (estimate - reference); contrasts, mean of (estimate - reference) x sign(reference),")
    out("           so below 0 means differences shrink toward zero and above 0 that they grow. Percentage points.")
    out("   MAE, RMSE: mean absolute and root mean squared error, pp.")
    out("   Cover: share of quantities whose error (estimate - reference, same bootstrap rows) is within sampling noise")
    out(f"          (p >= {ALPHA}, single test), %. Pairing keeps the noise the estimate shares with the reference out of the")
    out("          interval and the noise it does not share in, so the share is comparable across estimators; about 95%")
    out("          for an estimator whose errors are only sampling noise.")
    out(f"   {', '.join(ANNOT_RULES)} use annotated documents other than the test documents: their errors include the")
    out("   reference's own sampling noise, and on contrasts they regress to its mean (negative signed error even when")
    out("   unbiased). A study that annotates a sample of its own documents would see smaller errors for them.")
    out("   Sig. / other: MAE of the contrasts that are significant in the reference, and of the others.")
    out("   Totals: 19 values; genre: 19 manifesto-news contrasts; language pairs: 684 contrasts.")
    out("=" * 108)

    def accuracy(est: np.ndarray, ref: np.ndarray, sig: np.ndarray | None = None) -> dict:
        d = est[0] - ref[0]
        m = {"signed": (d if sig is None else d * np.sign(ref[0])).mean() * 100,
             "rmse": np.sqrt((d ** 2).mean()) * 100, "cover": (boot_p(est - ref) >= ALPHA).mean() * 100}
        if sig is not None:
            m["mae_refsig"], m["mae_refother"] = np.abs(d[sig]).mean() * 100, np.abs(d[~sig]).mean() * 100
        return m

    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'':12s} | {'totals':^25s} | {'manifesto vs. news':^37s} | {'language pairs':^37s}")
        out(f"  {'rule':12s} | {'signed':>7s} {'MAE':>5s} {'RMSE':>5s} {'cover':>5s} | {'signed':>7s} {'MAE':>5s} {'RMSE':>5s} "
            f"{'cover':>5s} {'sig.':>5s} {'other':>5s} | {'signed':>7s} {'MAE':>5s} {'RMSE':>5s} {'cover':>5s} {'sig.':>5s} {'other':>5s}")
        for r in RULES:
            vm = run_mean(system, r)
            acc = {"prev": accuracy(vm["all"], gold["all"]), "genre": accuracy(vm["genre"], gold["genre"], gsig["genre"]),
                   "lpair": accuracy(vm["lpair"], gold["lpair"], gsig["lpair"])}
            mae = {k: np.abs(vm[k2][0] - gold[k2][0]).mean() * 100 for k, k2 in [("prev", "all"), ("genre", "genre"), ("lpair", "lpair")]}
            cells = [f"{acc['prev']['signed']:+7.2f} {mae['prev']:5.2f} {acc['prev']['rmse']:5.2f} {acc['prev']['cover']:5.0f}"]
            for k in ["genre", "lpair"]:
                a = acc[k]
                cells.append(f"{a['signed']:+7.2f} {mae[k]:5.2f} {a['rmse']:5.2f} {a['cover']:5.0f} "
                             f"{a['mae_refsig']:5.2f} {a['mae_refother']:5.2f}")
            out(f"  {r:12s} | " + " | ".join(cells))
            for k, m in acc.items():
                for name, value in m.items():
                    put(system, r, f"{k}_{name}", value)

    # ---- I. totals per group (step 5.2, AE item 1)
    out()
    out("=" * 108)
    out("I. TOTALS PER GROUP (every rule fitted on all validation data; under shift: shift-tests.txt)")
    out("   Total err: |estimated / reference value mentions in the group - 1|, %; MAE over the 19 values, pp.")
    out("   Languages: mean (max) over the nine.")
    out("=" * 108)
    total_err = lambda e, g: abs(e[0].sum() / g[0].sum() - 1) * 100
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'rule':12s} {'manifestos: total err':>22s} {'MAE':>6s} {'news: total err':>16s} {'MAE':>6s} "
            f"{'languages: total err':>24s} {'MAE':>6s}")
        for r in RULES:
            vm = run_mean(system, r)
            te = {g: total_err(vm[g], gold[g]) for g in ["man", "news"]}
            te_lang = [total_err(vm["lang"][:, i], gold["lang"][:, i]) for i in range(len(langs))]
            gm = {g: np.abs(vm[g][0] - gold[g][0]).mean() * 100 for g in ["man", "news", "lang"]}
            out(f"  {r:12s} {te['man']:22.1f} {gm['man']:6.2f} {te['news']:16.1f} {gm['news']:6.2f} "
                f"{np.mean(te_lang):16.1f} ({max(te_lang):5.1f}) {gm['lang']:6.2f}")
            for name, value in [("man_total_err", te["man"]), ("news_total_err", te["news"]), ("lang_total_err", np.mean(te_lang)),
                                ("lang_total_err_max", max(te_lang)), ("man_mae", gm["man"]), ("news_mae", gm["news"]),
                                ("lang_mae", gm["lang"])]:
                put(system, r, name, value)

    # ---- J. language contrasts on value composition (step 5.2, AE item 5)
    out()
    out("=" * 108)
    out("J. LANGUAGE CONTRASTS ON VALUE COMPOSITION (each value's share of all value mentions in the language)")
    out("   Each language was annotated by its own team, and the share of sentences with a value ranges from about")
    out("   30% to 86%. Shares of all mentions remove those differences in labeling rate, so a contrast here compares")
    out("   which values a language's texts stress, not how many sentences its team labeled. Estimates are turned")
    out("   into shares the same way (negative PPI estimates set to 0 first). 684 contrasts, as in C.")
    out("   Slope, MAE, single test, cover: as in C and H. Prevalence columns repeat C for comparison.")

    def composition(lp: np.ndarray) -> np.ndarray:
        lp = np.clip(lp, 0.0, None)
        tot = lp.sum(2, keepdims=True)
        return np.divide(lp, tot, out=np.zeros_like(lp), where=tot > 0)

    def comp_pairs(lp: np.ndarray) -> np.ndarray:
        c = composition(lp)
        return np.stack([c[:, i] - c[:, j] for i, j in pairs], axis=1).reshape(B + 1, -1)

    gold_comp = comp_pairs(gold["lang"])
    comp_sig = bh(boot_p(gold_comp)) < ALPHA
    rel_c = 1 - gold_comp[1:].var(0).sum() / (gold_comp[0] ** 2).sum()
    out(f"   Reference: {comp_sig.sum()} of {len(comp_sig)} composition contrasts significant (BH q < {ALPHA}); "
        f"reliability {rel_c:.2f}.")
    out("=" * 108)
    for system in SYSTEMS:
        out(f"\n{system}")
        out(f"  {'rule':12s} | {'composition: slope':>26s} {'MAE':>6s} {'single %':>9s} {'cover':>6s} | "
            f"{'prevalence: slope':>18s} {'MAE':>6s} {'single %':>9s}")
        for r in RULES:
            vm = run_mean(system, r)
            ec = comp_pairs(vm["lang"])
            d0 = ec[0] - gold_comp[0]
            single = ((boot_p(ec) < ALPHA) & (np.sign(ec[0]) == np.sign(gold_comp[0])))[comp_sig].mean() * 100
            cover = (boot_p(ec - gold_comp) >= ALPHA).mean() * 100
            annot = r in ANNOT_RULES
            sl = slope(gold_comp, ec)
            sl_p = slope(gold["lpair"], vm["lpair"])
            sm = summary[(system, r)]
            out(f"  {r:12s} | {'--' if annot else ci(sl):>26s} {np.abs(d0).mean() * 100:6.2f} {single:9.1f} {cover:6.0f} | "
                f"{'--' if annot else f'{sl_p[0]:.2f}':>18s} {sm['lpair_mae']:6.2f} {sm['lpair_detected']:9.1f}")
            if not annot:
                put(system, r, "comp_slope", sl[0], sl)
            put(system, r, "comp_mae", np.abs(d0).mean() * 100)
            put(system, r, "comp_detected", single)
            put(system, r, "comp_cover", cover)

    # ---- CSVs
    RES.mkdir(exist_ok=True)
    rate_rows = []
    for system in SYSTEMS:
        for sp, groups in mean_rates(system).items():
            for g, a in groups.items():
                for j, val in enumerate(VALUES):
                    rate_rows.append({"system": system, "split": sp, "group": g, "value": val, "tpr": a[0, j],
                                      "fpr": a[1, j], "positives": a[2, j], "negatives": a[3, j]})
    pd.DataFrame(rate_rows).to_csv(RES / "measurement-error-rates.csv", index=False, float_format="%.5f")
    per_value, genre_rows, lang_rows = [], [], []
    for system in SYSTEMS:
        genre_sig = {r: bh(boot_p(run_mean(system, r)["genre"])) < ALPHA for r in RULES}
        for j, val in enumerate(VALUES):
            row = {"system": system, "value": val, "gold_prev": gold["all"][0, j],
                   "gold_lo": np.percentile(gold["all"][1:, j], 2.5), "gold_hi": np.percentile(gold["all"][1:, j], 97.5)}
            grow = {"system": system, "value": val, "gold_diff": gold["genre"][0, j],
                    "gold_lo": np.percentile(gold["genre"][1:, j], 2.5), "gold_hi": np.percentile(gold["genre"][1:, j], 97.5),
                    "gold_sig": bool(gsig["genre"][j])}
            for r in RULES:
                m = run_mean(system, r)["all"][:, j]
                row[r], row[f"{r}_lo"], row[f"{r}_hi"] = m[0], *np.percentile(m[1:], [2.5, 97.5])
                grow[r] = run_mean(system, r)["genre"][0, j]
                grow[f"{r}_sig"] = bool(genre_sig[r][j])
            per_value.append(row); genre_rows.append(grow)
        lp_sig = {r: bh(boot_p(run_mean(system, r)["lpair"])) < ALPHA for r in RULES}
        for k, (i, jl) in enumerate(pairs):
            for j, val in enumerate(VALUES):
                idx = k * len(VALUES) + j
                lrow = {"system": system, "lang_a": langs[i], "lang_b": langs[jl], "value": val,
                        "gold_diff": gold["lpair"][0, idx], "gold_sig": bool(gsig["lpair"][idx])}
                for r in RULES:
                    lrow[r] = run_mean(system, r)["lpair"][0, idx]
                    lrow[f"{r}_sig"] = bool(lp_sig[r][idx])
                lang_rows.append(lrow)
    pd.DataFrame(per_value).to_csv(RES / "measurement-per-value.csv", index=False, float_format="%.5f")
    pd.DataFrame(genre_rows).to_csv(RES / "measurement-genre-contrasts.csv", index=False, float_format="%.5f")
    pd.DataFrame(lang_rows).to_csv(RES / "measurement-language-contrasts.csv", index=False, float_format="%.5f")
    pd.DataFrame(list(summary.values())).to_csv(RES / "measurement-summary.csv", index=False, float_format="%.5f")
    (RES / "measurement-analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RES / 'measurement-analysis.txt'} and measurement-*.csv")


if __name__ == "__main__":
    main()
