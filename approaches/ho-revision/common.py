"""
Shared helpers for the Array revision of the higher-order (HO) paper (ARRAY-D-26-01445).

Everything here runs on cached validation/test predictions; nothing is retrained.
The threshold rules replicate the ones used to produce the submitted tables:

  * eq8 rule      core/evaluation.py:find_best_threshold -- per label, the threshold on
                  np.arange(0.10, 0.90, 0.01) with the highest recall among those whose
                  precision reaches the floor (0.40 in the paper); 0.5 if none does.
  * global sweep  approaches/*/eval-threshold.sh -- one threshold for all labels, the
                  first on 0.05, 0.10, ..., 1.00 whose validation macro-F1 (printed to
                  five decimals by eval.py) is strictly higher than the previous best.

The paired bootstrap follows core/eval_utils.py:bootstrap_f1_lower (sentences resampled
with replacement, B = 2000, lower bound = 5th percentile of the macro-F1 difference,
one-sided p = share of resamples with difference <= 0), but draws one set of resamples
and reuses it for every comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
APPROACHES = ROOT / "approaches"
RESULTS = Path(__file__).resolve().parent / "results"
ID = ["Text-ID", "Sentence-ID"]
SPLITS = {"val": "validation", "test": "test"}

PAIRS = {
    "Growth vs. Self-Protection": ("growth-vs-self-protection",
                                   ["Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance"]),
    "Social Focus vs. Personal Focus": ("social-vs-personal-focus",
                                        ["Social Focus", "Personal Focus"]),
    "Openness vs. Conservation": ("openness-vs-conservation",
                                  ["Openness to Change", "Conservation"]),
    "Self-Transcendence vs. Self-Enhancement": ("self-trans-vs-self-enh",
                                                ["Self-Transcendence", "Self-Enhancement"]),
}

# Output folder of each HO slice and the parent HO label that gates it
SLICES = {
    "Growth": ("growth", "Growth Anxiety-Free"),
    "Self-Protection": ("self-protection", "Self-Protection Anxiety-Avoidance"),
    "Social Focus": ("social-focus", "Social Focus"),
    "Personal Focus": ("personal-focus", "Personal Focus"),
    "Openness": ("openness", "Openness to Change"),
    "Conservation": ("conservation", "Conservation"),
    "Self-Transcendence": ("self-trans", "Self-Transcendence"),
    "Self-Enhancement": ("self-enh", "Self-Enhancement"),
}

# Gated configuration reported in Table 4 of the submitted manuscript, per slice
TABLE4_GATED = {
    "Growth": "1_Baseline_0.5_Baseline",
    "Self-Protection": "1_Baseline_0.5_Baseline",
    "Social Focus": "1_TD-NMF_0.5_Lex-MJD",
    "Personal Focus": "1_NER_0.19_Lex-MJD",
    "Openness": "1_Lex-LIWC-22_LingFeat_0.5_NER_TD-LDA",
    "Conservation": "1_Lex-LIWC-22_LingFeat_0.1_Lex-Schwartz",
    "Self-Transcendence": "1_Lex-MFD-20_0.5_Baseline",
    "Self-Enhancement": "1_Lex-WorryWords_0.23_Lex-Schwartz",
}

EQ8_GRID = np.arange(0.10, 0.90, 0.01)          # exactly as in core/evaluation.py
GLOBAL_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)  # exactly as in eval-threshold.sh
# Grid of the revised main rule (per-label F1). It excludes 0 and 1, which would turn a
# label into an all-positive or all-negative predictor, and matches Paper 1's re-analysis.
LABEL_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

_gold: dict[str, pd.DataFrame] = {}


def gold(split: str) -> pd.DataFrame:
    if split not in _gold:
        _gold[split] = pd.read_csv(ROOT / "data" / f"{SPLITS[split]}-english" / "labels-cat.tsv", sep="\t")
    return _gold[split]


def labels_of(path: Path) -> list[str]:
    return [c for c in pd.read_csv(path, sep="\t", nrows=1).columns if c not in ID]


def load(path: Path, labels: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    """Gold (bool) and scores (float) aligned to the gold file's sentence order."""
    g = gold(split)[ID + labels]
    p = pd.read_csv(path, sep="\t")[ID + labels]
    m = g.merge(p, on=ID, how="inner", suffixes=("_g", "_p"), validate="one_to_one")
    if len(m) != len(g):
        raise ValueError(f"{path}: {len(m)} rows matched, {len(g)} expected")
    y = m[[f"{l}_g" for l in labels]].to_numpy(float) > 0
    s = m[[f"{l}_p" for l in labels]].to_numpy(float)
    return y, s


def _counts(y: np.ndarray, s: np.ndarray, grid: np.ndarray):
    """TP, FP, FN for one label at every threshold in grid."""
    pred = s[:, None] >= grid[None, :]
    yy = y[:, None]
    return (pred & yy).sum(0), (pred & ~yy).sum(0), (~pred & yy).sum(0)


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def eq8_threshold(y: np.ndarray, s: np.ndarray, floor: float = 0.40) -> tuple[float, bool]:
    """Per-label rule of Eq. 8 as implemented; returns (threshold, fell_back_to_0.5)."""
    tp, fp, fn = _counts(y, s, EQ8_GRID)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    ok = (prec >= floor) & (rec > 0)
    if not ok.any():
        return 0.5, True
    r = np.where(ok, rec, -1.0)
    return float(EQ8_GRID[int(np.argmax(r))]), False   # argmax keeps the first maximum


def label_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Per-label threshold on 0.01-0.99 that maximises that label's validation F1.

    Ties go to the threshold closest to 0.5, so a flat F1 curve does not push the
    threshold to an edge of the grid."""
    tp, fp, fn = _counts(y, s, LABEL_GRID)
    f = np.round(_f1(tp, fp, fn), 10)
    best = np.flatnonzero(f == f.max())
    return float(LABEL_GRID[best[np.argmin(np.abs(LABEL_GRID[best] - 0.5))]])


def label_f1_thresholds(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.array([label_f1_threshold(y[:, i], s[:, i]) for i in range(y.shape[1])])


def macro_f1(y: np.ndarray, s: np.ndarray, th) -> float:
    th = np.broadcast_to(np.asarray(th, float), (y.shape[1],))
    pred = s >= th[None, :]
    tp = (pred & y).sum(0); fp = (pred & ~y).sum(0); fn = (~pred & y).sum(0)
    return float(_f1(tp, fp, fn).mean())


def label_f1s(y: np.ndarray, s: np.ndarray, th) -> np.ndarray:
    th = np.broadcast_to(np.asarray(th, float), (y.shape[1],))
    pred = s >= th[None, :]
    return _f1((pred & y).sum(0), (pred & ~y).sum(0), (~pred & y).sum(0))


def global_sweep_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best_f, best_t = 0.0, None
    for t in GLOBAL_GRID:
        f = round(macro_f1(y, s, t), 5)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


RULES = ["fixed 0.5", "Eq. 8, floor 0.30", "Eq. 8, floor 0.40", "Eq. 8, floor 0.50",
         "label F1", "global F1"]


def fit_rule(rule: str, yv: np.ndarray, sv: np.ndarray) -> tuple[np.ndarray, int]:
    """Thresholds fitted on validation under one rule, and the number of 0.5 fallbacks."""
    L = yv.shape[1]
    if rule == "fixed 0.5":
        return np.full(L, 0.5), 0
    if rule.startswith("Eq. 8"):
        floor = float(rule.split()[-1])
        out = [eq8_threshold(yv[:, i], sv[:, i], floor) for i in range(L)]
        return np.array([t for t, _ in out]), sum(fb for _, fb in out)
    if rule == "label F1":
        return np.array([label_f1_threshold(yv[:, i], sv[:, i]) for i in range(L)]), 0
    if rule == "global F1":
        return np.full(L, global_sweep_threshold(yv, sv)), 0
    raise ValueError(rule)


class Bootstrap:
    """One set of sentence resamples of the test split, reused for every comparison."""

    def __init__(self, n: int, B: int = 2000, seed: int = 20261123, chunk: int = 250):
        rng = np.random.default_rng(seed)
        self.W = np.empty((B, n), dtype=np.float32)
        for a in range(0, B, chunk):
            idx = rng.integers(0, n, size=(min(chunk, B - a), n))
            for r in range(idx.shape[0]):
                self.W[a + r] = np.bincount(idx[r], minlength=n)

    def macro_f1s(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        tp = self.W @ (pred & y).astype(np.float32)
        fp = self.W @ (pred & ~y).astype(np.float32)
        fn = self.W @ (~pred & y).astype(np.float32)
        return _f1(tp, fp, fn).mean(1)

    def compare(self, y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> tuple[float, float]:
        """One-sided test of system 2 > system 1: (lower 95% bound of the difference, p)."""
        d = self.macro_f1s(y, pred2) - self.macro_f1s(y, pred1)
        return float(np.percentile(d, 5)), float(np.mean(d <= 0))

    def both_ways(self, y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict:
        """Difference 2 - 1 with its 90% interval and both one-sided p-values."""
        d = self.macro_f1s(y, pred2) - self.macro_f1s(y, pred1)
        return dict(lower95=float(np.percentile(d, 5)), upper95=float(np.percentile(d, 95)),
                    p_2_better=float(np.mean(d <= 0)), p_1_better=float(np.mean(d >= 0)))


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def mcnemar_positives(y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> list[float]:
    """Per-label exact McNemar on the gold positives, as core/eval_utils.per_label_mcnemar
    (NaN where the two systems never disagree); BH is applied by the caller."""
    from statsmodels.stats.contingency_tables import mcnemar
    out = []
    for i in range(y.shape[1]):
        m = y[:, i]
        c1, c2 = pred1[m, i], pred2[m, i]
        a, b, c = np.sum(c1 & c2), np.sum(c1 & ~c2), np.sum(~c1 & c2)
        out.append(np.nan if b + c == 0 else
                   mcnemar([[a, b], [c, m.sum() - a - b - c]], exact=True).pvalue)
    return out
