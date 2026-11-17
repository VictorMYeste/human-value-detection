#!/usr/bin/env python3
"""
Classification results under one protocol: decision rules, auxiliary inputs, ensembles
and presence gating.

The reviewers of the previous submission raised five objections that cached
predictions can answer:
  (i)   only one global threshold was tuned; per-label thresholds were not tried;
  (ii)  the ensemble gain was shown for one seed only;
  (iii) features were judged with seed tests but the ensemble with a sentence bootstrap;
  (iv)  the presence-gating comparison changed the value classifier as well as the gate;
  (v)   presence F1 was reported without the trivial all-positive baseline.

This script answers all five with a single protocol:
  * every threshold is fitted on validation and applied unchanged to test, on the same
    grid (0.01-0.99) for every system;
  * every system is run for seeds 42, 7 and 1701, and ensembles are rebuilt within each seed;
  * every comparison uses the three-seed mean and the paired, stratified document
    bootstrap of measurement_analysis.py (B = 2000; documents resampled within source
    language x genre strata; identical resamples), with Benjamini-Hochberg correction
    within each family, and reports whether the difference keeps its sign in every seed.

Decision rules:
  t=0.5        the default threshold;
  global t*    one threshold that maximises validation macro-F1;
  label t*     per-label thresholds that maximise each label's validation F1 (SCut, Yang 2001);
  label prev   per-label thresholds that reproduce each label's validation prevalence
               (PCut-style; the CC@prev rule of measurement_analysis.py).

Sections:
  A  every decision rule on every system;
  B  paired comparisons: rules, auxiliary inputs, ensembles, presence gate;
  C  presence detection against trivial baselines;
  D  the same value classifier with and without the presence gate.

Usage:
    python3 threshold_schemes.py
Writes results/threshold-schemes.txt, results/classification-table.csv,
results/classification-comparisons.csv and results/presence-diagnostics.csv.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from analyze_provenance import GRID as PUBLISHED_GRID, KEY, VALUES, load_gold
from measurement_analysis import (ALPHA, ENS_MEMBERS, SEEDS, bh, boot_p, document_design,
                                  global_threshold, label_f1_thresholds, load_probs, macro_f1,
                                  prevalence_thresholds)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PRESENCE_OUT = HERE / ".." / "presence" / "output"

GATE_GRID = np.round(np.arange(1, 20) * 0.05, 2)      # candidate gate thresholds 0.05-0.95
PUBLISHED_GATE = 0.10
GATE_MODEL = "Previous-Sentences-2-Lex-LIWC-22"       # the published presence gate
PRESENCE_MODELS = {"text-only": "Baseline", "Prev-2 + LIWC-22 (gate)": GATE_MODEL}

RULES = ["t=0.5", "global t*", "label t*", "label prev"]
SINGLE = {  # system -> (output file stem, RoBERTa naming)
    "DeBERTa": ("Baseline", False),
    "DeBERTa + LIWC-22": ("Lex-LIWC-22", False),
    "DeBERTa + Prev-2": ("Previous-Sentences-2", False),
    "DeBERTa + MJD": ("Lex-MJD", False),
    "DeBERTa + BERTopic": ("TD-BERTopic", False),
    "RoBERTa": ("RoBERTa-Baseline", True),
}
FEATURE_ENS = "Feature ensemble"      # Baseline + LIWC-22 + Prev-2, soft vote, within seed
SEED_ENS = "Seed ensemble"            # text-only DeBERTa, seeds 42 + 7 + 1701, soft vote
GATE_SYSTEMS = ["DeBERTa + hard gate 0.10", "DeBERTa + hard gate tuned", "DeBERTa + soft gate"]


# ----------------------------------------------------------------------------- loading
@lru_cache(maxsize=None)
def probs(name: str, seed: int, split: str, roberta: bool = False) -> np.ndarray:
    p = load_probs(name, seed, split, roberta)
    p.setflags(write=False)
    return p


@lru_cache(maxsize=None)
def presence_probs(name: str, seed: int, split: str) -> np.ndarray:
    gold = load_gold("validation" if split == "val" else "test")
    tag = "" if seed == 42 else f"-s{seed}"
    pred = pd.read_csv(PRESENCE_OUT / f"{name}{tag}-{split}.tsv", sep="\t")[KEY + ["Presence"]]
    merged = gold[KEY].merge(pred, on=KEY, how="left", validate="1:1")
    if merged["Presence"].isna().any():
        raise ValueError(f"presence {name} seed {seed} {split}: predictions missing for some sentences")
    p = merged["Presence"].to_numpy(dtype=float)
    p.setflags(write=False)
    return p


def realisations(system: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, validation probabilities, test probabilities) for each run of a system."""
    if system in SINGLE:
        stem, roberta = SINGLE[system]
        return [(f"seed {s}", probs(stem, s, "val", roberta), probs(stem, s, "test", roberta)) for s in SEEDS]
    if system == FEATURE_ENS:
        return [(f"seed {s}",
                 np.mean([probs(m, s, "val") for m in ENS_MEMBERS], axis=0),
                 np.mean([probs(m, s, "test") for m in ENS_MEMBERS], axis=0)) for s in SEEDS]
    if system == SEED_ENS:
        return [("seeds " + "+".join(map(str, SEEDS)),
                 np.mean([probs("Baseline", s, "val") for s in SEEDS], axis=0),
                 np.mean([probs("Baseline", s, "test") for s in SEEDS], axis=0))]
    raise KeyError(system)


# ----------------------------------------------------------------------------- thresholds
# global_threshold, label_f1_thresholds and prevalence_thresholds come from measurement_analysis,
# so both scripts apply exactly the same rules on the same grid (0.01-0.99).
def fit_thresholds(pv: np.ndarray, yv: np.ndarray) -> dict[str, np.ndarray]:
    k = pv.shape[1]
    return {
        "t=0.5": np.full(k, 0.5),
        "global t*": np.full(k, global_threshold(pv, yv)),
        "label t*": label_f1_thresholds(pv, yv),
        "label prev": prevalence_thresholds(pv, yv),
    }


# ----------------------------------------------------------------------------- scoring
class Scorer:
    """Test macro-F1 and predicted/gold mention ratio on the original sample and every resample."""

    def __init__(self, design: dict, yt: np.ndarray):
        self.S, self.W, self.yt = design["S"], design["W"], yt
        self.gold_mentions = self.W @ np.asarray(self.S @ yt).sum(1)

    def _boot(self, x: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(self.S @ x)

    def f1(self, pb: np.ndarray) -> np.ndarray:
        y = self.yt
        tp, fp, fn = self._boot(y & pb), self._boot((1 - y) & pb), self._boot(y & (1 - pb))
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros(den.shape), where=den > 0).mean(1)
        assert abs(f1[0] - macro_f1(y, pb)) < 1e-12, "bootstrap row 0 must equal the test macro-F1"
        return f1

    def ratio(self, pb: np.ndarray) -> np.ndarray:
        return (self.W @ np.asarray(self.S @ pb).sum(1)) / self.gold_mentions

    def record(self, label: str, pb: np.ndarray, thr, **extra) -> dict:
        return {"label": label, "f1": self.f1(pb), "ratio": self.ratio(pb), "thr": np.atleast_1d(thr), **extra}


def mean_rows(recs: list[dict], key: str) -> np.ndarray:
    return np.mean([r[key] for r in recs], axis=0)


def ci_text(x: np.ndarray, nd: int = 3) -> str:
    lo, hi = np.percentile(x[1:], [2.5, 97.5])
    return f"{x[0]:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def thr_text(thr: np.ndarray) -> str:
    if np.ptp(thr) == 0:
        return f"{thr[0]:.2f}"
    return f"median {np.median(thr):.2f} ({thr.min():.2f}-{thr.max():.2f})"


def compare(a: list[dict], b: list[dict]) -> dict:
    """Paired bootstrap of (mean B) - (mean A) test macro-F1."""
    d = mean_rows(b, "f1") - mean_rows(a, "f1")
    lo, hi = np.percentile(d[1:], [2.5, 97.5])
    out = {"delta": d[0], "lo": lo, "hi": hi, "p": float(boot_p(d[:, None])[0]),
           "per_seed": "", "same_sign": ""}
    if len(a) == len(b) == len(SEEDS):
        per = [rb["f1"][0] - ra["f1"][0] for ra, rb in zip(a, b)]
        out["per_seed"] = " ".join(f"{x:+.4f}" for x in per)
        out["same_sign"] = "yes" if (all(x > 0 for x in per) or all(x < 0 for x in per)) else "no"
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s); lines.append(s)

    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    design = document_design(gt)
    sc = Scorer(design, yt)
    n_boot = design["W"].shape[0] - 1

    out("CLASSIFICATION UNDER ONE PROTOCOL -- ValueEval'24 English test split")
    out(f"{len(gt)} test sentences in {len(design['docs'])} documents; thresholds fitted on validation "
        f"({len(gv)} sentences), grid 0.01-0.99 for every system; seeds {', '.join(map(str, SEEDS))}.")
    out(f"Intervals and p-values: paired stratified document bootstrap, B = {n_boot}, "
        f"Benjamini-Hochberg within each family (alpha = {ALPHA}).")
    out()

    # ---- run every system x realisation x rule
    systems = list(SINGLE) + [FEATURE_ENS, SEED_ENS]
    recs: dict[str, dict[str, list[dict]]] = {}
    for system in systems:
        recs[system] = {r: [] for r in RULES}
        for label, pv, pt in realisations(system):
            th = fit_thresholds(pv, yv)
            for r in RULES:
                recs[system][r].append(sc.record(label, (pt >= th[r]).astype(int), th[r]))

    # ---- sanity checks against the published numbers
    pv42, pt42 = probs("Baseline", 42, "val"), probs("Baseline", 42, "test")
    t_pub = global_threshold(pv42, yv, np.array(PUBLISHED_GRID))
    f1_pub = macro_f1(yt, (pt42 >= t_pub).astype(int))
    ens42 = recs[FEATURE_ENS]["global t*"][0]
    out(f"Sanity: DeBERTa seed 42 on the published 0.05 grid: t* = {t_pub:.2f}, test macro-F1 {f1_pub:.4f} "
        f"(published 0.3115); feature ensemble seed 42: t* = {ens42['thr'][0]:.2f}, {ens42['f1'][0]:.4f} "
        f"(published 0.332).")
    out()

    # ---- A. rules x systems
    out("=" * 118)
    out("A. TEST MACRO-F1 AND COUNT BIAS FOR EVERY DECISION RULE AND SYSTEM")
    out("   macro-F1: mean over runs [95% bootstrap CI of the mean]; sd over seeds; per-seed values.")
    out("   pred/gold: predicted value mentions / gold value mentions (1 = no count bias), mean [95% CI].")
    out("=" * 118)
    table_rows = []
    for system in systems:
        n = len(recs[system]["t=0.5"])
        out(f"\n{system}  ({n} run{'s' if n > 1 else ''})")
        out(f"  {'rule':11s} {'macro-F1 [95% CI]':>24s} {'sd':>7s}  {'per run':24s} {'pred/gold [95% CI]':>22s}  thresholds (first run)")
        for r in RULES:
            rr = recs[system][r]
            f1s = [x["f1"][0] for x in rr]
            sd = f"{np.std(f1s, ddof=1):.4f}" if n > 1 else "--"
            out(f"  {r:11s} {ci_text(mean_rows(rr, 'f1')):>24s} {sd:>7s}  {' '.join(f'{v:.4f}' for v in f1s):24s} "
                f"{ci_text(mean_rows(rr, 'ratio'), 2):>22s}  {thr_text(rr[0]['thr'])}")
            for x in rr:
                lo, hi = np.percentile(x["f1"][1:], [2.5, 97.5])
                rlo, rhi = np.percentile(x["ratio"][1:], [2.5, 97.5])
                table_rows.append({"system": system, "rule": r, "run": x["label"],
                                   "macro_f1": x["f1"][0], "f1_lo": lo, "f1_hi": hi,
                                   "pred_gold_ratio": x["ratio"][0], "ratio_lo": rlo, "ratio_hi": rhi,
                                   "threshold_median": float(np.median(x["thr"])),
                                   "threshold_min": float(x["thr"].min()), "threshold_max": float(x["thr"].max())})

    # ---- C/D data: presence models and the gate (computed before the comparison section)
    zt = (yt.sum(1) > 0).astype(int)
    presence_rows = []
    for mlabel, mname in PRESENCE_MODELS.items():
        for seed in SEEDS:
            g = presence_probs(mname, seed, "test")
            pred = (g >= 0.5).astype(int)
            presence_rows.append({
                "model": mlabel, "seed": seed,
                "F1@0.5": f1_score(zt, pred), "all-positive F1": f1_score(zt, np.ones_like(zt)),
                "accuracy@0.5": accuracy_score(zt, pred), "majority accuracy": max(zt.mean(), 1 - zt.mean()),
                "MCC@0.5": matthews_corrcoef(zt, pred), "ROC AUC": roc_auc_score(zt, g),
                "PR AUC": average_precision_score(zt, g), "prevalence": zt.mean(),
                f"value-bearing removed @{PUBLISHED_GATE:.2f}": ((g < PUBLISHED_GATE) & (zt == 1)).sum() / zt.sum(),
                f"value-free removed @{PUBLISHED_GATE:.2f}": ((g < PUBLISHED_GATE) & (zt == 0)).sum() / (zt == 0).sum(),
            })
    presence = pd.DataFrame(presence_rows)

    for gs in GATE_SYSTEMS:
        recs[gs] = {"global t*": []}
    for seed in SEEDS:
        pv, pt = probs("Baseline", seed, "val"), probs("Baseline", seed, "test")
        g_v, g_t = presence_probs(GATE_MODEL, seed, "val"), presence_probs(GATE_MODEL, seed, "test")
        label = f"seed {seed}"
        # hard gate at the published threshold
        mv, mt = pv * (g_v >= PUBLISHED_GATE)[:, None], pt * (g_t >= PUBLISHED_GATE)[:, None]
        t = global_threshold(mv, yv)
        recs["DeBERTa + hard gate 0.10"]["global t*"].append(
            sc.record(label, (mt >= t).astype(int), t, gate=PUBLISHED_GATE))
        # hard gate with the gate threshold tuned on validation jointly with the value threshold
        best = (-1.0, None, None)
        for tg in GATE_GRID:
            mv = pv * (g_v >= tg)[:, None]
            t = global_threshold(mv, yv)
            s = macro_f1(yv, (mv >= t).astype(int))
            if s > best[0]:
                best = (s, tg, t)
        _, tg, t = best
        mt = pt * (g_t >= tg)[:, None]
        recs["DeBERTa + hard gate tuned"]["global t*"].append(sc.record(label, (mt >= t).astype(int), t, gate=tg))
        # soft gate: value probability times presence probability
        t = global_threshold(pv * g_v[:, None], yv)
        recs["DeBERTa + soft gate"]["global t*"].append(
            sc.record(label, ((pt * g_t[:, None]) >= t).astype(int), t, gate="soft"))
    for gs in GATE_SYSTEMS:
        for x in recs[gs]["global t*"]:
            lo, hi = np.percentile(x["f1"][1:], [2.5, 97.5])
            rlo, rhi = np.percentile(x["ratio"][1:], [2.5, 97.5])
            table_rows.append({"system": gs, "rule": "global t*", "run": x["label"],
                               "macro_f1": x["f1"][0], "f1_lo": lo, "f1_hi": hi,
                               "pred_gold_ratio": x["ratio"][0], "ratio_lo": rlo, "ratio_hi": rhi,
                               "threshold_median": float(x["thr"][0]), "threshold_min": float(x["thr"][0]),
                               "threshold_max": float(x["thr"][0]), "gate": x["gate"]})

    # ---- B. paired comparisons
    G, P = "global t*", "label prev"
    families = {
        "Decision rules": [
            *[(f"{s}: global t* vs t=0.5", (s, "t=0.5"), (s, G)) for s in ["DeBERTa", "RoBERTa", FEATURE_ENS, SEED_ENS]],
            *[(f"{s}: label t* vs global t*", (s, G), (s, "label t*")) for s in ["DeBERTa", "RoBERTa", FEATURE_ENS, SEED_ENS]],
            *[(f"{s}: label prev vs global t*", (s, G), (s, P)) for s in ["DeBERTa", "RoBERTa", FEATURE_ENS, SEED_ENS]],
        ],
        "Auxiliary inputs (vs text-only DeBERTa, same rule)": [
            (f"{s} [{r}]", ("DeBERTa", r), (s, r))
            for r in [G, P] for s in ["DeBERTa + LIWC-22", "DeBERTa + Prev-2", "DeBERTa + MJD", "DeBERTa + BERTopic"]
        ],
        "Encoders and ensembles (same rule)": [
            x for r in [G, P] for x in [
                (f"RoBERTa vs DeBERTa [{r}]", ("DeBERTa", r), ("RoBERTa", r)),
                (f"Feature ensemble vs DeBERTa [{r}]", ("DeBERTa", r), (FEATURE_ENS, r)),
                (f"Feature ensemble vs DeBERTa + LIWC-22 [{r}]", ("DeBERTa + LIWC-22", r), (FEATURE_ENS, r)),
                (f"Seed ensemble vs DeBERTa [{r}]", ("DeBERTa", r), (SEED_ENS, r)),
                (f"Feature ensemble vs seed ensemble [{r}]", (SEED_ENS, r), (FEATURE_ENS, r)),
            ]
        ],
        "Presence gate (same value classifier, vs no gate, global t*)": [
            (f"{gs} vs no gate", ("DeBERTa", G), (gs, G)) for gs in GATE_SYSTEMS
        ],
    }
    out()
    out("=" * 118)
    out("B. PAIRED COMPARISONS OF TEST MACRO-F1  (delta = B - A, three-seed means unless a system has one run)")
    out("   q: Benjamini-Hochberg-adjusted bootstrap p within the family. same sign: the delta has the same sign in all seeds.")
    out("=" * 118)
    comp_rows = []
    for fam, items in families.items():
        res = [compare(recs[a[0]][a[1]], recs[b[0]][b[1]]) for _, a, b in items]
        q = bh(np.array([x["p"] for x in res]))
        out(f"\n{fam}")
        out(f"  {'comparison':58s} {'delta [95% CI]':>28s} {'q':>7s} {'same sign':>9s}  per-seed deltas")
        for (lab, a, b), x, qq in zip(items, res, q):
            delta = f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
            out(f"  {lab:58s} {delta:>28s} {qq:7.3f} {x['same_sign'] or '--':>9s}  {x['per_seed']}")
            comp_rows.append({"family": fam, "comparison": lab, "A": f"{a[0]} [{a[1]}]", "B": f"{b[0]} [{b[1]}]",
                              **{k: x[k] for k in ["delta", "lo", "hi", "p"]}, "q": qq,
                              "same_sign": x["same_sign"], "per_seed_delta": x["per_seed"]})

    # ---- C. presence
    out()
    out("=" * 118)
    out("C. PRESENCE DETECTION (does a sentence express any value?) AGAINST TRIVIAL BASELINES, test split, mean ± sd over seeds")
    out("=" * 118)
    metrics = [c for c in presence.columns if c not in ("model", "seed")]
    for mlabel in PRESENCE_MODELS:
        sub = presence[presence["model"] == mlabel]
        out(f"\n{mlabel}")
        for m in metrics:
            vals = sub[m].to_numpy(float)
            per = " ".join(f"{v:.3f}" for v in vals)
            out(f"  {m:32s} {vals.mean():.3f} ± {vals.std(ddof=1):.3f}   (seeds: {per})")

    # ---- D. gate
    out()
    out("=" * 118)
    out("D. THE SAME VALUE CLASSIFIER (text-only DeBERTa) WITH AND WITHOUT THE PRESENCE GATE, global t* re-tuned on validation")
    out("=" * 118)
    out(f"  {'variant':28s} {'macro-F1 [95% CI]':>24s}  {'per seed':24s} {'pred/gold':>8s}  gate threshold / value threshold per seed")
    for gs in ["DeBERTa"] + GATE_SYSTEMS:
        rr = recs[gs][G]
        f1s = " ".join(f"{x['f1'][0]:.4f}" for x in rr)
        gates = "; ".join(f"{x.get('gate', 'none')} / {x['thr'][0]:.2f}" for x in rr)
        name = "no gate" if gs == "DeBERTa" else gs.replace("DeBERTa + ", "")
        out(f"  {name:28s} {ci_text(mean_rows(rr, 'f1')):>24s}  {f1s:24s} {mean_rows(rr, 'ratio')[0]:8.2f}  {gates}")

    # ---- write
    RES.mkdir(exist_ok=True)
    (RES / "threshold-schemes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(table_rows).to_csv(RES / "classification-table.csv", index=False, float_format="%.6f")
    pd.DataFrame(comp_rows).to_csv(RES / "classification-comparisons.csv", index=False, float_format="%.6f")
    presence.to_csv(RES / "presence-diagnostics.csv", index=False, float_format="%.6f")
    print(f"\nWrote {RES / 'threshold-schemes.txt'} and three CSVs.")


if __name__ == "__main__":
    main()
