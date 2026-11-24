#!/usr/bin/env python3
"""
Main-text results rebuilt under one protocol (step 1.7 of the revision plan).

Protocol, applied to every system:
  * thresholds: per label, the value on 0.01-0.99 that maximises that label's validation
    F1 (Reviewer #2, comment 3c); fixed 0.5 and one global macro-F1 threshold are kept as
    comparison rules;
  * selection: within each family (Direct, Presence-gated, HO-gated), the configuration
    with the highest validation macro-F1 under its own validation thresholds;
  * test: the selected systems are evaluated once; differences are tested with a one-sided
    paired sentence bootstrap (B = 2000) in both directions, with Benjamini-Hochberg
    correction across the slices of a table, plus per-label exact McNemar on the gold
    positives (BH within the slice), as in the submitted paper.

Parts:
  A  HO detection by bipolar pair (Table 1): every detector variant; the effect of tuning.
  B  Presence-gated HO detection (Tables 2-3): the selected cascade against the selected
     Direct detector, with the in-gate validation score.
  C  HO->values hard gating (Tables 4, 6 'Single', 7 'Direct vs gated'): the selected
     Direct model against the selected gated pipeline in each slice, under each rule.

Usage:
    python3 approaches/ho-revision/rebuild_tables.py
Writes results/rebuild-candidates.csv, results/rebuild-selected.csv,
results/rebuild-selections.json and results/rebuild-tables.txt.
"""
import contextlib
import io
import json
import re
import sys

import numpy as np
import pandas as pd

from common import (APPROACHES, PAIRS, RESULTS, SLICES, Bootstrap, bh, global_sweep_threshold,
                    gold, label_f1_thresholds, label_f1s, labels_of, load, macro_f1,
                    mcnemar_positives)

DIRECT_MODELS = ["Baseline", "Lex-LIWC-22", "Lex-MJD", "Previous-Sentences-2", "TD-BERTopic"]
RULE_FNS = {
    "label F1": label_f1_thresholds,
    "global F1": lambda y, s: np.full(y.shape[1], global_sweep_threshold(y, s)),
    "fixed 0.5": lambda y, s: np.full(y.shape[1], 0.5),
}
MAIN = "label F1"
CASCADE = re.compile(r"^1_(?P<gate>.+)_(?P<tg>0\.\d+)_(?P<model>.+)-val\.tsv$")


def candidate(family, unit, name, labs, vf, tf, rule=MAIN, extra=None):
    yv, sv = load(vf, labs, "val")
    yt, st = load(tf, labs, "test")
    th = RULE_FNS[rule](yv, sv)
    return dict(family=family, unit=unit, name=name, rule=rule, labels=labs,
                val_f1=macro_f1(yv, sv, th), test_f1=macro_f1(yt, st, th),
                test_label_f1=label_f1s(yt, st, th), thresholds=th, yt=yt,
                pred=st >= th[None, :], **(extra or {}))


def pick(cands):
    return max(cands, key=lambda c: c["val_f1"])


def clean(c):
    return {k: (v.round(3).tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v)
            for k, v in c.items() if k not in ("yt", "pred")}


def part_a(boot, cand_rows, sel_rows, selections):
    print("=== A. HO detection by bipolar pair (Table 1); per-label F1 thresholds")
    for pair, (folder, labs) in PAIRS.items():
        out = APPROACHES / folder / "output"
        cands = [candidate("HO detector", pair, vf.name[:-8], labs, vf, out / f"{vf.name[:-8]}-test.tsv")
                 for vf in sorted(out.glob("*-val.tsv"))]
        cand_rows += [clean(c) for c in cands]
        best = pick(cands)
        base = next(c for c in cands if c["name"] == "Baseline")
        base_fx = candidate("HO detector", pair, "Baseline", labs, out / "Baseline-val.tsv",
                            out / "Baseline-test.tsv", rule="fixed 0.5")
        best_fx = candidate("HO detector", pair, best["name"], labs, out / f"{best['name']}-val.tsv",
                            out / f"{best['name']}-test.tsv", rule="fixed 0.5")
        t_base = boot.both_ways(base["yt"], base_fx["pred"], base["pred"])
        t_best = boot.both_ways(best["yt"], best_fx["pred"], best["pred"])
        selections.setdefault("ho_detector", {})[pair] = best["name"]
        sel_rows.append(dict(part="A", unit=pair, system=f"selected: {best['name']}", rule=MAIN,
                             val_f1=best["val_f1"], test_f1=best["test_f1"],
                             poles=best["test_label_f1"].round(3).tolist()))
        print(f"\n{pair}")
        print(f"  Baseline      fixed {base_fx['test_f1']:.3f} -> tuned {base['test_f1']:.3f}"
              f"  (diff {base['test_f1'] - base_fx['test_f1']:+.3f}, lower95 {t_base['lower95']:+.3f},"
              f" p {t_base['p_2_better']:.3f}; thresholds {base['thresholds'].round(2).tolist()})")
        print(f"  Selected on validation: {best['name']}  val {best['val_f1']:.3f}  test {best['test_f1']:.3f}"
              f"  poles {best['test_label_f1'].round(2).tolist()}  thresholds {best['thresholds'].round(2).tolist()}")
        print(f"    same detector at 0.5: {best_fx['test_f1']:.3f}  (tuning diff {best['test_f1'] - best_fx['test_f1']:+.3f},"
              f" lower95 {t_best['lower95']:+.3f}, p {t_best['p_2_better']:.3f})")
        ranked = sorted(cands, key=lambda c: -c["val_f1"])
        print("  all variants (val / test): " + "; ".join(
            f"{c['name']} {c['val_f1']:.3f}/{c['test_f1']:.3f}" for c in ranked))


def part_b(boot, cand_rows, sel_rows, selections):
    print("\n=== B. Presence-gated HO detection (Tables 2-3); per-label F1 thresholds")
    pres_cache = {}
    rows = []
    for pair, (folder, labs) in PAIRS.items():
        out = APPROACHES / f"p_{folder}" / "output"
        cands = []
        for vf in sorted(out.glob("1_*-val.tsv")):
            m = CASCADE.match(vf.name)
            if not m or "champion" in vf.name or re.search(r"-s\d+$", m["model"]):
                continue
            c = candidate("Presence cascade", pair, vf.name[:-8], labs, vf, out / f"{vf.name[:-8]}-test.tsv",
                          extra=dict(gate=m["gate"], tg=float(m["tg"]), model=m["model"]))
            cands.append(c)
        cand_rows += [clean(c) for c in cands]
        best = pick(cands)
        d_out = APPROACHES / folder / "output"
        dname = selections["ho_detector"][pair]
        direct = candidate("HO detector", pair, dname, labs, d_out / f"{dname}-val.tsv", d_out / f"{dname}-test.tsv")
        # In-gate validation score: only sentences whose Presence probability passes the gate
        g = best["gate"]
        if g not in pres_cache:
            pres_cache[g] = gold("val")[["Text-ID", "Sentence-ID"]].merge(
                pd.read_csv(APPROACHES / "presence" / "output" / f"{g}-val.tsv", sep="\t"),
                on=["Text-ID", "Sentence-ID"], how="left")["Presence"].fillna(0).to_numpy()
        keep = pres_cache[g] >= best["tg"]
        yv, sv = load(out / f"{best['name']}-val.tsv", labs, "val")
        in_gate = macro_f1(yv[keep], sv[keep], best["thresholds"])
        t = boot.both_ways(direct["yt"], best["pred"], direct["pred"])
        rows.append((pair, direct, best, in_gate, keep.mean(), t))
        selections.setdefault("presence_cascade", {})[pair] = best["name"]
    padj = bh([r[5]["p_2_better"] for r in rows])
    padj_rev = bh([r[5]["p_1_better"] for r in rows])
    print(f"\n{'pair':40s} {'Direct val/test':>16s} {'cascade val(in-gate)/test':>27s}  diff(D-C) lower95  p_D>C(BH)  p_C>D(BH)  selected cascade")
    for (pair, direct, best, in_gate, share, t), pa, pr in zip(rows, padj, padj_rev):
        print(f"{pair:40s} {direct['val_f1']:.3f}/{direct['test_f1']:.3f}       "
              f"{best['val_f1']:.3f} ({in_gate:.3f})/{best['test_f1']:.3f}        "
              f"{direct['test_f1'] - best['test_f1']:+.3f}  {t['lower95']:+.3f}   {pa:.3f}      {pr:.3f}   "
              f"{best['name']} (gate passes {share:.0%} of val)")
        sel_rows.append(dict(part="B", unit=pair, system=f"cascade: {best['name']}", rule=MAIN,
                             val_f1=best["val_f1"], val_in_gate=in_gate, test_f1=best["test_f1"],
                             direct=direct["name"], direct_test_f1=direct["test_f1"],
                             diff=direct["test_f1"] - best["test_f1"], lower95=t["lower95"],
                             p_direct_better_bh=pa, p_cascade_better_bh=pr))

    # Table 3: one fixed cascade (HO model Baseline, Presence gate Lex-LIWC-22_LingFeat),
    # varying only the gate threshold, with fixed and tuned HO thresholds.
    print("\n--- Table 3: HO model Baseline, Presence gate Lex-LIWC-22_LingFeat (test; HO thresholds fixed 0.5 / label F1)")
    print(f"{'pair':40s} {'Direct':>7s}   {'gate 0.5':>13s}   {'gate 0.1':>13s}")
    for pair, direct, *_ in rows:
        out = APPROACHES / f"p_{PAIRS[pair][0]}" / "output"
        cells = []
        for tg in ("0.5", "0.1"):
            name = f"1_Lex-LIWC-22_LingFeat_{tg}_Baseline"
            fx, tu = (candidate("Presence cascade", pair, name, PAIRS[pair][1], out / f"{name}-val.tsv",
                                out / f"{name}-test.tsv", rule=r) for r in ("fixed 0.5", MAIN))
            cells.append(f"{fx['test_f1']:.3f} / {tu['test_f1']:.3f}")
            sel_rows.append(dict(part="B3", unit=pair, system=name, rule="fixed 0.5 / " + MAIN,
                                 test_f1_fixed=fx["test_f1"], test_f1=tu["test_f1"],
                                 direct=direct["name"], direct_test_f1=direct["test_f1"]))
        print(f"{pair:40s} {direct['test_f1']:7.3f}   " + "   ".join(cells))


def part_c(boot, cand_rows, sel_rows, selections):
    print("\n=== C. HO->values hard gating by slice (Tables 4, 6, 7)")
    direct_dir = APPROACHES / "moral-values" / "output"
    out_by_rule = {}
    for rule in RULE_FNS:
        res = []
        for sl, (folder, _) in SLICES.items():
            out = APPROACHES / f"{folder}_moral-values" / "output"
            gated_files = [vf for vf in sorted(out.glob("1_*-val.tsv")) if CASCADE.match(vf.name)]
            labs = labels_of(out / f"{gated_files[0].name[:-8]}-test.tsv")
            dc = [candidate("Direct", sl, m, labs, direct_dir / f"{m}-val.tsv", direct_dir / f"{m}-test.tsv", rule)
                  for m in DIRECT_MODELS]
            gc = []
            for vf in gated_files:
                m = CASCADE.match(vf.name)
                gc.append(candidate("HO-gated", sl, vf.name[:-8], labs, vf, out / f"{vf.name[:-8]}-test.tsv", rule,
                                    extra=dict(gate=m["gate"], tg=float(m["tg"]), model=m["model"])))
            if rule == MAIN:
                cand_rows += [clean(c) for c in dc + gc]
            d, g = pick(dc), pick(gc)
            t = boot.both_ways(d["yt"], g["pred"], d["pred"])
            mc = mcnemar_positives(d["yt"], g["pred"], d["pred"])
            res.append(dict(slice=sl, d=d, g=g, t=t, mc=mc, n_gated=len(gc)))
        pa = bh([r["t"]["p_2_better"] for r in res])
        pr = bh([r["t"]["p_1_better"] for r in res])
        for r, a, b in zip(res, pa, pr):
            r["p_direct_bh"], r["p_gated_bh"] = a, b
        out_by_rule[rule] = res

    print("\n--- Main rule (per-label F1). Diff = Direct - gated on test.")
    print(f"{'slice':18s} {'#val':>4s} {'Direct (selected)':>30s} {'val':>6s} {'test':>6s} | "
          f"{'gated (selected)':>40s} {'val':>6s} {'test':>6s} | {'diff':>6s} {'low95':>6s} {'pD>G':>5s} {'pG>D':>5s} McNemar sig (BH)")
    for r in out_by_rule[MAIN]:
        d, g, t = r["d"], r["g"], r["t"]
        mc = [p for p in r["mc"] if not np.isnan(p)]
        mc_adj = bh(mc) if mc else []
        labs_sig = [lab for lab, p in zip([l for l, q in zip(d["labels"], r["mc"]) if not np.isnan(q)], mc_adj) if p < 0.05]
        print(f"{r['slice']:18s} {len(d['labels']):4d} {d['name']:>30s} {d['val_f1']:.3f} {d['test_f1']:.3f} | "
              f"{g['name']:>40s} {g['val_f1']:.3f} {g['test_f1']:.3f} | "
              f"{d['test_f1'] - g['test_f1']:+.3f} {t['lower95']:+.3f} {r['p_direct_bh']:.3f} {r['p_gated_bh']:.3f} "
              f"{len(labs_sig)}/{len(d['labels'])} {labs_sig}")
        sel_rows.append(dict(part="C", unit=r["slice"], rule=MAIN, direct=d["name"], direct_val=d["val_f1"],
                             direct_test=d["test_f1"], gated=g["name"], gated_val=g["val_f1"],
                             gated_test=g["test_f1"], diff=d["test_f1"] - g["test_f1"],
                             lower95=t["lower95"], p_direct_better_bh=r["p_direct_bh"],
                             p_gated_better_bh=r["p_gated_bh"], mcnemar_sig=labs_sig,
                             direct_thresholds=d["thresholds"].round(2).tolist(),
                             gated_thresholds=g["thresholds"].round(2).tolist()))
        selections.setdefault("direct", {})[r["slice"]] = d["name"]
        selections.setdefault("ho_gated", {})[r["slice"]] = g["name"]

    print("\n--- The same comparison under each rule (validation selection within each rule).")
    print(f"{'slice':18s} " + " | ".join(f"{rule:^34s}" for rule in RULE_FNS))
    print(f"{'':18s} " + " | ".join(f"{'Direct':>7s} {'gated':>7s} {'diff':>7s} {'sig':>9s}" for _ in RULE_FNS))
    for i, sl in enumerate(SLICES):
        cells = []
        for rule in RULE_FNS:
            r = out_by_rule[rule][i]
            sig = "Direct" if r["p_direct_bh"] < 0.05 else "gated" if r["p_gated_bh"] < 0.05 else "n.s."
            cells.append(f"{r['d']['test_f1']:7.3f} {r['g']['test_f1']:7.3f} {r['d']['test_f1'] - r['g']['test_f1']:+7.3f} {sig:>9s}")
        print(f"{sl:18s} " + " | ".join(cells))
    for rule in RULE_FNS:
        for r in out_by_rule[rule]:
            sel_rows.append(dict(part="C-rules", unit=r["slice"], rule=rule, direct=r["d"]["name"],
                                 direct_val=r["d"]["val_f1"], direct_test=r["d"]["test_f1"],
                                 gated=r["g"]["name"], gated_val=r["g"]["val_f1"], gated_test=r["g"]["test_f1"],
                                 diff=r["d"]["test_f1"] - r["g"]["test_f1"], lower95=r["t"]["lower95"],
                                 p_direct_better_bh=r["p_direct_bh"], p_gated_better_bh=r["p_gated_bh"]))
    for rule in RULE_FNS:
        res = out_by_rule[rule]
        print(f"  {rule:10s}: Direct higher in {sum(r['d']['test_f1'] > r['g']['test_f1'] for r in res)}/8,"
              f" significantly in {sum(r['p_direct_bh'] < 0.05 for r in res)}; gated significantly higher in"
              f" {sum(r['p_gated_bh'] < 0.05 for r in res)}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    boot = Bootstrap(len(gold("test")))
    cand_rows, sel_rows, selections = [], [], {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        part_a(boot, cand_rows, sel_rows, selections)
        part_b(boot, cand_rows, sel_rows, selections)
        part_c(boot, cand_rows, sel_rows, selections)
    text = buf.getvalue()
    pd.DataFrame(cand_rows).drop(columns=["labels"]).to_csv(RESULTS / "rebuild-candidates.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(RESULTS / "rebuild-selected.csv", index=False)
    (RESULTS / "rebuild-selections.json").write_text(json.dumps(selections, indent=2))
    (RESULTS / "rebuild-tables.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
