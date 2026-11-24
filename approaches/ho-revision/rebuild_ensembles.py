#!/usr/bin/env python3
"""
Ensembles rebuilt under the main protocol (step 1.8 of the revision plan; main-text
Tables 5-7).

What the submitted tables used (checked 23 Nov 2026):
  * 'Tr ens.' was not one method. Five slices used the 19-value Direct soft ensemble
    (LIWC-22 + Previous-Sentences-2 + Baseline, global threshold 0.29/0.30); Openness and
    Self-Enhancement used a Presence-gated ensemble; Self-Transcendence a gated ensemble.
  * The Transformer+LLM rows mixed voting modes (hard, soft, tuned) by slice.
  * LLM outputs are 0/1 labels, their SBERT gate thresholds were chosen on validation, and
    the LLM-only ensembles were selected on validation, so Table 5 does not depend on the
    threshold rule and is not rebuilt.

Protocol here, per HO slice (scored on the values of the slice):
  * Single: the Direct model selected in rebuild_tables.py (per-label F1 thresholds).
  * Tr ens.: soft voting (mean probability) over the five Direct 19-value models, built by
    forward selection on validation as in Section 3.7 of the paper: start from the best
    single model, add the candidate with the largest one-sided bootstrap lower bound on
    validation, and accept it only if that bound is >= max(0.001, 1% of the current
    Macro-F1). Every candidate ensemble gets its own per-label F1 thresholds on validation.
  * Tr+LLM: hard (majority) voting over the binarized Tr ens. and every cached Gemma-2-9B
    system of the slice (zero-/few-shot, with and without the SBERT gate, QLoRA, and the
    LLM-only ensemble), with the same forward selection on validation.
  * Test is used once: Tr ens. vs Single and Tr+LLM vs Tr ens., one-sided paired bootstrap
    (B = 2000) in both directions, Benjamini-Hochberg across the slices where an
    ensemble was formed.

Usage:
    python3 approaches/ho-revision/rebuild_ensembles.py
Writes results/ensembles.csv, results/ensembles.json and results/ensembles.txt.
"""
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from common import (APPROACHES, RESULTS, SLICES, Bootstrap, bh, gold, label_f1_thresholds, labels_of,
                    load, macro_f1, mcnemar_positives)

DIRECT_DIR = APPROACHES / "moral-values" / "output"
DIRECT_MODELS = ["Baseline", "Lex-LIWC-22", "Lex-MJD", "Previous-Sentences-2", "TD-BERTopic"]
LLM_DIR = APPROACHES / "llm_moral-values" / "output"
LLM_ENS_DIR = APPROACHES / "ensembles" / "llm"
LLM_PREFIX = {"Growth": "growth", "Self-Protection": "self-protection", "Social Focus": "social-focus",
              "Personal Focus": "personal-focus", "Openness": "openness", "Conservation": "conservation",
              "Self-Transcendence": "self-transcendence", "Self-Enhancement": "self-enhancement"}
ABS_MIN, REL_MIN = 0.001, 0.01


def slice_labels(sl):
    folder = SLICES[sl][0]
    f = next((APPROACHES / f"{folder}_moral-values" / "output").glob("1_*-test.tsv"))
    return labels_of(f)


def llm_systems(sl, labs):
    """Every cached Gemma-2-9B system of the slice with validation and test files."""
    out = {}
    for d in sorted(LLM_DIR.glob(f"{LLM_PREFIX[sl]}_*")):
        for vf in sorted(d.glob("google-gemma-2-9b-it*-val.tsv")):
            tf = vf.with_name(vf.name.replace("-val.tsv", "-test.tsv"))
            if tf.exists() and set(labs) <= set(labels_of(vf)):
                kind = d.name[len(LLM_PREFIX[sl]) + 1:]
                out[f"{kind}{' +gate' if '-gated_' in vf.name else ''}"] = (vf, tf)
    ens = LLM_ENS_DIR / LLM_PREFIX[sl]
    if (ens / "fixed-hard-champion-val.tsv").exists() and (ens / "fixed-hard-champion-test.tsv").exists():
        out["LLM ensemble"] = (ens / "fixed-hard-champion-val.tsv", ens / "fixed-hard-champion-test.tsv")
    return out


def forward(cands, yv, boot_v, combine):
    """Forward selection on validation. cands: name -> validation array (prob or bool).
    combine(list of names) -> validation predictions (bool). Returns members and a log."""
    singles = {n: macro_f1(yv, combine([n]), 0.5) for n in cands}
    members = [max(singles, key=singles.get)]
    log = [f"seed {members[0]} (val {singles[members[0]]:.4f})"]
    cur = combine(members)
    cur_f1 = macro_f1(yv, cur, 0.5)
    while True:
        best = None
        for n in cands:
            if n in members:
                continue
            pred = combine(members + [n])
            lower, _ = boot_v.compare(yv, cur, pred)
            if lower >= max(ABS_MIN, REL_MIN * cur_f1) and (best is None or lower > best[1]):
                best = (n, lower, pred)
        if best is None:
            break
        members.append(best[0])
        cur, cur_f1 = best[2], macro_f1(yv, best[2], 0.5)
        log.append(f"+ {best[0]} (lower {best[1]:+.4f}, val {cur_f1:.4f})")
    return members, log


def main() -> None:
    sel = json.loads((RESULTS / "rebuild-selections.json").read_text())
    boot_v, boot_t = Bootstrap(len(gold("val"))), Bootstrap(len(gold("test")))
    rows, detail = [], {}
    for sl in SLICES:
        labs = slice_labels(sl)
        yv = yt = None
        pv, pt = {}, {}
        for m in DIRECT_MODELS:
            yv, pv[m] = load(DIRECT_DIR / f"{m}-val.tsv", labs, "val")
            yt, pt[m] = load(DIRECT_DIR / f"{m}-test.tsv", labs, "test")

        # Single: the Direct model selected in rebuild_tables.py
        single = sel["direct"][sl]
        th_s = label_f1_thresholds(yv, pv[single])
        single_v, single_t = pv[single] >= th_s, pt[single] >= th_s

        # Tr ens.: soft voting, per-label F1 thresholds refitted for every member set
        th_cache = {}

        def soft(names, split="val"):
            key = tuple(sorted(names))
            if key not in th_cache:
                th_cache[key] = label_f1_thresholds(yv, np.mean([pv[n] for n in names], axis=0))
            src = pv if split == "val" else pt
            return np.mean([src[n] for n in names], axis=0) >= th_cache[key]

        tr_members, tr_log = forward(pv, yv, boot_v, soft)
        tr_v, tr_t = soft(tr_members), soft(tr_members, "test")

        # Tr+LLM: hard voting over the binarized Tr ens. and the LLM systems
        bv, bt = {"Tr ens.": tr_v}, {"Tr ens.": tr_t}
        for name, (vf, tf) in llm_systems(sl, labs).items():
            _, bv[name] = load(vf, labs, "val")
            _, bt[name] = load(tf, labs, "test")
            bv[name], bt[name] = bv[name] >= 0.5, bt[name] >= 0.5

        def hard(names, split="val"):
            src = bv if split == "val" else bt
            return np.sum([src[n] for n in names], axis=0) >= len(names) / 2

        hy_members, hy_log = forward(bv, yv, boot_v, hard)
        hy_v, hy_t = hard(hy_members), hard(hy_members, "test")
        best_llm = max((n for n in bv if n != "Tr ens."), key=lambda n: macro_f1(yv, bv[n], 0.5))

        t_ens = boot_t.both_ways(yt, single_t, tr_t)
        t_hyb = boot_t.both_ways(yt, tr_t, hy_t)
        mc = mcnemar_positives(yt, single_t, tr_t)
        rows.append(dict(
            slice=sl, n_values=len(labs), single=single,
            single_val=macro_f1(yv, single_v, 0.5), single_test=macro_f1(yt, single_t, 0.5),
            tr_members=" + ".join(tr_members), tr_val=macro_f1(yv, tr_v, 0.5), tr_test=macro_f1(yt, tr_t, 0.5),
            hy_members=" + ".join(hy_members), hy_val=macro_f1(yv, hy_v, 0.5), hy_test=macro_f1(yt, hy_t, 0.5),
            best_llm=best_llm, best_llm_test=macro_f1(yt, bt[best_llm], 0.5),
            ens_lower95=t_ens["lower95"], ens_p=t_ens["p_2_better"], ens_p_worse=t_ens["p_1_better"],
            hyb_lower95=t_hyb["lower95"], hyb_p=t_hyb["p_2_better"], hyb_p_worse=t_hyb["p_1_better"],
            ens_mcnemar=[(l, p) for l, p in zip(labs, mc) if not np.isnan(p)]))
        detail[sl] = dict(tr_log=tr_log, hy_log=hy_log, llm_systems=sorted(n for n in bv if n != "Tr ens."))

    df = pd.DataFrame(rows)
    for col in ["ens", "hyb"]:
        formed = df[f"{'tr' if col == 'ens' else 'hy'}_members"].str.split(" + ", regex=False).str.len() > 1
        # BH over the comparisons actually tested: a slice where forward selection added no
        # member has no ensemble to test ('--' in Tables 6-7) and is not part of the family.
        for side in ["p", "p_worse"]:
            df[f"{col}_{side}_bh"] = np.nan
            if formed.any():
                df.loc[formed, f"{col}_{side}_bh"] = bh(df.loc[formed, f"{col}_{side}"].to_numpy())
        df[f"{col}_formed"] = formed
    df["ens_mcnemar_sig"] = [
        [l for (l, _), q in zip(r, bh([p for _, p in r]) if r else []) if q < 0.05] for r in df.ens_mcnemar]
    df.drop(columns=["ens_mcnemar"]).to_csv(RESULTS / "ensembles.csv", index=False)
    (RESULTS / "ensembles.json").write_text(json.dumps(detail, indent=2))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 80)
        print("=== Ensembles under per-label F1 thresholds, validation-only selection (Tables 6-7)")
        print(df[["slice", "single", "single_test", "tr_members", "tr_test", "ens_lower95", "ens_p_bh",
                  "ens_p_worse_bh"]].round(3).to_string(index=False))
        print()
        print(df[["slice", "hy_members", "hy_test", "hyb_lower95", "hyb_p_bh", "hyb_p_worse_bh",
                  "best_llm", "best_llm_test"]].round(3).to_string(index=False))
        print("\n--- Forward-selection logs (validation)")
        for sl, d in detail.items():
            print(f"{sl}: Tr ens. {' | '.join(d['tr_log'])}")
            print(f"{'':{len(sl)}}  Tr+LLM  {' | '.join(d['hy_log'])}  [LLM pool: {', '.join(d['llm_systems'])}]")
        print("\n--- Per-label McNemar (Tr ens. vs Single, BH within slice): significant values")
        for _, r in df.iterrows():
            print(f"{r.slice}: {r.ens_mcnemar_sig}")
    text = buf.getvalue()
    (RESULTS / "ensembles.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
