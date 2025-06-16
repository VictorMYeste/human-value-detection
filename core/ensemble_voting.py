#!/usr/bin/env python3
"""
Forward‐selection ensemble for multi‐label models via hard, soft, or probability weighted voting,
compared against a baseline with Macro‐F1 bootstrap and per‐label McNemar.

Configuration (edit paths and dev-F1s and thresholds below):
  BASELINE: path to baseline predictions TSV
  MODELS:   list of candidate model TSV paths (excluding the baseline)
  GOLD:     path to gold labels TSV
  DEV_F1:   dict mapping basename → validation Macro-F1 for weighted voting
  TUNED_THRESHOLDS: dict mapping basename → tuned threshold for binarization

Usage:
  python ensemble_voting.py --mode [hard|soft|weighted] [--debug] [--threshold T] [--use-tuned] [--save-preds F]

Options:
  --debug        Print per-candidate ΔF1 lower bounds and p-values each round
  --threshold T  Global decision threshold (default 0.5)
  --use-tuned    Use per-model tuned thresholds for binarization instead of fixed 0.5
  --save-preds F Save predictions in a TSV file to be used as a champion prediction

Procedure:
  1) Load baseline & candidates
  2) Precompute binary arrays (for hard/weighted) or probability arrays (for soft)
  3) Forward selection:
       - start with baseline only
       - iteratively add candidate giving largest positive one‐sided lower ΔF1 bound
       - stop when no positive bound
       - in debug mode, show all candidates each round
  4) Final evaluation of selected ensemble
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import f1_score
from core.eval_utils import load_multilabel, bootstrap_f1_lower, per_label_mcnemar

# === Configuration ===
BASELINE = "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Baseline-test.tsv"
MODELS = [
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-eMFD-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-EmoLex-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-EmotionIntensity-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-LIWC-22-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-LIWC-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-MFD-20-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-MJD-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-VAD-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Lex-WorryWords-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_Previous-Sentences-2-test.tsv",
    "../approaches/p_moral-values/output/1_Lex-LIWC-22_LingFeat_0.5_ResidualBlock-test.tsv",
]
GOLD_VAL  = "../data/validation-english/labels-cat.tsv"
GOLD      = "../data/test-english/labels-cat.tsv"
ID_COLS   = ['Text-ID','Sentence-ID']
PROB_COLS = None   # comma-separated list to explicitly select floats
BOOTSTRAP = 5000
ALPHA     = 0.
# Practical filter: do not add a model unless its *one-sided lower* ΔF1 bound exceeds this minimal effect size
MIN_GAIN  = 0.005  # ≈ +.5 macro-F1 point

# Tuned thresholds per model basename (for binarization)
TUNED_THRESHOLDS = {
    os.path.basename(BASELINE):     0.25,
    os.path.basename(MODELS[0]):    0.3,    # eMFD
    os.path.basename(MODELS[1]):    0.25,   # EmoLex
    os.path.basename(MODELS[2]):    0.25,   # EmotionIntensity
    os.path.basename(MODELS[3]):    0.25,   # LIWC-22
    os.path.basename(MODELS[4]):    0.2,    # LIWC
    os.path.basename(MODELS[5]):    0.25,   # MFD-20
    os.path.basename(MODELS[6]):    0.2,    # MJD
    os.path.basename(MODELS[7]):    0.25,   # VAD
    os.path.basename(MODELS[8]):    0.25,   # WorryWords
    os.path.basename(MODELS[9]):    0.3,    # Previous-Sentences-2
    os.path.basename(MODELS[10]):   0.3,   # ResidualBlock
}

def compute_weights(baseline_path: str,
                    model_paths: list[str],
                    gold_val_path: str,
                    threshold: float = 0.5,
                    tuned: Optional[dict] = None) -> dict[str, float]:

    paths      = [baseline_path] + model_paths
    basenames  = [os.path.basename(p) for p in paths]
    val_paths  = [p.replace('-test.tsv', '-val.tsv') for p in paths]

    # ---------- load gold validation ----------
    gold_df = load_multilabel(gold_val_path, ID_COLS, PROB_COLS, threshold)

    weights  = []
    for vp in val_paths:
        thr    = tuned.get(os.path.basename(vp), threshold) if tuned else threshold
        pred_df = load_multilabel(vp, ID_COLS, PROB_COLS, thr).reindex(gold_df.index)

        # ✱ NEW: keep only columns that exist in BOTH dfs
        common = [c for c in pred_df.columns if c in gold_df.columns]
        if not common:
            raise ValueError(f"No shared label columns between {vp} and gold file.")

        gold = gold_df[common].values
        pred = pred_df[common].values
        f1   = f1_score(gold, pred, average="macro", zero_division=0)
        weights.append(max(f1, 1e-6))        # avoid zeros

    weights = np.array(weights)
    weights = weights / weights.sum()

    return {bn: w for bn, w in zip(basenames, weights)}


def assemble_binary(threshold_map):
    """
    Load binary predictions for baseline and models using per-model thresholds.
    threshold_map: dict basename->threshold
    """
    base_name = os.path.basename(BASELINE)
    base_thr = threshold_map.get(base_name, 0.5)
    base_df = load_multilabel(BASELINE, ID_COLS, PROB_COLS, base_thr)
    # models using their thresholds
    model_dfs = []
    for m in MODELS:
        name = os.path.basename(m)
        thr = threshold_map.get(name, 0.5)
        dfm = load_multilabel(m, ID_COLS, PROB_COLS, thr)[base_df.columns]
        model_dfs.append(dfm)
    # align
    df = base_df.copy()
    for i, mdf in enumerate(model_dfs):
        df = df.join(mdf.add_suffix(f'_{i}'), how='inner')
    gold_df = load_multilabel(GOLD, ID_COLS, PROB_COLS, threshold_map.get(base_name, 0.5))
    labels = base_df.columns.tolist()
    true = gold_df.reindex(df.index)[labels].values
    base_arr = df[labels].values
    cand_arrs = [base_arr] + [df[[f"{lbl}_{i}" for lbl in labels]].values
                              for i in range(len(MODELS))]
    names = [os.path.basename(BASELINE)] + [os.path.basename(m) for m in MODELS]
    return df, true, base_arr, cand_arrs, names, labels


def assemble_prob(df):
    """Load raw probabilities for baseline+models (soft voting)."""
    paths = [BASELINE] + MODELS
    prob_arrs = []
    for path in paths:
        dfp = pd.read_csv(path, sep='\t').set_index(ID_COLS)
        cols = PROB_COLS.split(',') if PROB_COLS else dfp.select_dtypes(include=['float']).columns.tolist()
        prob_arrs.append(dfp[cols].reindex(df.index).values)
    return prob_arrs


def forward_selection(baseline, candidates, true, names, labels,
                      mode, weights=None, debug=False, decision_thr=0.5):
    selected = [0]
    candidates_idx = list(range(1, len(candidates)))
    selected_names = [names[0]]
    print(f"\nForward selection ({mode} voting):")
    round_num = 1
    while True:
        best_lower = 0.0
        best_idx = None
        if debug:
            print(f"\nRound {round_num}: Testing candidates: {[names[i] for i in candidates_idx]}")
        for idx in candidates_idx:
            members = selected + [idx]
            stack = np.stack([candidates[i] for i in members], axis=1)
            if mode == 'hard':
                ens = (stack.sum(axis=1) >= (len(members)/2)).astype(int)
            elif mode == 'soft':
                ens = (stack.mean(axis=1) >= decision_thr).astype(int)
            else:  # weighted
                w = np.array([weights[names[i]] for i in members])
                avg = (stack * w[None,:,None]).sum(axis=1) / w.sum()
                ens = (avg >= decision_thr).astype(int)
            lower, p_one = bootstrap_f1_lower(baseline, ens, true, B=BOOTSTRAP, alpha=ALPHA)
            if debug:
                print(f"  {names[idx]:<30} lower={lower:.4f} p={p_one:.4f}")
            if (lower >= MIN_GAIN) and (lower > best_lower):
                best_lower, best_idx = lower, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        candidates_idx.remove(best_idx)
        selected_names.append(names[best_idx])
        print(f"\nAdded {names[best_idx]} -> lower bound={best_lower:.4f}")
        round_num += 1
    print(f"Selected: {selected_names}")
    stack = np.stack([candidates[i] for i in selected], axis=1)
    if mode == 'hard':
        final_ens = (stack.sum(axis=1) >= (len(selected)/2)).astype(int)
    elif mode == 'soft':
        final_ens = (stack.mean(axis=1) >= decision_thr).astype(int)
    else:
        w = np.array([weights[names[i]] for i in selected])
        avg = (stack * w[None,:,None]).sum(axis=1) / w.sum()
        final_ens = (avg >= decision_thr).astype(int)
    return final_ens, selected_names


def main():

    print("Parsing arguments...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hard','soft','weighted'], default='hard', help='Voting mode')
    parser.add_argument('--debug', action='store_true', help='Show per-candidate stats')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for soft/weighted voting')
    parser.add_argument('--use-tuned', action='store_true', help='Use per-model tuned thresholds for binarization')
    parser.add_argument('--save-preds', metavar='FILE', default=None, help='Write ensemble probabilities to FILE (TSV) so the whole champion can be reused as a single model')
    args = parser.parse_args()

    print("Building threshold map...")

    # Build threshold map for binarization
    if args.use_tuned:
        threshold_map = TUNED_THRESHOLDS
    else:
        # fixed threshold for all models
        fixed = args.threshold
        threshold_map = {os.path.basename(BASELINE): fixed}
        for m in MODELS:
            threshold_map[os.path.basename(m)] = fixed

    # Compute weights if needed
    if args.mode == 'weighted':
        print("Computing weights from validation Macro-F1 …")
        weights = compute_weights(BASELINE, MODELS, GOLD_VAL,
                                      threshold=args.threshold,
                                      tuned=TUNED_THRESHOLDS if args.use_tuned else None)
        print("Weights:", {k: f"{v:.3f}" for k,v in weights.items()})
    else:
        weights = None   # not used in hard / soft modes

    print("Assembling...")
    # Assemble
    df, true, base_arr, cand_bin, names, labels = assemble_binary(threshold_map)
    cand_prob_all = assemble_prob(df)
    if args.mode == 'hard':
        candidates = cand_bin
    elif args.mode == 'soft':
        candidates = cand_prob_all
    else: # weighted  → use probabilities
        candidates = cand_prob_all

    print("Forward selection...")

    # Forward selection
    ens_final, members = forward_selection(
        base_arr, candidates, true, names, labels,
        mode=args.mode, weights=weights, debug=args.debug,
        decision_thr=args.threshold)
    
    print("Evaluating...")

    # Evaluate
    f1_b = f1_score(true, base_arr, average='macro', zero_division=0)
    f1_e = f1_score(true, ens_final, average='macro', zero_division=0)
    lower, p_one = bootstrap_f1_lower(base_arr, ens_final, true, B=BOOTSTRAP, alpha=ALPHA)
    sig = 'YES' if p_one < ALPHA else 'no'
    print(f"\n=== Final {args.mode.capitalize()} Ensemble ({members}) ===")
    print(f"Macro-F1 Base = {f1_b:.5f}, Ensemble = {f1_e:.5f}")
    print(f"One-sided lower {100*(1-ALPHA):.0f}% ΔF1 = {lower:.5f}")
    print(f"p(ens>base) = {p_one:.5f} -> significant? {sig}")

    # Per-label improvements
    res = per_label_mcnemar(base_arr, ens_final, true, labels, alpha=ALPHA)
    if res:
        print("Per-label improvements:")
        for lbl, p_adj, rj in res:
            print(f" {lbl}: p_adj={p_adj:.5f}, sig={rj}")
    else:
        print("No per-label discordance on final ensemble.")

    print("Saving ensemble predictions...")

    if args.save_preds:
        # indices of the models that ended up in the ensemble
        sel_idx = [names.index(n) for n in members]
        stack   = np.stack([cand_prob_all[i] for i in sel_idx], axis=1)  # (N, |S|, L)

        if args.mode == 'hard':
            prob_final = stack.mean(axis=1)        # mean of 0/1
        elif args.mode == 'soft':
            prob_final = stack.mean(axis=1)        # mean of probs
        else:  # ③ NEW – probability-weighted average
            w = np.array([weights[n] for n in members])
            prob_final = (stack * w[None, :, None]).sum(axis=1) / w.sum()

        out = pd.DataFrame(prob_final, index=df.index, columns=labels)
        out.reset_index().to_csv(args.save_preds, sep='\t', index=False)
        print(f"Ensemble probabilities written to {args.save_preds}")

if __name__ == '__main__':
    main()