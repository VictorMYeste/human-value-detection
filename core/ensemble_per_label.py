#!/usr/bin/env python3
"""
Per-label forward-selection ensemble for multi-label models.
Each label has its own expert committee selected via hard, soft, or weighted voting.

Configuration (edit paths and dev-F1s below):
  BASELINE: path to baseline predictions TSV
  MODELS:   list of candidate model TSV paths (excluding baseline)
  GOLD:     path to gold labels TSV
  DEV_F1:   dict mapping basename → validation Macro-F1 for weighted voting

Usage:
  python ensemble_voting.py --mode [hard|soft|weighted] [--debug]

Options:
  --debug    Print per-candidate ΔF1 lower bounds and p-values each round

Procedure:
  1) Load baseline & candidates
  2) Precompute binary arrays (for hard) and probability arrays (for soft/weighted)
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
from sklearn.metrics import f1_score
from core.eval_utils import load_multilabel, bootstrap_f1_lower, per_label_mcnemar

# === Configuration ===
BASELINE = "../approaches/moral-values/output/Baseline-test.tsv"
MODELS = [
    # List your candidate models here (excluding the baseline):
    "../approaches/moral-values/output/Previous-Sentences-2-test.tsv",
    "../approaches/moral-values/output/Lex-LIWC-22-test.tsv",
    "../approaches/moral-values/output/TD-BERTopic-test.tsv",
]
GOLD      = "../data/test-english/labels-cat.tsv"
ID_COLS   = ['Text-ID','Sentence-ID']
PROB_COLS = None   # comma-separated list to explicitly select floats
BOOTSTRAP = 5000
ALPHA     = 0.05

# Validation Macro-F1 for each model basename (for weighted voting)

DEV_F1 = {
    os.path.basename(BASELINE): 0.28,
    os.path.basename(MODELS[0]): 0.29,
    os.path.basename(MODELS[1]): 0.29,
    os.path.basename(MODELS[2]): 0.29,
}
_total = sum(DEV_F1.values())
WEIGHTS = {k: v/_total for k, v in DEV_F1.items()}


def assemble_data(threshold=0.5):
    # load full binary and prob arrays for baseline+models
    # binary
    base_bin = load_multilabel(BASELINE, ID_COLS, PROB_COLS, threshold)
    model_bins = [load_multilabel(m, ID_COLS, PROB_COLS, threshold)[base_bin.columns]
                  for m in MODELS]
    # prob
    paths = [BASELINE] + MODELS
    prob_dfs = []
    for p in paths:
        dfp = pd.read_csv(p, sep='\t').set_index(ID_COLS)
        cols = PROB_COLS.split(',') if PROB_COLS else dfp.select_dtypes(include=['float']).columns.tolist()
        prob_dfs.append(dfp[cols])
    # align to same index and labels
    dfidx = base_bin.index
    labels = base_bin.columns.tolist()
    bin_arrays = [base_bin.values] + [mb.values for mb in model_bins]
    prob_arrays = [dfp.reindex(dfidx)[labels].values for dfp in prob_dfs]
    # gold
    gold = load_multilabel(GOLD, ID_COLS, PROB_COLS, threshold)
    true = gold.reindex(dfidx)[labels].values
    names = [os.path.basename(BASELINE)] + [os.path.basename(m) for m in MODELS]
    return true, bin_arrays, prob_arrays, names, labels


def forward_select_label(baseline_col, cand_cols, true_col, names, mode, weights, debug=False, label_name=None, threshold=0.5):
    """
    baseline_col: (n,) binary or prob
    cand_cols: list of (n,) arrays
    true_col: (n,) binary ground truth
    returns: final (n,) binary ensemble pred, selected names
    """
    # assemble per-label candidates
    arrs = [baseline_col] + cand_cols
    idxs = list(range(1, len(arrs)))
    selected = [0]
    sel_names = [names[0]]
    if debug: print(f"\nLabel '{label_name}': selecting among {names}")
    while True:
        best_lower = 0.0
        best_i = None
        for i in idxs:
            mem = selected + [i]
            stack = np.vstack([arrs[j] for j in mem]).T  # (n_samples, n_models)

            # build the candidate ensemble prediction (already binary)
            if mode == 'hard':
                ens = (stack.sum(1) >= (len(mem)/2)).astype(int)
            elif mode == 'soft':
                ens = (stack.mean(1) >= threshold).astype(int)
            else: # weighted
                w = np.array([weights[names[j]] for j in mem])
                ens = ((stack * w).sum(1) / w.sum() >= threshold).astype(int)

            # binarise baseline if it is still probabilities
            base_bin = (baseline_col >= threshold).astype(int) if mode == 'soft' else baseline_col
            
            lower, p_one = bootstrap_f1_lower(base_bin[:,None], ens[:,None], true_col[:,None], B=BOOTSTRAP, alpha=ALPHA)
            if debug: print(f"  try {names[i]:<25}: lower={lower:.4f} p={p_one:.4f}")
            if lower > best_lower and lower > 0:
                best_lower, best_i = lower, i
        if best_i is None: break
        selected.append(best_i)
        idxs.remove(best_i)
        sel_names.append(names[best_i])
        if debug: print(f" Added {names[best_i]}, bound={best_lower:.4f}")
    # build final
    mem = selected
    stack = np.vstack([arrs[j] for j in mem]).T
    if mode == 'hard':
        final = (stack.sum(1) >= (len(mem)/2)).astype(int)
    elif mode == 'soft':
        final = (stack.mean(1) >= threshold).astype(int)
    else:
        w = np.array([weights[names[j]] for j in mem])
        final = ((stack * w).sum(1) / w.sum() >= threshold).astype(int)
    return final, sel_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hard','soft','weighted'], default='hard')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5, help='Global decision threshold (prob space)')
    args = parser.parse_args()
    true, bin_arrs, prob_arrs, names, labels = assemble_data(args.threshold)
    n_labels = len(labels)
    # pick arrays based on mode
    use_arrays = bin_arrs if args.mode in ['hard','weighted'] else prob_arrs
    weights = WEIGHTS if args.mode=='weighted' else None
    # store per-label predictions & selections
    final_preds = np.zeros_like(true)
    selections = {}
    print(f"\nForward selection per label ({args.mode} voting):")
    for j, lbl in enumerate(labels):
        baseline_col = bin_arrs[0][:,j] if args.mode!='soft' else prob_arrs[0][:,j]
        cand_cols = [arr[:,j] for arr in use_arrays[1:]]
        true_col = true[:,j]
        final_j, sel = forward_select_label(baseline_col, cand_cols, true_col, names,
            args.mode, weights, debug=args.debug, label_name=lbl, threshold=args.threshold)
        final_preds[:,j] = final_j
        selections[lbl] = sel
    # evaluate overall
    f1_b = f1_score(true, bin_arrs[0], average='macro', zero_division=0)
    f1_e = f1_score(true, final_preds, average='macro', zero_division=0)
    lower, p = bootstrap_f1_lower(bin_arrs[0], final_preds, true, B=BOOTSTRAP, alpha=ALPHA)
    print(f"Base F1={f1_b:.4f}, Ensemble F1={f1_e:.4f}")
    print(f"ΔF1 one-sided lower 95% bound = {lower:.4f}, p_one={p:.4f}")
    print("Per-label selections:")
    for lbl, sel in selections.items():
        print(f" {lbl}: {sel}")
    # per-label McNemar on positives
    res = per_label_mcnemar(bin_arrs[0], final_preds, true, labels, alpha=ALPHA)
    print("Per-label McNemar:")
    for lbl, p_adj, sig in res:
        print(f" {lbl}: p_adj={p_adj:.4f}, sig={sig}")

if __name__ == '__main__':
    main()