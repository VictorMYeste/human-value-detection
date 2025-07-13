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
  python ensemble_voting.py --mode [hard|soft|weighted] [--debug] [--threshold T] [--use-tuned] [--save-preds F] [--test]

Options:
  --debug           Print per-candidate ΔF1 lower bounds and p-values each round
  --threshold T     Global decision threshold (default 0.5)
  --use-tuned       Use per-model tuned thresholds for binarization instead of fixed 0.5
  --save-preds F    Save predictions in a TSV file to be used as a champion prediction
  --test            Evaluate with the baseline using the test dataset
  --subset          Comma-separated list of moral-value labels to evaluate (overrides the baseline label set)

Procedure:
  1) Load baseline & candidates
  2) Precompute binary arrays (for hard/weighted) or probability arrays (for soft)
  3) Forward selection:
       - start with an empty selection, but comparing with using the baseline
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
from typing import Optional, List, Dict
from sklearn.metrics import f1_score
from core.eval_utils import load_multilabel, bootstrap_f1_lower, per_label_mcnemar

# === Configuration ===
BASELINE = "../approaches/self-trans_moral-values/output/self-trans-champion_tuned-soft-champion-test.tsv"
MODELS = [
    # "../approaches/moral-values/output/direct_champion-tuned-soft-champion-val.tsv",
    # "../approaches/p_moral-values/output/presence_champion-tuned-soft-champion-val.tsv",
]
GOLD_VAL  = "../data/validation-english/labels-cat.tsv"
GOLD      = "../data/test-english/labels-cat.tsv"
ID_COLS   = ['Text-ID','Sentence-ID']
PROB_COLS = None   # comma-separated list to explicitly select floats
BOOTSTRAP = 5000
ALPHA     = 0.05
# Practical filter: do not add a model unless its *one-sided lower* ΔF1 bound exceeds this minimal effect size
ABS_MIN_GAIN  = 0.001  # ≈ +.1 macro-F1 point
REL_MIN_GAIN = 0.01   # 1 %

# Tuned thresholds per model basename (for binarization)
TUNED_THRESHOLDS = {
    os.path.basename(BASELINE):     0.29,
    # os.path.basename(MODELS[0]):    0.3,
    # os.path.basename(MODELS[1]):    0.31,
}
def compute_weights(baseline_path: str,
                    model_paths: List[str],
                    gold_val_path: str,
                    threshold: float = 0.5,
                    tuned: Optional[Dict] = None) -> Dict[str, float]:

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


def assemble_binary(threshold_map, test, label_subset=None):
    """
    Load binary predictions for baseline and models using per-model thresholds.
    threshold_map: dict basename->threshold
    """
    base_name = os.path.basename(BASELINE)
    base_thr = threshold_map.get(base_name, 0.5)
    base_df = load_multilabel(BASELINE, ID_COLS, PROB_COLS, base_thr)

    # ── if the user supplied --subset, keep only those columns ───────────
    if label_subset is not None:
        missing = [c for c in label_subset if c not in base_df.columns]
        if missing:
            raise ValueError(f"Baseline file lacks requested labels: {missing}")
        base_df = base_df[label_subset]

    # models using their thresholds
    model_dfs = []
    for m in MODELS:
        name = os.path.basename(m)
        thr = threshold_map.get(name, 0.5)
        dfm = load_multilabel(m, ID_COLS, PROB_COLS, thr)[base_df.columns]
        # load, then force same rows & columns as the baseline
        dfm = load_multilabel(m, ID_COLS, PROB_COLS, thr)
        dfm = dfm.reindex(base_df.index).reindex(columns=base_df.columns, fill_value=0)
        model_dfs.append(dfm)
    # align
    df = base_df.copy()
    for i, mdf in enumerate(model_dfs):
        df = df.join(mdf.add_suffix(f'_{i}'), how='left')
        df = df.fillna(0)
    if test:
        gold = GOLD
    else:
        gold = GOLD_VAL
    gold_df = load_multilabel(gold, ID_COLS, PROB_COLS, threshold_map.get(base_name, 0.5))
    labels = base_df.columns.tolist()
    true = gold_df.reindex(df.index)[labels].values
    base_arr = df[labels].values
    cand_arrs = [base_arr] + [df[[f"{lbl}_{i}" for lbl in labels]].values
                              for i in range(len(MODELS))]
    names = [os.path.basename(BASELINE)] + [os.path.basename(m) for m in MODELS]
    return df, true, base_arr, cand_arrs, names, labels


def assemble_prob(df, labels):
    """Load raw probabilities for baseline+models (soft voting)."""
    paths = [BASELINE] + MODELS
    prob_arrs = []
    for path in paths:
        dfp = pd.read_csv(path, sep='\t').set_index(ID_COLS)
        # same rows as baseline, same columns (= label set) – missing → 0
        dfp = dfp.reindex(df.index).reindex(columns=labels, fill_value=0)
        prob_arrs.append(dfp.values)
    return prob_arrs


def forward_selection(baseline, candidates, true, names, labels,
                      mode, weights=None, debug=False, decision_thr=0.5):
    # --- turn probabilities into 0/1 if we are in soft / weighted mode ----
    def binarise(arr):
        return (arr >= decision_thr).astype(int)
    cand_bin_for_scores = (
        candidates if mode == "hard"
        else [binarise(c) for c in candidates]
    )
    # includes baseline at index 0
    scores = [f1_score(true, cand_bin_for_scores[i], average='macro', zero_division=0) for i in range(len(candidates))]
    best0          = int(np.argmax(scores))    # position of best stand-alone model
    selected       = [best0]                   # start ensemble
    selected_names = [names[best0]]
    base_ref       = cand_bin_for_scores[best0]
    current_macro  = scores[best0]
    candidates_idx = [i for i in range(len(candidates)) if i != best0]
    print(f"Seed model: {names[best0]}  (Macro-F1 = {current_macro:.5f})")
    print(f"\nForward selection ({mode} voting):")
    round_num = 1
    while True:
        best_lower = 0.0
        best_idx = None
        print(f"[Round {round_num}]. Current Macro-Average F1 = {current_macro:.5f}")
        if debug:
            print(f"\nTesting candidates: {[names[i] for i in candidates_idx]}")
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
            lower, p_one = bootstrap_f1_lower(base_ref, ens, true, B=BOOTSTRAP, alpha=ALPHA)
            if debug:
                print(f"  {names[idx]:<30} lower={lower:.5f} p={p_one:.5f}")
            if lower >= ABS_MIN_GAIN and lower >= REL_MIN_GAIN * current_macro and lower > best_lower:
                best_lower, best_idx = lower, idx
        if best_idx is None:
            break
        # Accept the candidate
        selected.append(best_idx)
        candidates_idx.remove(best_idx)
        selected_names.append(names[best_idx])
        # After the **first** successful addition, make sure the baseline (idx 0) is available for subsequent rounds (if not selected yet).
        if 0 not in selected and 0 not in candidates_idx:
            candidates_idx.append(0)
        # Update reference for next round
        base_pred = ens
        base_ref = ens
        current_macro = f1_score(true, base_pred, average='macro', zero_division=0)
        print(f"\nAdded {names[best_idx]} -> lower bound={best_lower:.5f} current f1 = {current_macro:.5f}")

        round_num += 1

    print(f"\nSelected: {selected_names or ['<none – fallback to baseline>']}")

    if not selected:
        return baseline, []

    stack = np.stack([candidates[i] for i in selected], axis=1)
    if mode == 'hard':
        final_ens = (stack.sum(axis=1) >= (len(selected)/2)).astype(int)
    elif mode == 'soft':
        final_ens = (stack.mean(axis=1) >= decision_thr).astype(int)
    else:
        w = np.array([weights[names[i]] for i in selected])
        avg = (stack * w[None,:,None]).sum(axis=1) / w.sum()
        final_ens = (avg >= decision_thr).astype(int)

    return final_ens, selected_names, best0

def build_ensemble(member_idx, candidates, names, mode, weights, decision_thr):
    """Return binary ensemble predictions for the given member indices."""
    stack = np.stack([candidates[i] for i in member_idx], axis=1)  # (N, |S|, L)
    if mode == 'hard':
        return (stack.sum(axis=1) >= (len(member_idx)/2)).astype(int)
    elif mode == 'soft':
        return (stack.mean(axis=1) >= decision_thr).astype(int)
    else:  # weighted
        w = np.array([weights[names[i]] for i in member_idx])
        avg = (stack * w[None,:,None]).sum(axis=1) / w.sum()
        return (avg >= decision_thr).astype(int)


def main():

    print("Parsing arguments...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hard','soft','weighted'], default='hard', help='Voting mode')
    parser.add_argument('--debug', action='store_true', help='Show per-candidate stats')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for soft/weighted voting')
    parser.add_argument('--use-tuned', action='store_true', help='Use per-model tuned thresholds for binarization')
    parser.add_argument('--save-preds', metavar='FILE', default=None, help='Write ensemble probabilities to FILE (TSV) so the whole champion can be reused as a single model')
    parser.add_argument('--test', action='store_true', help='If to use the list of Basenames (or full paths) of models that form the ensemble. If given, forward-selection is **skipped** and these models are evaluated as a fixed set.')
    parser.add_argument('--subset', metavar='LABELS', help='Comma-separated list of moral-value labels to evaluate (overrides the baseline label set)')

    args = parser.parse_args()

    # requested label subset?
    label_subset = None
    if args.subset:
        label_subset = [s.strip() for s in args.subset.split(',') if s.strip()]

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
    df, true, base_arr, cand_bin, names, labels = assemble_binary(threshold_map, args.test, label_subset)
    cand_prob_all = assemble_prob(df, labels)
    if args.mode == 'hard':
        candidates = cand_bin
    elif args.mode == 'soft':
        candidates = cand_prob_all
    else: # weighted  → use probabilities
        candidates = cand_prob_all

    if args.test:
        print("Using fixed ensemble")
        members_idx = [names.index(m) for m in names]
        members = [names[i] for i in members_idx]
        ens_final   = build_ensemble(members_idx, candidates, names, args.mode,
                                    weights, args.threshold)
        scores    = [f1_score(true, (cand if args.mode=="hard" else (cand >= args.threshold).astype(int)), average='macro', zero_division=0) for cand in candidates]
        seed_idx  = int(np.argmax(scores))
    
    else:
        print("Forward selection...")

        # Forward selection
        ens_final, members, seed_idx = forward_selection(
            base_arr, candidates, true, names, labels,
            mode=args.mode, weights=weights, debug=args.debug,
            decision_thr=args.threshold)
    
    print("Evaluating...")

    # Evaluate
    seed_pred = (candidates[seed_idx] if args.mode == "hard" else (candidates[seed_idx] >= args.threshold).astype(int))
    f1_seed = f1_score(true, seed_pred, average='macro', zero_division=0)
    f1_e = f1_score(true, ens_final, average='macro', zero_division=0)
    lower, p_one = bootstrap_f1_lower(seed_pred, ens_final, true, B=BOOTSTRAP, alpha=ALPHA)
    sig = 'YES' if p_one < ALPHA else 'no'
    print(f"\n=== Final {args.mode.capitalize()} Ensemble ({members}) ===")
    print(f"Macro-F1 Seed = {f1_seed:.5f}, Ensemble = {f1_e:.5f}")
    print(f"One-sided lower {100*(1-ALPHA):.0f}% ΔF1 = {lower:.5f}")
    print(f"p(ens>base) = {p_one:.5f} -> significant? {sig}")

    # Per-label improvements
    res = per_label_mcnemar(seed_pred, ens_final, true, labels, alpha=ALPHA)
    if res:
        print("Per-label improvements:")
        for lbl, p_adj, rj in res:
            print(f" {lbl}: p_adj={p_adj:.5f}, sig={rj}")
    else:
        print("No per-label discordance on final ensemble.")

    print("Saving ensemble predictions...")

    if args.save_preds:
        if not members:     # forward-selection kept only the baseline
            print("No model passed the gain threshold – ensemble = baseline. "
                "Skipping --save-preds.")
        else:
            if args.mode == 'hard':
                # hard voting → write the final binary decisions as 0.0 / 1.0
                prob_final = ens_final.astype(float)
                out = pd.DataFrame(prob_final, index=df.index, columns=labels)
                out.reset_index().to_csv(args.save_preds, sep='\t', index=False)
                print(f"Binary (0/1) ensemble predictions written to {args.save_preds}")
            else:
                # indices of the models that ended up in the ensemble
                sel_idx = [names.index(n) for n in members]
                stack   = np.stack([cand_prob_all[i] for i in sel_idx], axis=1)  # (N, |S|, L)

                if args.mode == 'soft':
                    prob_final = stack.mean(axis=1)        # mean of probs
                else:  # probability-weighted average
                    w = np.array([weights[n] for n in members])
                    prob_final = (stack * w[None, :, None]).sum(axis=1) / w.sum()

                out = pd.DataFrame(prob_final, index=df.index, columns=labels)
                out.reset_index().to_csv(args.save_preds, sep='\t', index=False)
                print(f"Ensemble probabilities written to {args.save_preds}")

if __name__ == '__main__':
    main()