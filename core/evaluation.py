import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import json
import torch.distributed as dist
import torch
import random
import logging

from core.config import MODEL_CONFIG
from core.cli import parse_args

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HVD")

# --------------------------------------------------------------------------- #
# Evaluation logic
# --------------------------------------------------------------------------- #
def find_best_threshold(y_true: np.ndarray,
                        y_pred_scores: np.ndarray,
                        min_precision: float = 0.4,
                        step: float = 0.01) -> float:
    """
    Searches [0.10 … 0.89] for the threshold that gives the highest recall
    on the positive class while meeting a precision constraint.
    """
    best_th, best_recall = 0.5, 0.0
    for th in np.arange(0.10, 0.90, step):
        y_pred = (y_pred_scores >= th).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0,1], zero_division=0
        )
        # index 1 == positive class
        if prec[1] >= min_precision and rec[1] > best_recall:
            best_recall, best_th = rec[1], th
    return best_th


def eval_labels(labels, predictions_path, gold_labels_path, fixed_threshold=0.5, thresholds_dict=None, thresholds_out=None, compute_tuned=True):
    # ------------------------------------------------------------------- #
    # Load & align data
    # ------------------------------------------------------------------- #
    preds_df = pd.read_csv(predictions_path, sep="\t")
    gold_df  = pd.read_csv(gold_labels_path, sep="\t")

    # Keep only matching columns so accidental extras don’t break the merge
    keep_cols = ["Text-ID", "Sentence-ID"] + labels
    preds_df  = preds_df[keep_cols]
    gold_df   = gold_df[keep_cols]

    merged = pd.merge(
        gold_df, preds_df,
        on=["Text-ID", "Sentence-ID"],
        suffixes=("_gold", "_pred"),
        validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("Merge produced an empty DataFrame – check IDs!")

    # ------------------------------------------------------------------- #
    # Per-label evaluation
    # ------------------------------------------------------------------- #
    if compute_tuned:
        macro_f1_tuned  = []  # for threshold-search F1s
        best_thresholds = {}
    else:
        logger.debug("Tuned thresholds disabled; only fixed evaluation will run")
    macro_f1_fixed  = []   # for fixed-threshold F1s

    for label in labels:
        g = merged[f"{label}_gold"].astype(float).values
        p_scores = merged[f"{label}_pred"].astype(float).values

        # --- (a) tuned threshold -------------------------------------------------
        if compute_tuned:
            # ---------- choose threshold ------------------------------------ #
            if thresholds_dict is not None:
                if label not in thresholds_dict:
                    raise ValueError(f"Threshold for label '{label}' not found "
                                    "in val_thresholds.json")
                best_th = thresholds_dict[label]
            else:
                best_th = find_best_threshold(g, p_scores)
                best_thresholds[label] = round(float(best_th), 2)

            p_bin_tuned = (p_scores >= best_th).astype(int)
            best_thresholds[label] = round(float(best_th), 2)

            prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
                g, p_bin_tuned, average=None, labels=[0,1], zero_division=0
            )
            macro_f1_tuned.append(f1_t[1])

        # --- (b) fixed threshold --------------------------------------------
        p_bin_fixed = (p_scores >= fixed_threshold).astype(int)
        prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
            g, p_bin_fixed, average=None, labels=[0,1], zero_division=0
        )
        macro_f1_fixed.append(f1_f[1])  

        # ---------------------------------------------------------------- #
        # Pretty printing
        # ---------------------------------------------------------------- #
        if compute_tuned:
            print(f"\nLabel: {label}  |  best threshold = {best_th:.2f}")
            print("  Class 0 (negative)")
            print(f"    Precision: {prec_t[0]:.2f}  Recall: {rec_t[0]:.2f}  F1: {f1_t[0]:.2f}")
            print("  Class 1 (positive)")
            print(f"    Precision: {prec_t[1]:.2f}  Recall: {rec_t[1]:.2f}  F1: {f1_t[1]:.2f}")

        print(f"\nLabel: {label}  |  Fixed {fixed_threshold} threshold")
        print("  Class 0 (negative)")
        print(f"    Precision: {prec_f[0]:.2f}  Recall: {rec_f[0]:.2f}  F1: {f1_f[0]:.2f}")
        print("  Class 1 (positive)")
        print(f"    Precision: {prec_f[1]:.2f}  Recall: {rec_f[1]:.2f}  F1: {f1_f[1]:.2f}")

    # ------------------------------------------------------------------- #
    # Macro-average over all labels (positive class only)
    # ------------------------------------------------------------------- #
    print("\n=== SUMMARY =================================================")
    if compute_tuned:
        macro_f1_tuned = np.mean(macro_f1_tuned)
        print(f"\nMacro-average F1 (tuned thresholds) across {len(labels)} labels: {macro_f1_tuned:.5f}")
    macro_f1_fixed = np.mean(macro_f1_fixed)
    print(f"\nMacro-average F1 (fixed threshold) across {len(labels)} labels: {macro_f1_fixed:.5f}")

    # -------- optional: save thresholds found on the validation set -------- #
    if compute_tuned and thresholds_out is not None:
        with open(thresholds_out, "w") as fh:
            json.dump(best_thresholds, fh, indent=2)
        logger.info(f"Per-label thresholds written to {thresholds_out}")


# --------------------------------------------------------------------------- #
# CLI entry-point – keeps the signature you were already using
# --------------------------------------------------------------------------- #
def run(model_group: str = "presence", compute_tuned: bool = True) -> None:
    """
    End-to-end evaluation entry point.

    Parameters
    ----------
    model_group : str, optional
        Key in `core.config.MODEL_CONFIG`.  Change this when you copy the tiny
        `eval.py` wrapper into another model folder.
    """
    # Suppress duplicate logs on multi-GPU runs (only rank-0 logs talk)
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        logger.setLevel(logging.WARNING)

    # ------------------------------- #
    # Config & CLI
    # ------------------------------- #
    model_cfg = MODEL_CONFIG[model_group]
    labels    = model_cfg["labels"]

    args = parse_args(prog_name=model_group)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Optional fixed seed
    if args.seed is not None:
        logger.info(f"Setting random seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # ---------------------------------------------------------------- #
    # Decide which split & which thresholds to use
    # ---------------------------------------------------------------- #
    if args.validation_dataset:
        logger.debug("Using validation dataset for threshold search")
        dataset_path   = args.validation_dataset
        thresholds_dict  = None
        thresholds_out = None
        if compute_tuned:
            thresholds_out = os.path.join(args.output_directory, "val_thresholds.json")
        pred_suffix = "val"
    else:
        dataset_path   = args.test_dataset
        thresholds_dict = None
        if compute_tuned:
            thresholds_path  = os.path.join(args.output_directory, "val_thresholds.json")
            if not os.path.exists(thresholds_path):
                raise FileNotFoundError(
                    f"val_thresholds.json not found in {args.output_directory}. "
                    "Run this script first with --validation-dataset to generate "
                    "the per-label thresholds before evaluating on the test set."
                )
            with open(thresholds_path) as fh:
                thresholds_dict = json.load(fh)
        thresholds_out = None
        pred_suffix = "test"

    # ------------------------------- #
    # File paths
    # ------------------------------- #
    filename = ""
    if args.filter_1_model:
        filename += "1_" + args.filter_1_model + "_" + str(args.filter_1_th) + "_"
    filename += args.model_name + "-" + pred_suffix + ".tsv"
    predictions_path  = os.path.join(args.output_directory, filename)
    gold_labels_path  = os.path.join(dataset_path, "labels-cat.tsv")

    # ------------------------------- #
    # Kick-off evaluation
    # ------------------------------- #
    eval_labels(labels, predictions_path, gold_labels_path, args.threshold, thresholds_dict, thresholds_out, compute_tuned)


# Allow “python evaluation.py” to run a quick presence check
if __name__ == "__main__":
    run()