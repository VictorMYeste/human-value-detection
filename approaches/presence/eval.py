import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch.distributed as dist
import torch
import random
import numpy as np

from core.config import MODEL_CONFIG
from core.cli import parse_args
from core.evaluation import eval

import logging
from core.log import logger

def main() -> None:

    # Suppress duplicate logs on multi-GPU runs (only rank 0 logs)
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        logger.setLevel(logging.WARNING)  # Reduce logging for non-primary ranks

    # Load model-specific configuration
    model_group = "presence"
    model_config = MODEL_CONFIG[model_group]
    labels = model_config["labels"]

    # Define CLI arguments for training script
    args = parse_args(prog_name=model_group)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Optionally set the seed if provided
    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name

    # Load predictions and gold labels
    predictions_path = args.output_directory + "/" + model_name + ".tsv"
    gold_labels_path = args.test_dataset + "labels-cat.tsv"

    eval(labels, predictions_path, gold_labels_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))