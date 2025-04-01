import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import random
import torch
import numpy as np

from core.config import MODEL_CONFIG
from core.utils import download_nltk_resources
from core.runner import run_training
from core.cli import parse_args
import optuna

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")
import logging
from core.log import logger


def main() -> None:

    # Load model-specific configuration
    model_group = "growth_selfprotection"
    model_config = MODEL_CONFIG[model_group]

    # filter_labels = ['Presence']
    filter_labels = []

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

    # Download resources only once
    download_nltk_resources()

    def objective(trial):
        # Suggest hyperparameters
        """
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        gradient_accumulation_steps = 2 if batch_size == 4 else 4
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.1, 0.3, log=True)
        """

        # Run training with these hyperparameters
        trainer = run_training(
            pretrained_model=model_config["pretrained_model"],
            labels=model_config["labels"],
            training_dataset_path=args.training_dataset,
            validation_dataset_path=args.validation_dataset,
            lexicon=args.lexicon,
            previous_sentences=args.previous_sentences,
            linguistic_features=args.linguistic_features,
            ner_features=args.ner_features,
            model_name=args.model_name,
            model_directory=args.model_directory,
            multilayer=args.multilayer,
            slice_data=args.slice,
            batch_size=4,
            num_train_epochs=10,
            learning_rate=2e-05,
            weight_decay=0.15,
            gradient_accumulation_steps=4,
            early_stopping_patience=4,
            #custom_stopwords = model_config["custom_stopwords"],
            augment_data=args.augment_data,
            topic_detection=args.topic_detection,
            filter_labels=filter_labels
        )

        # Evaluate and return metric
        eval_results = trainer.evaluate()
        macro_avg_f1 = eval_results["eval_macro-avg-f1-score"]
        return macro_avg_f1

    # If user passes --optimize, run Optuna optimization
    if args.optimize:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")

    else:
        # Normal training run
        model_group = "growth_selfprotection"
        model_config = MODEL_CONFIG[model_group]
    
        # Run the training pipeline
        run_training(
            pretrained_model=model_config["pretrained_model"],
            labels=model_config["labels"],
            training_dataset_path=args.training_dataset,
            validation_dataset_path=args.validation_dataset,
            lexicon=args.lexicon,
            previous_sentences=args.previous_sentences,
            linguistic_features=args.linguistic_features,
            ner_features=args.ner_features,
            model_name=args.model_name,
            model_directory=args.model_directory,
            multilayer=args.multilayer,
            slice_data=args.slice,
            batch_size=4,
            num_train_epochs=10,
            learning_rate=2e-05,
            weight_decay=0.15,
            gradient_accumulation_steps=4,
            early_stopping_patience=4,
            #custom_stopwords = model_config["custom_stopwords"],
            augment_data=args.augment_data,
            topic_detection=args.topic_detection,
            token_pruning=args.token_pruning,
            filter_labels=filter_labels
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))