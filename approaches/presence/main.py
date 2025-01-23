import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")

from core.config import MODEL_CONFIG
from core.utils import download_nltk_resources
from core.runner import run_training
from core.cli import parse_args
import optuna

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HVD")

def main() -> None:

    # Load model-specific configuration
    model_group = "presence"
    model_config = MODEL_CONFIG[model_group]

    # Define CLI arguments for training script
    args = parse_args(prog_name=model_group)

    # Download resources only once
    download_nltk_resources()

    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        gradient_accumulation_steps = 2 if batch_size == 4 else 4
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

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
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            custom_stopwords = model_config["custom_stopwords"]
        )

        # Evaluate and return metric
        eval_results = trainer.evaluate()
        macro_avg_f1 = eval_results["eval_marco-avg-f1-score"]
        return macro_avg_f1

    # If user passes --optimize, run Optuna optimization
    if args.optimize:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")

    else:
        # Normal training run
        model_group = "presence"
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
            weight_decay=0.01,
            gradient_accumulation_steps=4,
            early_stopping_patience=4
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))