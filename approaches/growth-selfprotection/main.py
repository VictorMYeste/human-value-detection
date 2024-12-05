from core.config import MODEL_CONFIG
from core.runner import run_training
from core.cli import parse_args
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HVD")

def main() -> None:
    # Load model-specific configuration
    model_group = "growth_selfprotection"
    model_config = MODEL_CONFIG[model_group]

    # Define CLI arguments for training script
    args = parse_args(prog_name=model_group)
    
    # Run the training pipeline
    run_training(
        pretrained_model=model_config["pretrained_model"],
        labels=model_config["labels"],
        training_dataset_path=args.training_dataset,
        validation_dataset_path=args.validation_dataset,
        lexicon=args.lexicon,
        previous_sentences=args.previous_sentences,
        linguistic_features=args.linguistic_features,
        model_name=args.model_name,
        model_directory=args.model_directory,
        slice_data=args.slice,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))