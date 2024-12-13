import transformers
from core.dataset_utils import prepare_datasets
from core.utils import slice_for_testing
from core.lexicon_utils import load_embeddings
from core.training import train
from core.models import save_model
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")

def run_training(
    pretrained_model: str,
    labels: list[str],
    training_dataset_path: str,
    validation_dataset_path: str = None,
    lexicon: str = None,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    model_name: str = None,
    model_directory: str = "models",
    slice_data: bool = False,
    batch_size: int = 4,
    num_train_epochs: int = 9,
    learning_rate: float = 2.07e-05,
    weight_decay: float = 1.02e-05,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 3
):

    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Tokenizer
    logger.info("Initializing tokenizer for model: %s", pretrained_model)
    tokenizer = transformers.DebertaTokenizer.from_pretrained(
        pretrained_model,
        truncation_side = "left" if previous_sentences else "right"
    )

    # Lexicon embeddings
    logger.info("Loading lexicon embeddings for: %s", lexicon if lexicon else "No lexicon used")
    lexicon_embeddings, num_categories = load_embeddings(lexicon)

    # Linguistic embeddings
    if linguistic_features:
        num_linguistic_features = 17  # Total number of linguistic features
        num_categories += num_linguistic_features
    
    # Prepare datasets
    logger.info("Preparing datasets for training and validation")
    training_dataset, validation_dataset = prepare_datasets(
        training_dataset_path,
        validation_dataset_path,
        tokenizer,
        labels,
        lexicon_embeddings,
        num_categories
    )

    # Slicing for testing purposes
    if slice_data:
        training_dataset = slice_for_testing(training_dataset)
        validation_dataset = slice_for_testing(validation_dataset)

    # Train and evaluate
    trainer = train(
        training_dataset,
        validation_dataset,
        pretrained_model,
        tokenizer,
        labels = labels,
        label2id = label2id,
        id2label = id2label,
        model_name = model_name,
        num_categories=num_categories,
        lexicon = lexicon,
        previous_sentences = previous_sentences,
        linguistic_features = linguistic_features,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience
    )

    # Save the model if required
    save_model(trainer, model_name, model_directory)

    # Return the trainer so that caller (objective function) can evaluate
    return trainer