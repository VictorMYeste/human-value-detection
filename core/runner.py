import transformers
from core.dataset_utils import prepare_datasets, load_and_optionally_prune_df
from core.lexicon_utils import load_embeddings
from core.training import train
from core.models import save_model, DynamicPrevLabelCallback
import torch.distributed as dist
from accelerate import Accelerator

from core.log import logger

def run_training(
    pretrained_model: str,
    labels: list[str],
    training_dataset_path: str,
    validation_dataset_path: str = None,
    lexicon: str = None,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    ner_features: bool = False,
    model_name: str = None,
    model_directory: str = "models",
    multilayer: bool = False,
    slice_data: bool = False,
    batch_size: int = 4,
    num_train_epochs: int = 9,
    learning_rate: float = 2.07e-05,
    weight_decay: float = 1.02e-05,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 3,
    custom_stopwords: list[str] = [],
    augment_data: bool = False,
    topic_detection: str = None,
    token_pruning: str = None
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
    if lexicon and lexicon not in ["LIWC-22", "eMFD", "MFD-20", "MJD"]:
        lexicon_embeddings, num_categories = load_embeddings(lexicon)
    else:
        lexicon_embeddings, num_categories = None, 0  # No lexicon features

    # Linguistic embeddings
    if linguistic_features:
        num_linguistic_features = 17  # Total number of linguistic features
        num_categories += num_linguistic_features
    
    # Prepare datasets
    logger.info("Preparing datasets for training and validation")
    training_dataset, validation_dataset = prepare_datasets(
        training_path=training_dataset_path,
        validation_path=validation_dataset_path,
        tokenizer=tokenizer,
        labels=labels,
        slice_data=slice_data,
        lexicon_embeddings=lexicon_embeddings,
        num_categories=num_categories,
        previous_sentences=previous_sentences,
        linguistic_features=linguistic_features,
        ner_features=ner_features,
        lexicon=lexicon,
        custom_stopwords=custom_stopwords,
        augment_data=augment_data,
        topic_detection=topic_detection,
        token_pruning=token_pruning
    )

    # Train and evaluate
    trainer = train(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        validation_path=validation_dataset_path,
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        labels=labels,
        label2id=label2id,
        id2label=id2label,
        model_name=model_name,
        num_categories=num_categories,
        lexicon=lexicon,
        previous_sentences=previous_sentences,
        linguistic_features=linguistic_features,
        ner_features=ner_features,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        multilayer=multilayer,
        custom_stopwords=custom_stopwords,
        augment_data=augment_data,
        topic_detection=topic_detection,
        token_pruning=token_pruning,
        slice_data=slice_data
    )

    # Save the model if required
    accelerator = Accelerator()
    if model_name and accelerator.is_main_process:
        logger.info(f"Saving best model to {model_directory} directory")
        save_model(trainer, model_name, model_directory)

    # Return the trainer so that caller (objective function) can evaluate
    return trainer