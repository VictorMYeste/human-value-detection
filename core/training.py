import os
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import AutoConfig
import torch.distributed as dist

from core.models import EnhancedDebertaModel, CustomTrainer, move_to_device, WarmupEvalCallback, DynamicPrevLabelCallback
from core.dataset_utils import load_and_optionally_prune_df
from core.utils import clear_directory

from core.log import logger

# ========================================================
# METRICS
# ========================================================

METRIC_F1_SCORE = "eval_f1-score"
METRIC_MACRO_F1_SCORE = "eval_macro-avg-f1-score"

def compute_metrics(eval_prediction, id2label):
    """Compute evaluation metrics like F1-score."""
    prediction_scores, label_scores = eval_prediction

    logger.debug(f"Prediction scores: {prediction_scores[:5]}")  # Log a few predictions
    logger.debug(f"Label scores: {label_scores[:5]}")            # Log a few labels
    logger.debug(f"Prediction scores shape: {np.array(prediction_scores).shape}")
    logger.debug(f"Label scores shape: {np.array(label_scores).shape}")

    # Convert to tensors
    predictions = torch.sigmoid(torch.tensor(np.array(prediction_scores))) >= 0.5
    labels = torch.tensor(label_scores) >= 0.5

    # Ensure tensors have proper dimensions
    predictions = predictions.unsqueeze(0) if predictions.dim() == 1 else predictions
    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")

    logger.debug(f"Predictions shape: {predictions.shape}")
    logger.debug(f"Labels shape: {labels.shape}")

    # Compute F1 scores for each label
    f1_scores = {}
    for i in range(predictions.shape[1]):
        predicted = predictions[:, i].sum().item()
        true = labels[:, i].sum().item()
        true_positives = torch.logical_and(predictions[:,i], labels[:,i]).sum().item()
        precision = 0 if predicted == 0 else true_positives / predicted
        recall = 0 if true == 0 else true_positives / true
        f1_scores[id2label[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)

    # Compute macro-average F1 score
    macro_average_f1_score = round(np.mean(list(f1_scores.values())), 2)
    return {'f1-score': f1_scores, 'macro-avg-f1-score': macro_average_f1_score}

# ========================================================
# TRAINING
# ========================================================

def create_training_args(output_dir, model_name, batch_size, num_train_epochs, learning_rate, weight_decay, gradient_accumulation_steps):
    return transformers.TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        hub_model_id=model_name,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='macro-avg-f1-score',
        gradient_accumulation_steps=gradient_accumulation_steps,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        save_on_each_node=True if dist.is_initialized() else False  # Ensure model is saved on each GPU node
    )

def train(
        training_dataset,
        validation_dataset,
        validation_path: str,
        pretrained_model: str,
        tokenizer: transformers.PreTrainedTokenizer,
        labels: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        model_name: str = None,
        batch_size: int = 4,
        num_train_epochs: int = 10,
        learning_rate: float = 2e-05,
        weight_decay: float = 0.15,
        gradient_accumulation_steps: int = 4,
        early_stopping_patience=4,
        num_categories: int = 0,
        lexicon: str = None,
        previous_sentences: bool = False,
        linguistic_features: bool = False,
        ner_features: bool = False,
        multilayer: bool = False,
        custom_stopwords: list[str] = [],
        augment_data: bool = False,
        topic_detection: str = None,
        token_pruning: bool = False,
        slice_data: bool = False
    ) -> transformers.Trainer:
    """Train the model and evaluate performance."""

    if previous_sentences or augment_data:
        scaled_gradient_accumulation_steps = int(gradient_accumulation_steps * batch_size / 2)
        logger.info(f"Previous sentences or augmented data detected. Adjusting batch size: {batch_size} -> 2 and gradient accumulation steps: {gradient_accumulation_steps} -> {scaled_gradient_accumulation_steps}")
        gradient_accumulation_steps = scaled_gradient_accumulation_steps
        batch_size = 2
    
    # Detect number of available GPUs
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        scaled_batch_size = batch_size * num_gpus  # Multiply batch size per GPU count
        scaled_gradient_accumulation_steps = int(gradient_accumulation_steps * scaled_batch_size / 2)
        logger.info(f"Multi-GPU detected ({num_gpus} GPUs). Adjusting batch size: {batch_size} -> {scaled_batch_size} and gradient accumulation steps: {gradient_accumulation_steps} -> {scaled_gradient_accumulation_steps}")
        batch_size = scaled_batch_size
        gradient_accumulation_steps = scaled_gradient_accumulation_steps
    """

    output_dir = "models/checkpoints"

    # Ensure only rank 0 (primary process) clears the directory before training starts
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Clearing old checkpoints in {output_dir}")
        clear_directory(output_dir)
    
    training_args = create_training_args(
        output_dir, model_name, batch_size, num_train_epochs, learning_rate, weight_decay, gradient_accumulation_steps
    )

    if ner_features:
        # ner_feature_dim = 8
        ner_feature_dim = 768 # DeBERTa hidden size
    else:
        ner_feature_dim = 0

    if topic_detection == "bertopic":
        topic_feature_dim = 40
    elif topic_detection == "lda":
        topic_feature_dim = 60
    elif topic_detection == "nmf":
        topic_feature_dim = 90
    else:
        topic_feature_dim = 0
    
    config = AutoConfig.from_pretrained(pretrained_model)
    # Add necessary attributes to config
    config.id2label = id2label
    config.label2id = label2id
    config.problem_type = "multi_label_classification"
    config.architectures = ["DebertaForSequenceClassification"]
    model = EnhancedDebertaModel(
        pretrained_model=pretrained_model,
        config=config,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        num_categories=num_categories,
        ner_feature_dim=ner_feature_dim,
        multilayer=multilayer,
        topic_feature_dim=topic_feature_dim,
        previous_sentences=previous_sentences
    )
    """
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, problem_type="multi_label_classification",
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    """
    
    model = move_to_device(model)

    logger.info("TRAINING")
    logger.info("========")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.0)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Log the training arguments
    config_details = (
        f"Pre-trained model: {pretrained_model}\n"
        f"Model name: {model_name if model_name else 'None'}\n"
        f"Batch size: {batch_size}\n"
        f"Number of epochs: {num_train_epochs}\n"
        f"Learning rate: {learning_rate}\n"
        f"Weight decay: {weight_decay}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Early stopping patience: {early_stopping_patience}\n"
        f"Multilayer: {'Yes' if multilayer else 'No'}\n"
        f"Previous sentences used: {'Yes' if previous_sentences else 'No'}\n"
        f"Using lexicon: {lexicon if lexicon else 'No'}\n"
        f"Adding linguistic features: {'Yes' if linguistic_features else 'No'}\n"
        f"Adding NER features: {'Yes' if ner_features else 'No'}\n"
        f"Number of categories (lexicon): {num_categories}\n"
        f"Using data augmentation with paraphrasing: {'Yes' if augment_data else 'No'}\n"
        f"Adding topic detection features: {'Yes' if topic_detection else 'No'}\n"
        f"Applying token pruning: {'Yes' if token_pruning else 'No'}\n"
    )
    logger.info("Training configuration:\n" + config_details)

    if lexicon:
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
    else:
        trainer = transformers.Trainer(
            model,
            training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )

    # Add a warmup of 2 epochs to avoid initial flukes
    warmup_callback = WarmupEvalCallback(warmup_epochs=2)
    trainer.add_callback(warmup_callback)

    # Add a callback for previous sentences
    if previous_sentences:
        logger.info("Rebuilding validation dataset with dynamically predicted previous labels.")
        # Load the raw validation DataFrame from file
        raw_val_df = load_and_optionally_prune_df(
            dataset_path=validation_path,
            augment_data=False,
            slice_data=slice_data,
            custom_stopwords=custom_stopwords,
            token_pruning=token_pruning,
            idf_map=None
        )

        if augment_data:
            labels_file = "labels-cat-aug.tsv"
        else:
            labels_file = "labels-cat.tsv"
        labels_file_path = os.path.join(validation_path, labels_file)
        labels_df = pd.read_csv(labels_file_path, sep="\t") if labels_file_path else None

        trainer.add_callback(
            DynamicPrevLabelCallback(
                trainer=trainer,
                val_df=raw_val_df,
                labels_df=labels_df,
                labels=labels,
                tokenizer=tokenizer
            )
        )

    trainer.train()

    logger.info("\n\nVALIDATION")
    logger.info("==========")
    evaluation = trainer.evaluate()
    for label in labels:
        logger.info(f"{label}: {evaluation[METRIC_F1_SCORE][label]:.2f}")
    logger.info(f"Macro average: {evaluation[METRIC_MACRO_F1_SCORE]:.2f}")

    # Ensure distributed training cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return trainer