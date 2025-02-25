import torch
from torch import nn
import transformers
from transformers import TrainerCallback
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.distributed as dist
import os
import pandas as pd

from core.dataset_utils import add_previous_label_features
from core.log import logger

# ========================================================
# UTILS
# ========================================================

def save_model(trainer, model_name, model_directory):
    # Ensure only the main GPU (rank 0) saves the model
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    
    if model_name:
        logger.info(f"UPLOAD to https://huggingface.co/{model_name} (using HF_TOKEN environment variable)")
        # trainer.push_to_hub()

    if model_directory and model_name:
        logger.info(f"SAVE to {model_directory}")
        trainer.save_model(f"{model_directory}/{model_name}")

        # Ensure the model's configuration is also saved
        if hasattr(trainer.model, 'config'):
            trainer.model.config.save_pretrained(f"{model_directory}/{model_name}")

        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(f"{model_directory}/{model_name}")

def move_to_device(model):
    if torch.cuda.is_available():
        logger.info("Using CUDA for training.")
        return model.to('cuda')
    else:
        logger.info("Using CPU for training.")
        return model

# ========================================================
# MODELS
# ========================================================

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_groups=8):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim),
            nn.GroupNorm(num_groups, output_dim),
            nn.ReLU()
        )
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.linear_layers(x) + self.projection(x)  # Add residual connection

class EnhancedDebertaModel(nn.Module):
    """Enhanced DeBERTa model with added lexicon feature layer."""
    def __init__(
            self,
            pretrained_model,
            config,
            num_labels,
            id2label,
            label2id,
            num_categories=0,
            ner_feature_dim=0,
            multilayer = False,
            num_groups=8,
            topic_feature_dim=0,
            previous_sentences=False
        ):
        #super(EnhancedDebertaModel, self).__init__()
        super().__init__()
        self.config = config  # Store config attribute

        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)

        """
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.transformer = torch.nn.DataParallel(self.transformer)  # Enables Multi-GPU
        """

        # Optional Lexicon Layer
        #self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        if num_categories > 0:
            self.lexicon_layer = nn.Sequential(
                nn.Linear(num_categories, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            logger.debug("Lexicon layer initialized at model")
        else:
            self.lexicon_layer = None
            logger.debug("No lexicon layer initialized at model")

        # Optional NER Layer
        #self.ner_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        if ner_feature_dim > 0:
            self.ner_layer = nn.Sequential(
                nn.Linear(ner_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
            )
            logger.debug("NER layer initialized at model")
        else:
            self.ner_layer = None
            logger.debug("No NER layer initialized at model")
        
        if topic_feature_dim > 0:
            self.topic_layer = nn.Sequential(
                nn.Linear(topic_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            logger.debug("Topic Detection layer initialized at model")
        else:
            self.topic_layer = None
            logger.debug("No Topic Detection layer initialized at model")
        
        # Multi-layer processing for transformer embeddings
        self.multilayer = multilayer
        if multilayer:
            self.text_embedding_layer = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, 512),
                nn.GroupNorm(num_groups, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.GroupNorm(num_groups, 256),
                nn.ReLU()
            )
            
            # self.text_embedding_layer = ResidualBlock(self.transformer.config.hidden_size, 256)
            hidden_size = 256
        else:
            hidden_size = self.transformer.config.hidden_size
        
        # Add labels from previous sentences
        if previous_sentences:
            self.prev_label_size = 2 * num_labels # 2 previous sentences
            self.prev_label_layer = nn.Sequential(
                nn.Linear(self.prev_label_size, 16),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            logger.debug(f"Previous label layer initialized with prev_label_size = {self.prev_label_size}.")
        else:
            self.prev_label_size = 0
            self.prev_label_layer = None
            logger.debug("No previous label layer initialized.")

        # Classification head. Combine all features
        input_dim = hidden_size
        if self.lexicon_layer:
            input_dim += 128
        if self.ner_layer:
            input_dim += 128
        if self.topic_layer:
            input_dim += 128
        if self.prev_label_layer:
            input_dim += 16

        logger.debug(f"Final computed input_dim for classification head: {input_dim}")

        self.classification_head = nn.Linear(input_dim, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(
        self,
        input_ids,
        attention_mask,
        lexicon_features=None,
        ner_features=None,
        topic_features=None,
        prev_label_features=None,
        labels=None
    ):
        """Forward pass for the enhanced model."""

        logger.debug(f"Lexicon features received: {lexicon_features is not None}")
        logger.debug(f"NER features received: {ner_features is not None}")

        if lexicon_features is not None:
            logger.debug(f"Lexicon feature shape: {lexicon_features.shape}")

        if ner_features is not None:
            logger.debug(f"NER feature shape: {ner_features.shape}")

        # Extract transformer embeddings
        hidden_state = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = hidden_state[:, 0, :] # CLS token representation

        # Process transformer embeddings through additional layers
        # (A) DeBERTa forward
        if self.multilayer:
            text_embeddings = self.text_embedding_layer(transformer_output)
        else:
            text_embeddings = transformer_output
        
        combined_output = text_embeddings

        logger.debug(f"DeBERTa combined_output shape: {combined_output.shape}")
        logger.debug(f"Checking prev_label_layer: {self.prev_label_layer}")
        logger.debug(f"Checking prev_label_features: {prev_label_features}")

        # (B) Add lexicon features
        if self.lexicon_layer and lexicon_features is not None:
            logger.debug(f"Lexicon features shape before processing: {lexicon_features.shape}")
            lexicon_features = lexicon_features.to(input_ids.device)
            #lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            lexicon_output = self.lexicon_layer(lexicon_features)
            logger.debug(f"Lexicon output shape: {lexicon_output.shape}")
            combined_output = torch.cat([combined_output, lexicon_output], dim=-1)

            logger.debug(f"Lexicon combined_output shape: {combined_output.shape}")
        
        # (C) Add NER features
        if self.ner_layer and ner_features is not None:
            logger.debug(f"NER features shape before processing: {ner_features.shape}")
            ner_features = ner_features.to(input_ids.device)
            ner_output = self.ner_layer(ner_features)
            logger.debug(f"NER output shape: {ner_output.shape}")
            combined_output = torch.cat([combined_output, ner_output], dim=-1)
            logger.debug(f"NER combined_output shape: {combined_output.shape}")

        # (D) Add topic features
        if self.topic_layer and topic_features is not None:
            logger.debug(f"Topic Detection features shape before processing: {topic_features.shape}")
            topic_features = topic_features.to(input_ids.device)
            topic_output = self.topic_layer(topic_features)
            logger.debug(f"Topic Detection output shape: {topic_output.shape}")
            combined_output = torch.cat([combined_output, topic_output], dim=-1)
            logger.debug(f"Topic Detection combined_output shape: {combined_output.shape}")
        
        # (E) Add previous labels

        if self.prev_label_layer and prev_label_features is not None:
            logger.info(f"Previous labels features received: {prev_label_features is not None}")
            logger.info(f"Previous labels features shape before processing: {prev_label_features.shape if prev_label_features is not None else 'None'}")
    
            prev_label_features = prev_label_features.to(input_ids.device)
            prev_labels_output = self.prev_label_layer(prev_label_features.float())

            logger.info(f"Previous labels output shape: {prev_labels_output.shape}")

            combined_output = torch.cat([combined_output, prev_labels_output], dim=-1)

            logger.info(f"Previous Sentences combined_output shape: {combined_output.shape}")
        
        logger.info(f"Final combined output shape: {combined_output.shape}")

        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        #logger.debug(f"Logits Shape: {logits.shape}")  # Should be [batch_size, num_labels]
        loss = None
        if labels is not None:
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)  # Ensure 2D labels
            loss = binary_cross_entropy_with_logits(logits, labels)

        return {"logits": logits, "loss": loss}

# ========================================================
# TRAINERS
# ========================================================
    
class CustomTrainer(transformers.Trainer):
    """Custom Trainer with modified loss function for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
        logger.debug(f"Input IDs Shape: {inputs['input_ids'].shape}")
        logger.debug(f"Attention Mask Shape: {inputs['attention_mask'].shape}")

        # Debug lexicon features
        if "lexicon_features" in inputs:
            logger.debug(f"Lexicon Features Shape: {inputs['lexicon_features'].shape}")
        else:
            logger.debug("No lexicon features")

        if "prev_label_features" in inputs:
            prev_label_features = inputs.pop("prev_label_features", None)
        
        # Pop labels for loss computation
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        # Forward pass through the model
        if "prev_label_features" in inputs:
            outputs = model(**inputs, labels=labels, prev_label_features=prev_label_features)
        else:
            outputs = model(**inputs, labels=labels)

        # Retrieve loss and logits from the model's outputs
        logits = outputs["logits"]
        loss = outputs["loss"]
        if loss.dim() > 0:
            loss = loss.mean()  # Reduce to a scalar value if necessary
        logger.debug(f"Logits Shape: {logits.shape}")
        logger.debug(f"Loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

# ========================================================
# CALLBACKS
# ========================================================

class WarmupEvalCallback(TrainerCallback):
    def __init__(self, warmup_epochs=2):
        self.warmup_epochs = warmup_epochs

    def on_evaluate(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch < self.warmup_epochs:
            logger.info(f"Skipping evaluation for warm-up phase (epoch {current_epoch}).")
            control.should_evaluate = False
            control.should_save = False
        else:
            control.should_evaluate = True
            control.should_save = True
        return control

class DynamicPrevLabelCallback(transformers.TrainerCallback):
    def __init__(self, trainer, val_df, labels, tokenizer, device=None):
        """
        val_df: The original raw validation DataFrame.
        labels: List of label names.
        tokenizer: Tokenizer used for inference.
        device (str): Computation device (CPU/GPU).
        """
        self.trainer = trainer
        self.val_df = val_df.copy()  # store a copy of the original raw DF
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def on_evaluate(self, args, state, control, **kwargs):
        # Obtain current model from trainer
        model = self.trainer.model
        model.eval()

        # (A) Add previous sentences label features
        logger.info(f"Epoch {state.epoch}: Running validation. Model: {model is not None}")

        # Dynamically compute the previous labels using our modified function:
        new_prev_feats = add_previous_label_features(
            self.val_df,
            self.val_df,  # assume the labels DataFrame is contained in self.val_df or stored separately
            self.labels,
            is_training=False,
            model=model,
            tokenizer_for_dynamic=self.tokenizer,
            device=self.device
        )

        # (B) Concatenate to text the previous sentences with their predicted labels

        # Reset "Text" column to its original state
        self.val_df["Text"] = self.val_df["original_text"]

        # Rebuild validation dataset with new labels
        self.val_df["Text"] = self.val_df.apply(lambda row: self.concatenate_text_with_prev_labels(row, model, self.tokenizer, self.labels), axis=1)
        logger.info(f"Text: {self.val_df['Text']}")

        # Now update the evaluation dataset:
        # Use the dataset's map method to update that field.
        def update_prev_label(dataset, idx):
            # Here, idx comes from the order of the dataset; we assume it matches self.val_df
            dataset["prev_label_features"] = new_prev_feats[idx]
            dataset["Text"] = self.val_df.iloc[idx]["Text"]
            return dataset

        # Update the dataset with the new prev_label_features
        new_eval_dataset = self.trainer.eval_dataset.map(
            update_prev_label, with_indices=True
        )

        self.trainer.eval_dataset = new_eval_dataset
        logger.info("Updated evaluation dataset with dynamic previous label features.")
        return control

def concatenate_text_with_prev_labels(self, row, model, tokenizer, labels):
        """
        Helper function to reconstruct the text with previous sentences and their labels.
        """
        idx = row.name
        prev_sentences = []

        for offset in [1, 2]:  # Get prev-1 first, then prev-2
            if idx >= offset:
                prev_text = str(self.val_df.iloc[idx - offset]["Text"])
                prev_labels = add_previous_label_features(
                    self.val_df.iloc[idx - offset]["Text"],
                    model, tokenizer, labels
                )

                label_str = " ".join(
                    [label for label, value in zip(labels, prev_labels) if value >= 0.5]
                )

                prev_sentences.append(f"{label_str} {prev_text}")

        current_text = str(row["Text"])

        if prev_sentences:
            return current_text + " </s> " + " </s> ".join(prev_sentences)
        else:
            return current_text