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
            self.prev_label_size = num_labels
            self.prev_label_layer = nn.Sequential(
                nn.Linear(self.prev_label_size, 16),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
        else:
            self.prev_label_size = 0
            self.prev_label_layer = None

        # Classification head. Combine all features
        input_dim = hidden_size
        if self.lexicon_layer:
            input_dim += 128
        if self.ner_layer:
            input_dim += 128
        if self.topic_layer:
            input_dim += 128
        if self.prev_label_layer is not None:
            input_dim += 16

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
        labels=None,
        **kwargs
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

        # (B) Add lexicon features
        if self.lexicon_layer and lexicon_features is not None:
            logger.debug(f"Lexicon features shape before processing: {lexicon_features.shape}")
            lexicon_features = lexicon_features.to(input_ids.device)
            #lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            lexicon_output = self.lexicon_layer(lexicon_features)
            logger.debug(f"Lexicon output shape: {lexicon_output.shape}")
            combined_output = torch.cat([combined_output, lexicon_output], dim=-1)
        
        # (C) Add NER features
        if self.ner_layer and ner_features is not None:
            logger.debug(f"NER features shape before processing: {ner_features.shape}")
            ner_features = ner_features.to(input_ids.device)
            ner_output = self.ner_layer(ner_features)
            logger.debug(f"NER output shape: {ner_output.shape}")
            combined_output = torch.cat([combined_output, ner_output], dim=-1)

        # (D) Add topic features
        if self.topic_layer and topic_features is not None:
            logger.debug(f"Topic Detection features shape before processing: {topic_features.shape}")
            topic_features = topic_features.to(input_ids.device)
            topic_output = self.topic_layer(topic_features)
            logger.debug(f"Topic Detection output shape: {topic_output.shape}")
            combined_output = torch.cat([combined_output, topic_output], dim=-1)
        
        # (E) Add previous labels
        if self.prev_label_layer and prev_label_features is not None:
            logger.debug(f"Previous labels features shape before processing: {prev_label_features.shape}")
            prev_label_features = prev_label_features.to(input_ids.device)
            prev_labels_output = self.prev_label_layer(prev_label_features.float())
            logger.debug(f"Previous labels output shape: {prev_labels_output.shape}")
            combined_output = torch.cat([combined_output, prev_labels_output], dim=-1)
        
        logger.debug(f"Final combined output shape: {combined_output.shape}")

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

        prev_label_features = inputs.pop("prev_label_features", None)
        
        # Pop labels for loss computation
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        # Forward pass through the model
        outputs = model(**inputs, labels=labels, prev_label_features=prev_label_features)

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
    def __init__(self, val_df, labels, tokenizer, device=None):
        """
        val_df: The original raw validation DataFrame.
        labels: List of label names.
        tokenizer: Tokenizer used for inference.
        """
        self.val_df = val_df.copy()  # store a copy of the original raw DF
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def on_evaluate(self, args, state, control, **kwargs):
        # Obtain current model from trainer
        trainer = kwargs["trainer"]
        model = trainer.model
        model.eval()
        # Dynamically compute the previous labels using our modified function:
        # (This function now uses model & tokenizer to sequentially compute predictions.)
        new_prev_feats = add_previous_label_features(
            self.val_df,
            self.val_df,  # assume the labels DataFrame is contained in self.val_df or stored separately
            self.labels,
            is_training=False,
            model=model,
            tokenizer_for_dynamic=self.tokenizer,
            device=self.device
        )
        # Now update the evaluation dataset:
        # Use the dataset's map method to update that field.
        def update_prev_label(dataset, idx):
            # Here, idx comes from the order of the dataset; we assume it matches self.val_df
            dataset["prev_label_features"] = new_prev_feats[idx]
            return dataset

        # Update the dataset with the new prev_label_features
        new_eval_dataset = trainer.eval_dataset.map(
            update_prev_label, with_indices=True
        )
        trainer.eval_dataset = new_eval_dataset
        logger.info("Updated evaluation dataset with dynamic previous label features.")
        return control