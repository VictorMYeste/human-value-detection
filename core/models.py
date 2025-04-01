import torch
from torch import nn
import transformers
from transformers import TrainerCallback
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.distributed as dist
import os
import pandas as pd
import numpy as np
import spacy

from core.dataset_utils import LEXICON_COMPUTATION_FUNCTIONS, add_previous_label_features, compute_lexicon_scores, compute_ner_embeddings, compute_precomputed_scores
from core.lexicon_utils import load_embeddings, load_lexicon
from core.config import LEXICON_PATHS, SCHWARTZ_VALUE_LEXICON
from core.topic_detection import TopicModeling
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
        input_ids=None,
        inputs_embeds=None,  # Add inputs_embeds for Captum support
        attention_mask=None,
        lexicon_features=None,
        ner_features=None,
        topic_features=None,
        prev_label_features=None,
        labels=None
    ):
        """Forward pass for the enhanced model."""

        # Ensure input_ids remains as LongTensor (int64)
        if input_ids is not None:
            input_ids = input_ids.to(torch.long)

        logger.debug(f"Lexicon features received: {lexicon_features is not None}")
        logger.debug(f"NER features received: {ner_features is not None}")

        if lexicon_features is not None:
            logger.debug(f"Lexicon feature shape: {lexicon_features.shape}")

        if ner_features is not None:
            logger.debug(f"NER feature shape: {ner_features.shape}")

        # Extract transformer embeddings
        if inputs_embeds is not None:
            hidden_state = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        else:
            hidden_state = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
            logger.debug(f"Previous labels features received: {prev_label_features is not None}")
            logger.debug(f"Previous labels features shape before processing: {prev_label_features.shape if prev_label_features is not None else 'None'}")
    
            prev_label_features = prev_label_features.to(input_ids.device)
            prev_labels_output = self.prev_label_layer(prev_label_features.float())

            logger.debug(f"Previous labels output shape: {prev_labels_output.shape}")

            for i in range(min(5, prev_label_features.shape[0])):  # Print first 5 only
                logger.debug(f"Sample {i}: Prev Labels Input: {prev_label_features[i].cpu().numpy()}")

            combined_output = torch.cat([combined_output, prev_labels_output], dim=-1)

            logger.debug(f"Previous Sentences combined_output shape: {combined_output.shape}")
        
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
        logger.debug(f"Keys in inputs: {inputs.keys()}")
        logger.debug(f"Input IDs Shape: {inputs['input_ids'].shape}")
        logger.debug(f"Attention Mask Shape: {inputs['attention_mask'].shape}")

        # Debug lexicon features
        if "lexicon_features" in inputs:
            logger.debug(f"Lexicon Features Shape: {inputs['lexicon_features'].shape}")
        else:
            logger.debug("No lexicon features")
        
        # Pop labels for loss computation
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        # Forward pass through the model
        if "prev_label_features" in inputs:
            prev_label_features = inputs.pop("prev_label_features", None)
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
    def __init__(
            self,
            trainer,
            val_df,
            labels_df,
            labels,
            tokenizer,
            device=None,
            num_categories=0,
            lexicon=None,
            ner_features=None,
            topic_detection=None):
        """
        val_df: The original raw validation DataFrame.
        labels: List of label names.
        tokenizer: Tokenizer used for inference.
        device (str): Computation device (CPU/GPU).
        """
        self.trainer = trainer
        self.val_df = val_df.copy()
        self.labels_df = labels_df.copy()
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_categories=num_categories
        self.lexicon=lexicon
        self.ner_features=ner_features
        self.topic_detection=topic_detection

    def on_evaluate(self, args, state, control, **kwargs):
        # Obtain current model from trainer
        model = self.trainer.model
        model.eval()

        # (A) Dynamically compute lexicon features for validation
        val_lexicon = None
        new_lexicon_feats = []
        val_num_cat = 0

        if self.lexicon:
            if self.lexicon in ["LIWC-22", "eMFD", "MFD-20", "MJD"]:
                val_lexicon, val_num_cat = load_lexicon(self.lexicon, LEXICON_PATHS[self.lexicon+"-validation"])
                logger.debug(f"Lexicon produced from LIWC 22 software with val_num_cat = {val_num_cat}")
            else:
                val_lexicon, val_num_cat = load_embeddings(self.lexicon)
            new_lexicon_feats = self.compute_lexicon_features(self.val_df, self.lexicon, val_lexicon, val_num_cat)
            logger.debug(f"New lexicon features for validation: {new_lexicon_feats[:5]}")

        # (B) Add previous sentences label features
        logger.debug(f"Epoch {state.epoch}: Running validation. Model: {model is not None}")

        # Dynamically compute the previous labels using our modified function:
        new_prev_feats = add_previous_label_features(
            df=self.val_df,
            labels_df=self.labels_df,
            labels=self.labels,
            is_training=False,
            model=model,
            tokenizer_for_dynamic=self.tokenizer,
            lexicon=self.lexicon,
            lexicon_embeddings=val_lexicon,
            num_categories=val_num_cat
        )

        # (C) Dynamically compute NER features for validation
        if self.ner_features:
            new_ner_feats = self.compute_ner_features(self.val_df)

        # (D) Dynamically compute topic features for validation
        if self.topic_detection:
            new_topic_feats = self.compute_topic_features(self.val_df)

        # Ensure new_prev_feats has the correct shape
        for i in range(len(new_prev_feats)):
            if len(new_prev_feats[i]) < 2 * len(self.labels):
                padding = [0.0] * (2 * len(self.labels) - len(new_prev_feats[i]))
                new_prev_feats[i].extend(padding)  # Pad with zeros if missing
            
            if self.lexicon and len(new_lexicon_feats[i]) < len(self.labels):
                padding = [0.0] * (len(self.labels) - len(new_lexicon_feats[i]))
                new_lexicon_feats[i].extend(padding)  # Pad with zeros if missing

            if self.ner_features and len(new_ner_feats[i]) < len(self.labels):
                padding = [0.0] * (len(self.labels) - len(new_ner_feats[i]))
                new_ner_feats[i].extend(padding)  # Pad with zeros if missing

            if self.topic_detection and len(new_topic_feats[i]) < len(self.labels):
                padding = [0.0] * (len(self.labels) - len(new_topic_feats[i]))
                new_topic_feats[i].extend(padding)  # Pad with zeros if missing

        # (E) Concatenate to text the previous sentences with their predicted labels
        self.val_df["Text"] = self.val_df["Original_Text"]
        self.val_df["Text"] = self.val_df.apply(
            lambda row: self.concatenate_text_with_prev_labels(row, self.labels, new_prev_feats), axis=1
        )

        logger.debug(f"Validation dataset first samples:\n{self.val_df[['Text', 'Original_Text']].head().to_string().strip()}")

        for i in range(min(5, len(new_prev_feats))):  # Print first 5 for comparison
            logger.debug(f"Validation Row {i}: prev_label_features = {new_prev_feats[i]}")
            logger.debug(f"Validation Row {i}: prev_1_labels = {new_prev_feats[i][:len(self.labels)]}")
            logger.debug(f"Validation Row {i}: prev_2_labels = {new_prev_feats[i][len(self.labels):]}")

        # Now update the evaluation dataset:
        # Use the dataset's map method to update that field.
        def update_prev_label(dataset, idx):
            # Here, idx comes from the order of the dataset; we assume it matches self.val_df
            dataset["prev_label_features"] = new_prev_feats[idx]
            if self.lexicon:
                dataset["lexicon_features"] = new_lexicon_feats[idx]
            if self.ner_features:
                dataset["ner_features"] = new_ner_feats[idx]
            if self.topic_detection:
                dataset["topic_features"] = new_topic_feats[idx]
            dataset["Text"] = self.val_df.iloc[idx]["Text"]
            return dataset

        # Update the dataset with the new prev_label_features
        new_eval_dataset = self.trainer.eval_dataset.map(update_prev_label, with_indices=True)

        self.trainer.eval_dataset = new_eval_dataset
        logger.info("Updated evaluation dataset with dynamic previous label features.")
        return control
    
    def compute_lexicon_features(self, df, lexicon, lexicon_embeddings, num_categories):
        """
        Compute lexicon features for the validation dataset. This function should generate
        lexicon features dynamically for each sentence in the validation dataset.
        """
        lexicon_features = []
        
        for _, row in df.iterrows():
            text = row["Text"]
            if lexicon in LEXICON_COMPUTATION_FUNCTIONS:
                # Token-based approach
                lexicon_feats = compute_lexicon_scores(text, lexicon, lexicon_embeddings, self.tokenizer, num_categories)
            else:
                # Precomputed (row-level) lexicon (e.g. LIWC-22 software generated)
                lexicon_feats = compute_precomputed_scores(row, lexicon_embeddings, num_categories)
            # If the above can produce NaNs, fix them
            lexicon_feats = [0.0 if (isinstance(x, float) and np.isnan(x)) else x for x in lexicon_feats]

            lexicon_features.append(lexicon_feats)

        return lexicon_features

    def compute_ner_features(self, df):
        """
        Compute NER features for the validation dataset. This function should generate
        NER features dynamically for each sentence in the validation dataset.
        """
        ner_features = []
        nlp = spacy.load("en_core_web_sm")
        for _, row in df.iterrows():
            text = row["Text"]
            # You will need a function that computes NER features for each text
            ner_feats = compute_ner_embeddings(text, nlp)
            ner_features.append(ner_feats)

        return ner_features

    def compute_topic_features(self, df):
        """
        Compute topic features for the validation dataset. This function should generate
        topic features dynamically for each sentence in the validation dataset.
        """
        topic_features = []
        texts = df["Text"].tolist()
        # You will need a function that computes topic features for each text
        topic_model = TopicModeling(method=self.topic_detection)
        topic_vectors = None
        topic_vectors = topic_model.fit_transform(texts)
        for row_vector in topic_vectors:  
            topic_features.append(row_vector.tolist())

        return topic_features

    def concatenate_text_with_prev_labels(self, row, labels, new_prev_feats):
        """
        Helper function to reconstruct the text with previous sentences and their labels.
        """
        idx = row.name
        text_id = row["Text-ID"]
        sentence_id = row["Sentence-ID"]
        prev_sentences = []

        for offset in [1, 2]:  # Get prev-1 first, then prev-2
            prev_idx = self.val_df[(self.val_df["Text-ID"] == text_id) & (self.val_df["Sentence-ID"] == sentence_id - offset)].index

            if len(prev_idx) > 0:
                prev_idx = prev_idx[0]
                prev_text = str(self.val_df.iloc[prev_idx]["Text"])
                
                # Extract correct previous label features
                prev_feat = new_prev_feats[prev_idx]

                # Assign prev-1 or prev-2 labels
                prev_labels = prev_feat[:len(labels)] if offset == 1 else prev_feat[len(labels):]

                # Ensure labels are formatted and added
                label_str = " ".join(f"<{label}>" for label, value in zip(labels, prev_labels) if value >= 0.5)
                label_str = label_str if label_str else "<NONE>"  # Ensure a placeholder when no label is present

                prev_sentences.append(f"{label_str} {prev_text}")

                # Debugging: Ensure correct previous label extraction
                logger.debug(f"Row {idx} (Text-ID {text_id}, Sentence-ID {sentence_id}) Offset {offset}: Extracted prev_labels = {prev_labels}")
                logger.debug(f"Row {idx} Offset {offset}: label_str before formatting = {label_str}")

        current_text = str(row["Text"])

        if prev_sentences:
            full_text = current_text + " </s> " + " </s> ".join(prev_sentences)
        else:
            full_text = current_text

        logger.debug(f"Concatenated text for row {idx}:\n{full_text}\n")

        return full_text