import torch
from torch import nn
import transformers
from transformers import TrainerCallback
from torch.nn.functional import binary_cross_entropy_with_logits
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")
#logger.setLevel(logging.DEBUG)

# ========================================================
# UTILS
# ========================================================

def save_model(trainer, model_name, model_directory):
    if model_name:
        logger.info(f"UPLOAD to https://huggingface.co/{model_name} (using HF_TOKEN environment variable)")
        # trainer.push_to_hub()

    if model_directory and model_name:
        logger.info(f"SAVE to {model_directory}")
        trainer.save_model(f"{model_directory}/{model_name}")

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
            # nn.GroupNorm(num_groups, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim),
            # nn.GroupNorm(num_groups, output_dim),
            nn.ReLU()
        )
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.linear_layers(x) + self.projection(x)  # Add residual connection

class EnhancedDebertaModel(nn.Module):
    """Enhanced DeBERTa model with added lexicon feature layer."""
    def __init__(self, pretrained_model, num_labels, id2label, label2id, num_categories, multilayer = False, num_groups=8):
        super(EnhancedDebertaModel, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)

        # Optional lexicon feature processing
        #self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        self.lexicon_layer = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU()
        ) if num_categories > 0 else None

        # Multi-layer processing for transformer embeddings
        self.multilayer = multilayer
        if multilayer:
            """
            self.text_embedding_layer = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, 512),
                nn.GroupNorm(num_groups, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.GroupNorm(num_groups, 256),
                nn.ReLU()
            )
            """
            self.text_embedding_layer = ResidualBlock(self.transformer.config.hidden_size, 256)
            hidden_size = 256
        else:
            hidden_size = self.transformer.config.hidden_size

        # Classification head
        input_dim = hidden_size + 128 if self.lexicon_layer else hidden_size # Adjust input dimension if lexicon features are used
        self.classification_head = nn.Linear(input_dim, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, lexicon_features=None, labels=None):
        """Forward pass for the enhanced model."""
        # Extract transformer embeddings
        hidden_state = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = hidden_state[:, 0, :] # CLS token representation

        # Process transformer embeddings through additional layers
        if self.multilayer:
            text_embeddings = self.text_embedding_layer(transformer_output)
        else:
            text_embeddings = transformer_output

        # Handle lexicon features if provided
        if self.lexicon_layer and lexicon_features is not None:
            lexicon_features = lexicon_features.to(input_ids.device)
            lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            if lexicon_output.shape[0] != text_embeddings.shape[0]:
                raise ValueError("Batch size mismatch between transformer and lexicon features.")
            combined_output = torch.cat([text_embeddings, lexicon_output], dim=-1)
        else:
            # Use only text embeddings if lexicon features are not provided
            combined_output = text_embeddings
        
        #logits_with_lexicon = self.classification_head(combined_output)
        #logits_without_lexicon = self.classification_head(transformer_output)
        #logger.debug(f"Logits with lexicon: {logits_with_lexicon[:5]}")
        #logger.debug(f"Logits without lexicon: {logits_without_lexicon[:5]}")

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
        
        # Pop labels for loss computation
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        # Forward pass through the model
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
        current_epoch = int(state.epoch) + 1
        if current_epoch <= self.warmup_epochs:
            logger.info(f"Skipping evaluation for warm-up phase (epoch {current_epoch}).")
            control.should_evaluate = False
            control.should_save = False
        else:
            control.should_evaluate = True
            control.should_save = True
        return control