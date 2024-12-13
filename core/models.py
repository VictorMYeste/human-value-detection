import torch
from torch import nn
import transformers
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

    if model_directory:
        logger.info(f"SAVE to {model_directory}")
        trainer.save_model(model_directory)

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

class EnhancedDebertaModel(nn.Module):
    """Enhanced DeBERTa model with added lexicon feature layer."""
    def __init__(self, pretrained_model, num_labels, id2label, label2id, num_categories):
        super(EnhancedDebertaModel, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)
        #self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        self.lexicon_layer = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classification_head = nn.Linear(self.transformer.config.hidden_size + 128, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, lexicon_features=None, labels=None):
        """Forward pass for the enhanced model."""
        hidden_state = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = hidden_state[:, 0, :] # CLS token representation

        if lexicon_features is not None:
            lexicon_features = lexicon_features.to(input_ids.device)
            lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            if lexicon_output.shape[0] != transformer_output.shape[0]:
                raise ValueError("Batch size mismatch between transformer and lexicon features.")
            combined_output = torch.cat([transformer_output, lexicon_output], dim=-1)
        else:
            combined_output = transformer_output
        
        #logits_with_lexicon = self.classification_head(combined_output)
        #logits_without_lexicon = self.classification_head(transformer_output)
        #logger.debug(f"Logits with lexicon: {logits_with_lexicon[:5]}")
        #logger.debug(f"Logits without lexicon: {logits_without_lexicon[:5]}")

        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        logger.debug(f"Logits Shape: {logits.shape}")  # Should be [batch_size, num_labels]
        return {"logits": logits}

# ========================================================
# TRAINERS
# ========================================================
    
class CustomTrainer(transformers.Trainer):
    """Custom Trainer with modified loss function for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
        logger.debug(f"Input IDs Shape: {inputs['input_ids'].shape}")
        logger.debug(f"Attention Mask Shape: {inputs['attention_mask'].shape}")
        if "lexicon_features" in inputs:
            logger.debug(f"Lexicon Features Shape: {inputs['lexicon_features'].shape}")
        else:
            logger.debug("No lexicon features")
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        outputs = model(**inputs)
        logits = outputs["logits"]
        logger.debug(f"Logits Shape: {logits.shape}")

        loss = binary_cross_entropy_with_logits(logits, labels.float())  # BCE loss
        logger.debug(f"Loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss