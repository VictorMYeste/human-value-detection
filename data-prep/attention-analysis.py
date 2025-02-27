import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
from collections import defaultdict

from core.models import EnhancedDebertaModel
from core.config import MODEL_CONFIG

# Load model-specific configuration
model_group = "presence"
model_config = MODEL_CONFIG[model_group]

# SETUP
model_path = "../approaches/presence/models/Baseline"

# Load tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = EnhancedDebertaModel(
    pretrained_model=model_config["pretrained_model"],
    config=transformers.AutoConfig.from_pretrained(model_path),
    num_labels=len(model_config["labels"]),
    id2label={idx: label for idx, label in enumerate(model_config["labels"])},
    label2id={label: idx for idx, label in enumerate(model_config["labels"])}
)

# Load model weights
state_dict = load_file(model_path + "/model.safetensors")
model.load_state_dict(state_dict)
model.eval()  # Set to evaluation mode

sigmoid = torch.nn.Sigmoid()

# Load validation dataset
sentences_df = pd.read_csv("../data/validation-english/sentences.tsv", sep="\t")
labels_df = pd.read_csv("../data/validation-english/labels-cat.tsv", sep="\t")

# Merge sentences with labels based on Text-ID and Sentence-ID
merged_df = pd.merge(sentences_df, labels_df, on=["Text-ID", "Sentence-ID"])

merged_df = merged_df.head(10)

# Dictionary to store aggregated token attributions
token_attributions = defaultdict(list)
presence_attributions = defaultdict(list)
absence_attributions = defaultdict(list)

# Define function to compute attributions
def get_attributions(sentence):
    encoded_sentences = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Remove token_type_ids if present
    encoded_sentences.pop("token_type_ids", None)

    # Convert input_ids to LongTensor
    encoded_sentences["input_ids"] = encoded_sentences["input_ids"].to(torch.long)
    encoded_sentences["attention_mask"] = encoded_sentences["attention_mask"].to(torch.long)

    # Extract embeddings
    with torch.no_grad():
        input_embeds = model.transformer.embeddings.word_embeddings(encoded_sentences["input_ids"])

    # Enable gradient computation
    input_embeds.requires_grad_()

    # Define attribution method
    def model_forward(embeds, attention_mask):
        return model(inputs_embeds=embeds, attention_mask=attention_mask)["logits"]
    
    ig = IntegratedGradients(model_forward)

    # Get model prediction
    with torch.no_grad():
        model_output = model(
            input_ids=encoded_sentences["input_ids"],
            attention_mask=encoded_sentences["attention_mask"]
        )
    
    sentences_predictions = sigmoid(model_output["logits"]).squeeze().detach().cpu().numpy()

    print("Raw model predictions for sentence:", sentences_predictions)
    
    if sentences_predictions.ndim == 0:  # If scalar output
        target_class = int(sentences_predictions > 0.5)  # Convert scalar to Python int
    else:
        target_class = int(sentences_predictions.argmax())  # Convert array output to int
        
    # Compute attributions
    attributions = ig.attribute(
        input_embeds,
        target=target_class,  # Compute attributions for predicted class
        additional_forward_args=(encoded_sentences["attention_mask"],)
    )

    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.abs(attributions).mean(axis=1)  # Aggregate across hidden dimensions

    tokens = tokenizer.convert_ids_to_tokens(encoded_sentences["input_ids"].squeeze())
    print("Sample tokenized output:", tokens[:50])

    decoded_sentence = tokenizer.decode(encoded_sentences["input_ids"].squeeze())
    print("Decoded sentence:", decoded_sentence)

    clean_tokens = [tokenizer.convert_tokens_to_string([token]) for token in tokens]

    return clean_tokens, attributions, target_class

# Process entire validation dataset
for idx, row in merged_df.iterrows():
    text = row["Text"]
    tokens, attributions, label = get_attributions(text)

    for token, attr in zip(tokens, attributions):
        token_attributions[token].append(attr)

        if label == 1:
            presence_attributions[token].append(attr)
        else:
            absence_attributions[token].append(attr)

# Compute average attributions per token
mean_presence_attributions = {token: np.mean(attrs) for token, attrs in presence_attributions.items()}
mean_absence_attributions = {token: np.mean(attrs) for token, attrs in absence_attributions.items()}

# Sort tokens by highest contribution
top_presence_tokens = sorted(mean_presence_attributions.items(), key=lambda x: x[1], reverse=True)[:20]
top_absence_tokens = sorted(mean_absence_attributions.items(), key=lambda x: x[1], reverse=True)[:20]

print(f"Presence Attributions: {len(presence_attributions)} tokens")
print(f"Absence Attributions: {len(absence_attributions)} tokens")

if presence_attributions:
    print("Sample presence tokens:", list(presence_attributions.keys())[:10])
    print("Sample presence scores:", list(presence_attributions.values())[:10])
else:
    print("No tokens detected for presence.")

# Convert to DataFrames for visualization
presence_df = pd.DataFrame(top_presence_tokens, columns=["Token", "Attribution"])
absence_df = pd.DataFrame(top_absence_tokens, columns=["Token", "Attribution"])

# Plot Top 20 Presence Tokens
plt.figure(figsize=(10, 5))
sns.barplot(x="Attribution", y="Token", data=presence_df, palette="Blues_r")
plt.title("Top 20 Tokens Contributing to Presence Label")
plt.xlabel("Mean Attribution Score")
plt.ylabel("Token")
plt.tight_layout()
plt.savefig("files/attribution_presence_1.png", dpi=300)
plt.close()

# Plot Top 20 Absence Tokens
plt.figure(figsize=(10, 5))
sns.barplot(x="Attribution", y="Token", data=absence_df, palette="Reds_r")
plt.title("Top 20 Tokens Contributing to Absence of Presence Label")
plt.xlabel("Mean Attribution Score")
plt.ylabel("Token")
plt.tight_layout()
plt.savefig("files/attribution_presence_0.png", dpi=300)
plt.close()

print("Plots saved as attribution_presence_1.png and attribution_presence_0.png")

tokens, attributions, _ = get_attributions(text)
print(f"Tokens: {tokens}")
print(f"Attributions: {attributions}")