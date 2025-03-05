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
from tqdm import tqdm
import re

from core.models import EnhancedDebertaModel
from core.config import MODEL_CONFIG

# -- 1) CONFIG -------------------
TOP_K = 300        # Number of top tokens to display
MIN_FREQ = 5       # Filter out tokens that occur < 5 times

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

# merged_df = merged_df.head(100)

# Dictionary to store aggregated token attributions
token_attributions = defaultdict(list)
presence_attributions = defaultdict(list)
absence_attributions = defaultdict(list)

# Dictionary to track frequency of each token (in the entire dataset)
token_frequency = defaultdict(int)

def get_attributions(sentence):
    """Compute integrated gradients for a single sentence."""
    # 1. Tokenize
    encoded_sentences = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Remove token_type_ids if present
    encoded_sentences.pop("token_type_ids", None)

    # Convert input_ids to LongTensor
    encoded_sentences["input_ids"] = encoded_sentences["input_ids"].to(torch.long)
    encoded_sentences["attention_mask"] = encoded_sentences["attention_mask"].to(torch.long)

    # 2. Extract embeddings with no_grad
    with torch.no_grad():
        input_embeds = model.transformer.embeddings.word_embeddings(encoded_sentences["input_ids"])

    # 3. Enable gradient for integrated gradients
    input_embeds.requires_grad_()

    # Define attribution method
    def model_forward(embeds, attention_mask):
        logits = model(inputs_embeds=embeds, attention_mask=attention_mask)["logits"]
        probs = sigmoid(logits)  # Convert logits to probabilities
        return probs  # Return probabilities, not logits
    
    ig = IntegratedGradients(model_forward)

    # 4. Model prediction
    with torch.no_grad():
        model_output = model(
            input_ids=encoded_sentences["input_ids"],
            attention_mask=encoded_sentences["attention_mask"]
        )
    
    sentences_predictions = sigmoid(model_output["logits"]).squeeze().detach().cpu().numpy()

    # print("Model output shape:", sentences_predictions.shape)
    # print("Model raw output:", sentences_predictions)

    # 5. Determine target class
    if sentences_predictions.ndim == 0:  # If scalar output
        target_class = int(sentences_predictions > 0.5)  # Binary classification threshold
    else:
        target_class = (sentences_predictions > 0.5).astype(int)  # Vectorized thresholding

    # 6. Compute attributions for predicted class=0 (presence) or separate usage
    attributions = ig.attribute(
        input_embeds,
        target=0,
        additional_forward_args=(encoded_sentences["attention_mask"],)
    )

    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.abs(attributions).mean(axis=1)  # Aggregate across hidden dimensions

    # 7. Normalize
    max_abs_value = max(abs(np.min(attributions)), abs(np.max(attributions)))
    if max_abs_value > 0:
        attributions = attributions / max_abs_value

    # 8. Convert input_ids -> tokens
    decoded_sentence = tokenizer.decode(encoded_sentences["input_ids"].squeeze())
    # Filter tokens (alphabetic only)
    clean_tokens = [t for t in decoded_sentence.split() 
                    if re.match(r'^[a-zA-Z]+$', t)]

    return clean_tokens, attributions, target_class

# -- 2) PROCESS DATASET -----------
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing Sentences", unit="sentence"):
    text = row["Text"]
    tokens, attributions, label = get_attributions(text)

    # Update frequency counts
    for t in tokens:
        token_frequency[t] += 1

    # Store integrated gradients
    for token, attr in zip(tokens, attributions):
        token_attributions[token].append(attr)
        if label == 1:
            presence_attributions[token].append(attr)
        else:
            absence_attributions[token].append(attr)

# -- 3) COMPUTE MEAN ATTRIBUTIONS & FILTER ---------------
mean_presence_attributions = {}
for token, attrs in presence_attributions.items():
    mean_presence_attributions[token] = np.mean(attrs)

mean_absence_attributions = {}
for token, attrs in absence_attributions.items():
    mean_absence_attributions[token] = np.mean(attrs)

# (A) Filter out tokens that appear < MIN_FREQ times
filtered_presence = [(tok, score) for tok, score in mean_presence_attributions.items()
                     if token_frequency[tok] >= MIN_FREQ]
filtered_absence = [(tok, score) for tok, score in mean_absence_attributions.items()
                    if token_frequency[tok] >= MIN_FREQ]

# (B) Sort & select top K
top_presence_tokens = sorted(filtered_presence, key=lambda x: x[1], reverse=True)[:TOP_K]
top_absence_tokens  = sorted(filtered_absence,  key=lambda x: x[1], reverse=True)[:TOP_K]

"""
print(f"Presence Attributions: {len(presence_attributions)} tokens")
print(f"Absence Attributions: {len(absence_attributions)} tokens")

if presence_attributions:
    print("Sample presence tokens:", list(presence_attributions.keys())[:10])
    print("Sample presence scores:", list(presence_attributions.values())[:10])
else:
    print("No tokens detected for presence.")
"""

# -- 4) CREATE DATAFRAMES WITH FREQUENCY ---------------
presence_df = pd.DataFrame([
    (tok, score, token_frequency[tok]) for tok, score in top_presence_tokens
], columns=["Token", "Attribution", "Frequency"])

absence_df = pd.DataFrame([
    (tok, score, token_frequency[tok]) for tok, score in top_absence_tokens
], columns=["Token", "Attribution", "Frequency"])

# -- 5) SAVE RESULTS ------------------------------------
presence_df.to_csv("files/top_presence_1_tokens.txt", sep="\t", index=False, header=True)
absence_df.to_csv("files/top_presence_0_tokens.txt", sep="\t", index=False, header=True)
print("Top presence and absence tokens saved with frequencies:")
print("files/top_presence_1_tokens.txt")
print("files/top_presence_0_tokens.txt")