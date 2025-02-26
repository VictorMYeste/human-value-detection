import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients

# Load fine-tuned DeBERTa model and tokenizer
model_path = "../approaches/presence/models/Baseline"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Load validation dataset
sentences_path = "../data/validation-english/sentences.tsv"
labels_path = "../data/validatoin-english/labels-cat.tsv"

sentences_df = pd.read_csv(sentences_path, sep="\t")
labels_df = pd.read_csv(labels_path, sep="\t")

# Merge sentences with labels based on Text-ID and Sentence-ID
merged_df = pd.merge(sentences_df, labels_df, on=["Text-ID", "Sentence-ID"])

# Filter only sentences where Presence = 1.0 (moral content present)
moral_sentences = merged_df[merged_df["Presence"] == 1.0]

# Define a function to get attributions for a given sentence
def get_attributions(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = inputs["input_ids"]
    
    # Define attribution method
    ig = IntegratedGradients(model)

    # Get model prediction
    with torch.no_grad():
        predictions = model(**inputs).logits.argmax(dim=1)

    # Compute attributions for the target class
    attributions = ig.attribute(input_ids, target=predictions.item(), internal_batch_size=1)

    return attributions.squeeze().cpu().detach().numpy(), tokenizer.convert_ids_to_tokens(input_ids.squeeze())

# Process a subset of moral sentences for analysis
num_samples = min(10, len(moral_sentences))  # Limit to 10 sentences for visualization
sample_sentences = moral_sentences.sample(num_samples)

# Plot attributions for selected sentences
for idx, row in sample_sentences.iterrows():
    text = row["Text"]
    attributions, tokens = get_attributions(text)

    # Normalize attributions for better visualization
    attributions = np.abs(attributions)
    attributions /= np.max(attributions)

    # Plot attributions as bar chart
    plt.figure(figsize=(12, 4))
    sns.barplot(x=tokens, y=attributions)
    plt.xticks(rotation=90)
    plt.title(f"Attention-Based Attribution for Sentence: {text}")
    plt.xlabel("Tokens")
    plt.ylabel("Importance Score")
    plt.show()
