import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Load predictions and gold labels
predictions_path = "output-v1/predictions.tsv"
gold_labels_path = "../../data/test-english/labels-cat.tsv"

# Load the data
predictions = pd.read_csv(predictions_path, sep='\t')
gold_labels = pd.read_csv(gold_labels_path, sep='\t')

# Ensure alignment by merging on Text-ID and Sentence-ID
merged = pd.merge(gold_labels, predictions, on=["Text-ID", "Sentence-ID"], suffixes=("_gold", "_pred"))

# Extract the new labels
labels = ["Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance"]

# Prepare gold and predicted arrays
gold = merged[[label + "_gold" for label in labels]].values
pred = merged[[label + "_pred" for label in labels]].values

# Apply threshold to predictions
threshold = 0.5
gold = (gold >= threshold).astype(int)  # Convert to binary (0 or 1)
pred = (pred >= threshold).astype(int)  # Convert predictions to binary (0 or 1)

# Compute metrics
precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)

# Display results per label
for i, label in enumerate(labels):
    print(f"{label}:")
    print(f"  Precision: {precision[i]:.2f}")
    print(f"  Recall:    {recall[i]:.2f}")
    print(f"  F1-Score:  {f1[i]:.2f}\n")

# Compute macro-average F1-score
macro_f1 = np.mean(f1)
print(f"Macro-Average F1-Score: {macro_f1:.2f}")
