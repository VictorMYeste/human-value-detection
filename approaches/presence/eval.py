import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.config import MODEL_CONFIG

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")
#logger.setLevel(logging.DEBUG)

# Load model-specific configuration
model_group = "presence"
model_config = MODEL_CONFIG[model_group]

# Load predictions and gold labels
predictions_path = "output/Text-NER-predictions.tsv"
gold_labels_path = "../../data/validation-english/labels-cat.tsv"

# Load the data
predictions = pd.read_csv(predictions_path, sep='\t')
gold_labels = pd.read_csv(gold_labels_path, sep='\t')

# Extract the columns that exist in predictions
relevant_columns = [col for col in predictions.columns if col in gold_labels.columns]
gold_labels = gold_labels[relevant_columns]

# Ensure alignment by merging on Text-ID and Sentence-ID
merged = pd.merge(gold_labels, predictions, on=["Text-ID", "Sentence-ID"], suffixes=("_gold", "_pred"))

if merged.empty:
    raise ValueError("The merge operation resulted in an empty DataFrame. Ensure matching IDs.")

# Extract the new labels
labels = model_config["labels"]

# Prepare gold and predicted arrays
gold = merged[[label + "_gold" for label in labels]].values
pred = merged[[label + "_pred" for label in labels]].values

# Check for missing values
if pd.isnull(gold).any().any() or pd.isnull(pred).any().any():
    raise ValueError("There are missing values in the gold or prediction data.")

if gold.shape != pred.shape:
    raise ValueError(f"Shape mismatch: gold shape {gold.shape}, pred shape {pred.shape}")

# Ensure numeric types
gold = gold.astype(float)
pred = pred.astype(float)

# Debugging output

logger.debug(f"Gold labels shape: {gold.shape}")
logger.debug(f"Predicted labels shape: {pred.shape}")
logger.debug(f"Unique values in gold: {np.unique(gold)}")
logger.debug(f"Unique values in pred: {np.unique(pred)}")

# Apply threshold to predictions
threshold = 0.5
gold = (gold >= threshold).astype(int)  # Convert to binary (0 or 1)
pred = (pred >= threshold).astype(int)  # Convert predictions to binary (0 or 1)

assert set(np.unique(gold)).issubset({0, 1}), "Gold labels are not binary."
assert set(np.unique(pred)).issubset({0, 1}), "Predicted labels are not binary."

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
