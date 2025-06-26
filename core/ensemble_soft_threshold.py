"""
Tune a global decision threshold for an ensemble by averaging
the validation‐set probabilities of a fixed list of models,
then sweeping thresholds to maximize Macro-F1.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# === Configuration ===
MODELS = [
    "../approaches/moral-values/output/Baseline-val.tsv",
    "../approaches/moral-values/output/Previous-Sentences-2-val.tsv",
    "../approaches/moral-values/output/Lex-LIWC-22-val.tsv",
    "../approaches/moral-values/output/TD-BERTopic-val.tsv",
]
GOLD     = "../data/validation-english/labels-cat.tsv"
ID_COLS  = ['Text-ID','Sentence-ID']

# === load & extract probabilities ===
df_list = []
for path in MODELS:
    df = pd.read_csv(path, sep='\t').set_index(ID_COLS)
    float_cols = df.select_dtypes(include=float).columns
    df_list.append(df[float_cols])

# align on index & average
prob_df = pd.concat(df_list, axis=1).groupby(level=0, axis=1).mean()

# remember exactly which fine-grained labels we care about
fg_labels = df_list[0].columns

# load gold & align
gold_df = pd.read_csv(GOLD, sep='\t').set_index(ID_COLS)

# --- keep only labels that appear in both predictions and gold ---
common = prob_df.columns.intersection(gold_df.columns)
if len(common)==0:
    raise RuntimeError("No overlapping label columns between models and gold!")
prob_df = prob_df[common]
gold_df = gold_df[common]

# --- align rows so index and order match ---
idx = prob_df.index.intersection(gold_df.index)
prob_df = prob_df.loc[idx]
gold_df = gold_df.loc[idx]

y_true = gold_df.values.astype(int)
probs  = prob_df.values

# === sweep thresholds ===
best_T, best_f1 = 0.0, -1.0
for T in np.linspace(0, 1, 101):
    y_pred = (probs >= T).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, best_T = f1, T
    print(f"T={T:.2f} → Macro-F1={f1:.4f}")

print("\n" + "-"*50)
print(f"→ Best threshold = {best_T:.2f} with Macro-F1 = {best_f1:.4f}")
