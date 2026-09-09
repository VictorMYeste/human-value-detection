#!/usr/bin/env python3
"""
D7 — Per-genre and pre-machine-translation language composition (R3-C9).

The English ValueEval'24 release preserves the source identity in the Text-ID:
  <LANG>_<n>      -> news article in <LANG>
  <LANG>_M_<n>    -> political manifesto in <LANG>
so the original-language and genre composition can be recovered directly.

Reports, over train+validation+test, the number of source documents (unique
Text-IDs) and sentences per source language and per genre.

Usage:  python3 data-prep/corpus_composition.py   (run from repo root)
"""
import pandas as pd
from pathlib import Path

SPLITS = ["training-english", "validation-english", "test-english"]
LANG = {"EN": "English", "DE": "German", "NL": "Dutch", "FR": "French",
        "IT": "Italian", "EL": "Greek", "BG": "Bulgarian", "HE": "Hebrew",
        "TR": "Turkish"}


def parse(tid):
    parts = tid.split("_")
    lang = parts[0]
    genre = "manifesto" if len(parts) >= 2 and parts[1] == "M" else "news"
    return lang, genre


rows = []
for sp in SPLITS:
    df = pd.read_csv(Path("data") / sp / "sentences.tsv", sep="\t", usecols=["Text-ID"])
    df["lang"], df["genre"] = zip(*df["Text-ID"].map(parse))
    df["split"] = sp.split("-")[0]
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)

docs = all_df.drop_duplicates("Text-ID")
n_docs, n_sent = len(docs), len(all_df)
print(f"TOTAL: {n_docs} source documents, {n_sent} sentences\n")

print("=== By genre ===")
for g in ["news", "manifesto"]:
    d = (docs.genre == g).sum()
    s = (all_df.genre == g).sum()
    print(f"  {g:9s}: {d:4d} docs ({100*d/n_docs:4.1f}%)   {s:6d} sentences ({100*s/n_sent:4.1f}%)")

print("\n=== By source language (before machine translation) ===")
print(f"  {'lang':10s} {'docs':>5} {'doc%':>6} {'sent':>7} {'sent%':>7}")
gd = docs.groupby("lang").size()
gs = all_df.groupby("lang").size()
for code in sorted(gs.index, key=lambda c: -gs[c]):
    name = LANG.get(code, code)
    print(f"  {name:10s} {gd.get(code,0):5d} {100*gd.get(code,0)/n_docs:5.1f}% "
          f"{gs[code]:7d} {100*gs[code]/n_sent:6.1f}%")

print("\n=== Language x genre (documents) ===")
print(pd.crosstab(docs.lang, docs.genre).to_string())
