# Human Value Detection

Code for our work on sentence-level detection of **Schwartz human values** on the ValueEval/ValuesML corpus.  
This repository contains implementations, experiments, and pre-/post-processing used in two companion papers:

1. **Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum**
2. **Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? When Hard Gating Hurts**  

Both papers are currently available as arXiv preprints:

- Yeste & Rosso, *Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? When Hard Gating Hurts*, arXiv:-
- Yeste & Rosso, *Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum*, arXiv:-

If you use this code, models, or results in your research, **please cite at least one of the two papers** (see [Citation](#citation)).

---

## Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)
  - [Baseline predictions](#baseline-predictions)
  - [Evaluation](#evaluation)
- [Reproducing the papers](#reproducing-the-papers)
  - [Higher-Order values paper](#1-higher-order-values-paper)
  - [Moral presence & ensembles paper](#2-moral-presence--ensembles-paper)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository focuses on **sentence-level detection** of the 19 refined Schwartz basic values, on top of the ValueEval’24 / ValuesML corpus. We study:

- A **binary “moral presence” task**: does a sentence express *any* Schwartz value?
- **19-way multi-label value detection**: which of the 19 values are expressed in the sentence?
- The role of:
  - **Higher-order (HO) value categories** and hierarchical gating,
  - **Moral presence gates**,
  - **Lightweight auxiliary signals** (lexica, short context, topics),
  - **Supervised DeBERTa models vs. instruction-tuned LLMs** (7–9B),
  - **Small, compute-frugal ensembles** under an 8 GB GPU constraint.

The code is shared between the two papers; they correspond to two “views” over the same codebase and experiments.

---

## Tasks

We work with the English machine-translated split of ValueEval’24 / ValuesML, at the **sentence** level.

For each sentence \(s\):

- The dataset provides labels for the **19 Schwartz basic values**, with separate “attained” and “constrained” indicators.
- We collapse these into a single **binary value label** per value:
  - \(y_{s,v} = 1\) if value \(v\) is either attained or constrained in \(s\), else \(0\).
- We then define a **moral presence label**:
  - \(z_s = 1\) if any of the 19 values is active in \(s\), else \(0\).

The two main prediction problems we study are:

1. **Moral presence**: predict \(z_s\) (binary).
2. **Value detection**: predict the 19-dimensional vector \(\{y_{s,v}\}_{v=1}^{19}\) (multi-label).

---

## Repository structure

At a high level:

- `predict.py`  
  Entry point to **run inference** for a given model on a given split (e.g. validation).

- `eval.py`  
  Entry point to **evaluate** predictions on validation and test, including threshold tuning.

- `approaches/`  
  Model families and experiment definitions for both papers, including:
  - **Direct** multi-label DeBERTa baselines
  - Models with **lexical / topic / context** features
  - **Presence-gated** and **HO-gated** hierarchies
  - **LLM-based** baselines and QLoRA variants
  - Ensemble definitions

- `core/`  
  Shared utilities:
  - Data loading and batching
  - Training and inference loops
  - Threshold calibration and metrics
  - Logging, configuration handling

- `data-prep/`  
  Helpers to adapt the official ValueEval’24 / ValuesML release to the expected TSV format
  (e.g., checking fields, generating presence labels if needed, sanity checks).

- `requirements.txt`  
  Python dependencies.

You do **not** need to touch the internals of `approaches/` or `core/` to run the main experiments; the high-level scripts (`predict.py`, `eval.py`) expose the functionality you need via CLI arguments.

---

## Installation

We recommend Python ≥ 3.10 and a recent pip.

```bash
git clone https://github.com/VictorMYeste/human-value-detection.git
cd human-value-detection

# (optional, but recommended)
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

The repo uses standard libraries such as:
- torch
- transformers
- accelerate
- scikit-learn
- pandas, numpy
- sentencepiece
- etc. (see requirements.txt for the full list)

---

## Data

The repository does not redistribute the ValueEval’24 / ValuesML corpus.
You must obtain it yourself and accept the corresponding Data Usage Agreement.

1. Download ValueEval’24 / ValuesML
	1.	Register and download the English machine-translated splits (train / validation / test) from the official release (see the papers for the exact Zenodo link / DOI).
	2.	You should obtain at least:
- sentences.tsv
- labels-cat.tsv
for each split.

We assume the official columns are present, including:
- Text-ID, Sentence-ID
- 19 value columns

2. Directory layout

The scripts expect something like:

human-value-detection/
  data/
    train-english/
      sentences.tsv
      labels.tsv
    validation-english/
      sentences.tsv
      labels.tsv
    test-english/
      sentences.tsv
      labels.tsv

You can of course keep data elsewhere; just pass the correct paths with --train-dataset, --validation-dataset, and --test-dataset (see below). The example commands below assume:

../../data/validation-english/
../../data/test-english/

relative to the repo root (as in the original minimal README).

You can either:
- regenerate the presence and high-order categories fields yourself (presence = 1 iff any of the 19 value columns is non-zero), or
- adapt/use a small helper script in data-prep/.

---

## Quickstart

Predictions

To generate predictions for a model on the English validation split:

```bash
python3 predict.py \
  --validation-dataset ../../data/validation-english/ \
  --model-name MODEL_NAME
```

This will:
- load the specified model configuration,
- run inference on the validation sentences,
- write predictions to a standard location (typically under an internal outputs/ or similar directory, as configured in the code).

Evaluation

To evaluate a model, use:

```bash
python3 eval.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name MODEL_NAME
```

Typical behaviour:
- load validation predictions (or run them if missing),
- tune thresholds on the validation split,
- apply the tuned thresholds to test predictions,
- print and/or save evaluation metrics (e.g., macro-averaged F1 over the 19 values and presence F1).

Note:
The argument --model-name controls which configuration in approaches/ is instantiated.
Baseline is the simplest DeBERTa-based model. Other names correspond to feature-augmented, hierarchical, or ensemble variants used in the papers.

---

## Reproducing the papers

The codebase is shared; different configurations and model names correspond to the two papers.

1. Moral presence & ensembles paper

Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum (arXiv:-)

This paper focuses on:
- A sentence-level moral presence task (presence = any value yes/no).
- Direct 19-value detection vs. presence-gated hierarchies.
- Lightweight signals (LIWC-22, eMFD/MJD, topics, prior-sentence context).
- Compute-frugal ensembles of:
- Supervised DeBERTa models,
- Instruction-tuned LLMs (Gemma, Llama, Qwen, Mistral),
- QLoRA-adapted LLMs.

In this repo, the main pieces are:
- Presence-only models in approaches/ (gate training and evaluation).
- Value models with optional presence gating.
- LLM integrations and their wrappers (for zero-/few-shot and QLoRA).
- Ensemble definitions (soft-voting, hard-voting, hybrid ensembles).

The workflow is:
1.	Run predict.py for the model you want (e.g., presence gate, direct value model, presence-gated hierarchy, ensemble).
2.	Run eval.py to tune thresholds and evaluate.

Example (placeholder model names – replace with your actual ones):

```bash
# Presence-only gate
python3 predict.py \
  --validation-dataset ../../data/validation-english/ \
  --model-name Presence_LIWC22

python3 eval.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name Presence_LIWC22

# Best supervised ensemble for value detection
python3 predict.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name Deberta_Ensemble

python3 eval.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name Deberta_Ensemble
```

Check the code in approaches/ for the exact model names used in the paper tables (e.g., baseline, LIWC-22-augmented, topic-augmented, presence-gated, Gemma-related, ensembles, etc.).

2. Higher-Order values paper

Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? When Hard Gating Hurts (arXiv:-)

This paper focuses on:
- The 19 values and eight derived higher-order (HO) categories (e.g., Openness to Change, Conservation, Self-Transcendence, etc.).
- Comparing:
- Direct 19-value prediction,
- Category → Values hierarchical pipelines,
- Presence → Category → Values cascades.
- Effects of:
- Threshold calibration,
- Auxiliary features (lexica, topics, short context),
- Small supervised ensembles,
- Medium-sized instruction-tuned LLMs.

In this repo, the main ingredients for this paper are:
- Model definitions and configs in approaches/ (HO-related models and gates).
- Shared components in core/ (data loading, training, evaluation, thresholding).

Typical workflow (conceptual):
1.	Choose the HO-aware model you want to reproduce (see the comments and model registry inside approaches/).
2.	Run predict.py on the validation and/or test sets with that --model-name.
3.	Run eval.py to perform threshold tuning and evaluation.

Example (placeholder model name – update to your actual one):

```bash
python3 predict.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name HO_Cascade_Deberta

python3 eval.py \
  --validation-dataset ../../data/validation-english/ \
  --test-dataset ../../data/test-english/ \
  --model-name HO_Cascade_Deberta
```

Please consult the model registry (e.g., in approaches/__init__.py or similar) for the exact names corresponding to the HO models reported in the paper.

---

## Citation

Please cite the relevant paper(s) if you use this repository:

-TO BE PUBLISHED IN THE FOLLOWING DAYS-

@article{yeste2026human-values-single-sentence,
  title   = {Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum},
  author  = {Yeste, V{\'\i}ctor and Rosso, Paolo},
  journal = {arXiv preprint arXiv:-},
  year    = {2026}
}

@article{yeste2026schwartz-ho-values,
  title   = {Do Schwartz Higher-Order Values Help Sentence-Level Human Value Detection? When Hard Gating Hurts},
  author  = {Yeste, V{\'\i}ctor and Rosso, Paolo},
  journal = {arXiv preprint arXiv:-},
  year    = {2026}
}

---

## License

The code in this repository is released under the **Apache License 2.0**.
See the `LICENSE.txt` file for details.

Please also respect the licenses and Data Usage Agreements of any external
resources (e.g., ValueEval/ValuesML datasets, proprietary lexica such as LIWC).

---

## Contact

For questions, issues, or suggestions, feel free to open a GitHub issue or contact:

Víctor Yeste – vicyesmo [at] upv [dot] es