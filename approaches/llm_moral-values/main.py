"""
Unified entry‑point for sentence‑level human–value detection with large language models.

Supports three modes that you can toggle via CLI:
  • zero‑shot    – no training, prompt the base model
  • few‑shot     – k in‑context exemplars sampled from the train split
  • qlora        – parameter‑efficient fine‑tuning with QLoRA adapters

Hardware budget: ≤ 8 GB VRAM. The code therefore defaults to 4‑bit quantisation
for inference and QLoRA (+paged optimisers) for training.

Example usage
-------------
# Zero‑shot evaluation on the validation split
python main.py \
    --mode zero-shot \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_dir data/ \
    --split val
    --run_name zero-llama

# Few‑shot, 3 examples per coarse value, evaluate on val
python main.py --mode few-shot --k 3 --split val …

# QLoRA: train on train, early‑stop on val, evaluate on test
python main.py --mode qlora --train \
    --qlora_lr 2e-4 --qlora_r 16 --qlora_alpha 32 …

The script expects three TSV files inside --data_dir:
    train.tsv, val.tsv, test.tsv
with the columns:  Text-ID, Sentence-ID, Text

Outputs
-------
• predictions.tsv – probability columns ready for evaluation
• *_metrics.json   – macro‑F1, AUROC, Brier, etc.
• lora_adapters/   – (qlora mode) PEFT checkpoint only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
import gc

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

import torch
# ---- safe defaults for older CUDA/kernels ----
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = False
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    generation,
    logging as hf_logging,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList
)
from datasets import Dataset
# tell Transformers to log only errors
hf_logging.set_verbosity_error()
# hf_logging.set_verbosity_info()
from tqdm.auto import tqdm

try:
    # Optional PEFT imports – only required in qlora mode
    from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
    from trl import SFTTrainer, SFTConfig
except ImportError:
    # Delay failure until user actually requests qlora
    pass
try:
    from peft import AutoPeftModelForCausalLM
except Exception:
    AutoPeftModelForCausalLM = None

from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# CONSTANTS & PROMPTS
# ---------------------------------------------------------------------

VALUES = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    "Achievement", "Power: dominance", "Power: resources", "Face",
    "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
    "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature",
    "Universalism: tolerance",
]
VALUES_STR = ", ".join(VALUES)
VALUES_JSON = json.dumps(VALUES, ensure_ascii=False)

# One-line definition for each of the 19 refined values in Table 1 of the paper “Refining the Theory of Basic Individual Values” (Journal of Personality & Social Psychology, 2012).
VALUE_DEFINITIONS = {
    "Self-direction: thought":      "Freedom to cultivate one’s own ideas and abilities",
    "Self-direction: action":       "Freedom to determine one’s own actions",
    "Stimulation":                  "Excitement, novelty, and change",
    "Hedonism":                     "Pleasure and sensuous gratification",
    "Achievement":                  "Success according to social standards",
    "Power: dominance":             "Power through exercising control over people",
    "Power: resources":             "Power through control of material and social resources",
    "Face":                         "Maintaining one’s public image and avoiding humiliation",
    "Security: personal":           "Safety in one’s immediate environment",
    "Security: societal":           "Safety and stability in the wider society",
    "Tradition":                    "Maintaining and preserving cultural, family, or religious traditions",
    "Conformity: rules":            "Compliance with rules, laws, and formal obligations",
    "Conformity: interpersonal":    "Avoidance of upsetting or harming other people",
    "Humility":                     "Recognising one’s insignificance in the larger scheme of things",
    "Benevolence: caring":          "Devotion to the welfare of in-group members",
    "Benevolence: dependability":   "Being a reliable and trustworthy member of the in-group",
    "Universalism: concern":        "Commitment to equality, justice, and protection for all people",
    "Universalism: nature":         "Preservation of the natural environment",
    "Universalism: tolerance":      "Acceptance and understanding of those who are different from oneself",
}

# Build one multiline definition block – used only in the “definition” prompt
DEFS_BLOCK = "\n".join(
    f"- **{k}**: {v}" for k, v in VALUE_DEFINITIONS.items()
)

# ---------------------------------------------------------------------
# PROMPT TEMPLATES  – all return *only* a JSON array of the values present
# ---------------------------------------------------------------------
PROMPTS: dict[str, str] = {

    # 0 ───────────────────────────────────────────────────────────────
    # “direct” / keyword spotting
    "direct": (
        "List **only** the basic values that are clearly expressed or implied "
        "in the SENTENCE.\n"
        f"Return them as a JSON array of strings (choose from: {VALUES_STR}). "
        "No other text.\n\n"
        "SENTENCE: {sentence}"
    ),

    # 1 ───────────────────────────────────────────────────────────────
    # hidden Chain-of-Thought (the model may reason silently)
    "cot_hidden": (
        "Think step-by-step about which of the 19 basic human values are present "
        "in the SENTENCE, **but do not reveal your reasoning**.\n"
        "Output **only** the JSON array of value names.\n\n"
        f"Allowed values: {VALUES_STR}\n\n"
        "SENTENCE: {sentence}"
    ),

    # 2 ───────────────────────────────────────────────────────────────
    # with short definitions (often helps)
    "definition": (
        "### Value definitions\n"
        f"{DEFS_BLOCK}\n\n"
        "### Task\n"
        "Identify which of the above values the SENTENCE relates to.  "
        "Return **only** a JSON array of the matching value names.\n\n"
        "SENTENCE: {sentence}"
    ),

    # 3 ───────────────────────────────────────────────────────────────
    # question–answer style
    "qa": (
        "Question: Which basic human values are present in the SENTENCE?\n"
        "Answer: Return **only** a JSON array (strings) chosen from:\n"
        f"{VALUES_STR}\n\n"
        "SENTENCE: {sentence}"
    ),

    # 4 ───────────────────────────────────────────────────────────────
    # few-shot exemplar template – *MUST* keep the three placeholders
    "one_shot": (
        "SENTENCE: {example_sentence}\n"
        "OUTPUT: {example_labels}\n"
        "–––\n"
    ),
}

SYS_PROMPT = "You are a moral-psychology assistant. Using the “refined basic values” taxonomy (Schwartz 1992; Schwartz et al. 2012), answer the user’s labeling requests exactly as instructed."

MAX_BATCH = 2

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_split(path: Path) -> pd.DataFrame:
    """
    Read the SENTENCE file and, if present in the same folder, merge the
    companion *labels-cat.tsv* so train/val splits carry the gold labels
    needed by QLoRA and evaluation.
    """
    # 1) sentences ----------------------------------------------------
    df_sent = pd.read_csv(path, sep="\t")            # header row already in file

    # 2) optional labels ---------------------------------------------
    label_path = path.parent / "labels-cat.tsv"
    if label_path.exists():                          # train / val
        df_lab = pd.read_csv(label_path, sep="\t")
        df = df_sent.merge(df_lab, on=["Text-ID", "Sentence-ID"], how="inner")
    else:                                            # test → no labels available
        df = df_sent

    return df

# ─── add once, near the top of the file ────────────────────────────────
def save_cache(cache_path: Path | None, cache: dict):
    """Write the current cache dict atomically (safe on interrupt)."""
    if cache_path is None:
        return
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))   # ① write to a temp file
    tmp.replace(cache_path)             # ② atomic move → cache_path

def load_gate_scores(fp: Path, id_cols=("Text-ID","Sentence-ID")) -> pd.DataFrame:
    if fp is None or (not fp.exists()):
        raise FileNotFoundError(f"--gate_scores file not found: {fp}")
    df = pd.read_csv(fp, sep="\t")
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"--gate_scores missing ID column: {c}")
    return df

def _ensure_01(s: pd.Series) -> pd.Series:
    # Accept probs in [0,1]. If logits/unknown range, min-max as last resort.
    if (s.min() >= 0.0) and (s.max() <= 1.0):
        return s.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi > lo:
        return ((s - lo) / (hi - lo)).clip(0,1)
    return (s * 0.0)

def apply_hard_gate(preds_df: pd.DataFrame, gate_df: pd.DataFrame, gate_col: str, tau: float,
                    values: list[str]) -> pd.DataFrame:
    """Join by (Text-ID, Sentence-ID); zero all value columns where gate < tau."""
    if gate_col not in gate_df.columns:
        raise ValueError(f"Gate column '{gate_col}' not found in --gate_scores.")
    g = gate_df[["Text-ID","Sentence-ID", gate_col]].copy()
    g[gate_col] = _ensure_01(g[gate_col])
    out = preds_df.merge(g, on=["Text-ID","Sentence-ID"], how="left", validate="one_to_one")
    if out[gate_col].isna().any():
        miss = out[out[gate_col].isna()][["Text-ID","Sentence-ID"]].head(3).to_dict("records")
        raise ValueError(f"Missing gate scores for some sentences, examples: {miss} …")
    mask = out[gate_col] < float(tau)
    out.loc[mask, values] = 0.0
    out = out.drop(columns=[gate_col])
    return out

def macro_f1_from_frames(gold_df: pd.DataFrame, pred_df: pd.DataFrame, values: list[str]) -> float:
    """Compute macro-F1 over the 19 labels (binary)."""
    y_true = gold_df[values].astype(int).values
    y_pred = pred_df[values].round().astype(int).values
    return f1_score(y_true, y_pred, average="macro")

def tune_tau_on_val(preds_df: pd.DataFrame, gold_df: pd.DataFrame, gate_df: pd.DataFrame,
                    gate_col: str, values: list[str]) -> dict:
    """Grid search τ ∈ {0.00, 0.01, …, 1.00} to maximise end-to-end macro-F1 on val."""
    best = {"tau": 0.0, "macro_f1": -1.0, "coverage": 1.0}
    s = _ensure_01(gate_df[gate_col])
    taus = np.linspace(0.0, 1.0, 101)
    # Join once for speed
    base = preds_df.merge(gate_df[["Text-ID","Sentence-ID", gate_col]], on=["Text-ID","Sentence-ID"], how="left")
    if base[gate_col].isna().any():
        raise ValueError("Gate scores missing for some validation items.")
    for t in taus:
        tmp = base.copy()
        tmp.loc[tmp[gate_col] < t, values] = 0.0
        f1 = macro_f1_from_frames(gold_df, tmp.drop(columns=[gate_col]), values)
        cov = float((tmp[gate_col] >= t).mean())
        if f1 > best["macro_f1"]:
            best = {"tau": float(t), "macro_f1": float(f1), "coverage": cov}
    return best

# ---------------------------------------------------------------------
# DEBUG LOGGING (optional)
# ---------------------------------------------------------------------
DEBUG_N: int = 0                 # number of generations to log
DEBUG_FILE: str | None = None    # path to JSONL file

def _log_sample(prompt: str, generation: str, present: set[str]):
    """Append a single debug record to DEBUG_FILE, if enabled."""
    global DEBUG_N, DEBUG_FILE
    if DEBUG_N <= 0 or not DEBUG_FILE:
        return
    rec = {
        "prompt": prompt,
        "generation": generation,
        "present": sorted(list(present)),
    }
    try:
        with open(DEBUG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        DEBUG_N -= 1
    except Exception:
        pass

# ---------------------------------------------------------------------
# MODEL WRAPPERS
# ---------------------------------------------------------------------
class ValueLLM:
    """Thin wrapper for causal LMs in 4‑bit for fast inference."""

    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str | None = None, hf_token: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure padding token exists for safe batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=hf_token,
        )
        self.model.eval()
        self.generation_cfg = GenerationConfig(
            max_new_tokens  = 200,
            do_sample       = False,
            pad_token_id    = self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # no_repeat_ngram_size=32,
            # repetition_penalty=1.05,
            early_stopping=True,
        )
    
# ---------------------------------------------------------------------
# BATCHED INFERENCE HELPER
# ---------------------------------------------------------------------
def extract_json(text: str):
    """
    Return the last JSON array or object found in the text (multi-line safe),
    or None if nothing could be parsed.
    """
    s = text.strip()

    # find last complete {...} or [...]
    def last_balanced(s, open_ch, close_ch):
        span = _last_balanced_span(s, open_ch, close_ch)
        if not span:
            return None
        a, b = span
        frag = s[a:b]
        try:
            return json.loads(frag)
        except Exception:
            return None

    obj = last_balanced(s, "{", "}") or last_balanced(s, "[", "]")
    return obj


def extract_present_values(text: str) -> set[str]:
    """
    Accept either:
      - a JSON array of value names, or
      - a JSON object {value_name: 0/1 or prob} and select keys >= 0.5/True.
    """
    obj = extract_json(text)
    if obj is None:
        return set()

    if isinstance(obj, list):
        return {str(v).strip() for v in obj if isinstance(v, str)}

    if isinstance(obj, dict):
        out = set()
        for k, v in obj.items():
            name = str(k).strip()
            if isinstance(v, (int, float)) and float(v) >= 0.5:
                out.add(name)
            elif isinstance(v, str) and v.strip().lower() in {"1", "true", "yes", "present"}:
                out.add(name)
        return out

    return set()

def _slice_after_first_output(text: str) -> str:
    i = text.find("OUTPUT:")
    return text[i+7:] if i != -1 else text

def _last_balanced_span(s: str, open_ch: str, close_ch: str) -> tuple[int, int] | None:
    """
    Return (start, end_exclusive) of the last balanced {...} or [...] span.
    Very lightweight: ignores escapes/quotes which is fine for these keys.
    """
    start = -1
    depth = 0
    last = None
    for i, ch in enumerate(s):
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                last = (start, i + 1)
    return last

def clip_to_last_json(text: str) -> str:
    """
    Keep only the last complete JSON object or array (after OUTPUT: if present).
    If none complete → return original (so downstream can still log it).
    """
    tail = _slice_after_first_output(text)
    # try object first, then array
    span = _last_balanced_span(tail, "{", "}") or _last_balanced_span(tail, "[", "]")
    if not span:
        return text
    a, b = span
    clipped = tail[a:b]
    return clipped

def infer_batch(prompts: List[str], model: ValueLLM) -> List[Dict[str, float]]:
    """
    Tokenize a list of prompts, run one .generate() and return list of floats.
    """
    chat_prompts = []
    for p in prompts:
        msg = [
            {"role": "system",
             "content": SYS_PROMPT},
            {"role": "user",
             "content": p}  # p already contains the formatted sentence
        ]
        try:
            chat_prompts.append(
                model.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True)
            )
        except Exception:
            # Fallback for tokenizers/templates that don't support a system role
            chat_prompts.append(
                model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": SYS_PROMPT + "\n\n" + p}],
                    tokenize=False, add_generation_prompt=True
                )
            )
    """
    toks = model.tokenizer(chat_prompts, padding=True, return_tensors="pt").to(model.model.device)
    # move to device and preserve correct dtypes
    for k, v in toks.items():
        if v.dtype.is_floating_point:          # (rare for tokenizer outputs)
            toks[k] = v.half().to(model.model.device)
        else:
            toks[k] = v.to(model.model.device)
    """
    toks = model.tokenizer(chat_prompts, padding=True, return_tensors="pt")

    # >>> move to the embedding device (works for sharded models too)
    try:
        emb_device = model.model.get_input_embeddings().weight.device
    except Exception:
        emb_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for k, v in toks.items():
        toks[k] = v.to(emb_device, non_blocking=True)

    # optional: some tokenizers add token_type_ids; remove if present
    toks.pop("token_type_ids", None)
    # choose per-model policy (ValueLLM defaults to fp16 autocast; PEFT wrapper disables it)
    use_fp16_ac = getattr(model, "use_autocast_fp16", True)

    class StopAfterFirstClosingBrace(StoppingCriteria):
        def __init__(self, tokenizer):
            self.started = False
            self.p_output = tokenizer.encode("OUTPUT:", add_special_tokens=False)
            self.p_close  = tokenizer.encode("}\n", add_special_tokens=False)  # also add "}" alone below

        def _endswith(self, seq, pat):
            L = len(pat)
            return L and len(seq) >= L and seq[-L:] == pat

        def _contains(self, seq, pat):
            L = len(pat)
            if L == 0 or len(seq) < L: return False
            for i in range(len(seq) - L + 1):
                if seq[i:i+L] == pat:
                    return True
            return False

        def __call__(self, input_ids, scores, **kw):
            seq = input_ids[0].tolist()
            if not self.started and self._contains(seq, self.p_output):
                self.started = True
            if not self.started:
                return False
            # stop on "}\n" or bare "}"
            return self._endswith(seq, self.p_close) or self._endswith(seq, self.p_close[:-1])

    stops = StoppingCriteriaList([StopAfterFirstClosingBrace(model.tokenizer)])
    
    if torch.cuda.is_available():
        if use_fp16_ac:
            # New API to silence the FutureWarning
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                out_ids = model.model.generate(**toks, generation_config=model.generation_cfg, stopping_criteria=stops)
        else:
            # Disable autocast for fp32-compute inference
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
                out_ids = model.model.generate(**toks, generation_config=model.generation_cfg, stopping_criteria=stops)
    else:
        with torch.inference_mode():
            out_ids = model.model.generate(**toks, generation_config=model.generation_cfg, stopping_criteria=stops)

    # ─── DEBUG ───
    # print("\n\n---- RAW DECODED ----")
    # for t in model.tokenizer.batch_decode(out_ids, skip_special_tokens=False):
    #     print(repr(t));  print("---------------")
    # print("---- END ----\n\n")

    # strip off the input tokens
    gen_texts = model.tokenizer.batch_decode(out_ids[:, toks.input_ids.shape[1]:], skip_special_tokens=True)

    out = []
    for j, g in enumerate(gen_texts):
        clipped = clip_to_last_json(g)
        present = extract_present_values(clipped)
        # log the raw generation + parsed set (first DEBUG_N items only)
        try:
            _log_sample(prompts[j], clipped, present)
        except Exception:
            pass
        out.append({v: 1.0 if v in present else 0.0 for v in VALUES})
    return out

# ---------------------------------------------------------------------
# INFERENCE MODES
# ---------------------------------------------------------------------

def run_zero_or_few_shot(
    df: pd.DataFrame,
    model: ValueLLM,
    prompt_template: str,
    df_for_exemplars,
    fewshot_k: int = 0,
    seed: int = 42,
    cache_path: Path | None = None,
):
    """Run zero‑shot (k=0) or few‑shot (k>0) inference, return DataFrame with probs."""

    cache: Dict[str, Any] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    # ------------------------------------------------------------------
    # Few-shot exemplar pool: pick at most one exemplar per *dominant* value
    # ------------------------------------------------------------------
    exemplar_pool: list[tuple[str, list[str]]] = []
    if fewshot_k > 0 and df_for_exemplars is not None:
        rng = random.Random(seed)

        # build a dict {value_name -> list[rows_with_that_positive_label]}
        by_value: dict[str, list[pd.Series]] = {v: [] for v in VALUES}
        for _, row in df_for_exemplars.iterrows():          # ← use the *training* split!
            for v in VALUES:
                if row.get(v, 0) == 1.0:
                    by_value[v].append(row)

        # shuffle each list so we get random examples
        for rows in by_value.values():
            rng.shuffle(rows)

        # draw up to k distinct values
        values_shuffled = rng.sample([v for v in VALUES if by_value[v]], k=min(fewshot_k, 19))
        for v in values_shuffled:
            row = by_value[v][0]                   # take the first random row
            labels = [x for x in VALUES if row.get(x, 0) == 1.0]
            exemplar_pool.append((row["Text"], labels))

        # if the user selects k >= 20, add "none" exemplars
        if fewshot_k >= len(VALUES) + 1:
            try:
                mask = (df_for_exemplars[VALUES].sum(axis=1) == 0)
                if mask.any():
                    none_row = df_for_exemplars[mask].sample(n=1, random_state=seed).iloc[0]
                    exemplar_pool.append((none_row["Text"], []))  # OUTPUT: []
                else:
                    print("[warn] No 'none' exemplar available in df_for_exemplars; skipping.")
            except Exception:
                print("[warn] Could not add 'none' exemplar due to missing labels; skipping.")

    # prepare storage
    rows: List[Dict[str,float]] = [{} for _ in range(len(df))]
    pending_prompts: List[str] = []
    pending_meta: List[int] = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 1) build the target prompt for the current sentence
        prompt = prompt_template.format(sentence=row["Text"])

        # 2) prepend k exemplars if requested **once per sentence**
        if fewshot_k > 0:
            exemplar_prompts = []
            for ex_sent, ex_labels in exemplar_pool:
                exemplar_prompts.append(
                    PROMPTS["one_shot"].format(
                        example_sentence = ex_sent,
                        example_labels   = json.dumps(ex_labels, ensure_ascii=False)
                    )
                )
            prompt = "\n".join(exemplar_prompts) + "\n" + prompt

        if prompt in cache:
            # immediate hit
            rows[idx] = cache[prompt]
        else:
            # schedule for batch
            pending_prompts.append(prompt)
            pending_meta.append((idx))

            # once we reach BATCH, fire it off
            if len(pending_prompts) >= MAX_BATCH:
                scores = infer_batch(pending_prompts, model)
                for i, score, pr in zip(pending_meta, scores, pending_prompts):
                    cache[pr] = score
                    rows[i] = score
                pending_prompts.clear()
                pending_meta.clear()
                save_cache(cache_path, cache)

    # flush any remaining prompts
    if pending_prompts:
        scores = infer_batch(pending_prompts, model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for i, score, pr in zip(pending_meta, scores, pending_prompts):
            cache[pr] = score
            rows[i] = score
        save_cache(cache_path, cache)

    # write back cache            
    # if cache_path:
    #     cache_path.write_text(json.dumps(cache))
    
    # assemble final DataFrame
    out_df = pd.concat([df[["Text-ID", "Sentence-ID"]].reset_index(drop=True), pd.DataFrame(rows).astype(float)], axis=1)
    return out_df

# ---------------------------------------------------------------------
# QLoRA TRAINING
# ---------------------------------------------------------------------

def guess_lora_targets(model):
    wanted = {
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",   # LLaMA/Mistral/Gemma style
        "wq","wk","wv","wo","w1","w2","w3"  # some Qwen/GPTQ variants
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in wanted:
                found.add(suffix)
    # Prefer canonical names if present; otherwise fall back to w*
    canonical = [m for m in ["q_proj","k_proj","v_proj","o_proj",
                             "gate_proj","up_proj","down_proj"] if m in found]
    if canonical:
        return canonical
    fallback = [m for m in ["wq","wk","wv","wo","w1","w2","w3"] if m in found]
    return fallback or ["q_proj","v_proj"]  # last-ditch default

# add near imports
try:
    from peft.tuners.lora.layer import LoraLayer   # PEFT ≥0.7
except Exception:
    try:
        from peft.tuners.lora import LoraLayer     # older PEFT
    except Exception:
        LoraLayer = None

def _cast_lora_to_dtype(model: torch.nn.Module, dtype: torch.dtype):
    if LoraLayer is None:
        return
    for m in model.modules():
        if isinstance(m, LoraLayer):
            # cast LoRA A/B weights and any cached buffers
            for name, p in m.named_parameters(recurse=False):
                if p.dtype != dtype:
                    p.data = p.data.to(dtype=dtype)
            for name, b in m.named_buffers(recurse=False):
                if isinstance(b, torch.Tensor) and b.dtype.is_floating_point and b.dtype != dtype:
                    m.register_buffer(name, b.to(dtype=dtype), persistent=False)

try:
    import safetensors.torch as st
except Exception:
    st = None

def _load_sd(fp: Path):
    if fp.suffix == ".safetensors":
        if st is None:
            raise RuntimeError("Install safetensors")
        return st.load_file(str(fp))
    return torch.load(fp, map_location="cpu")

def adapter_checkpoint_is_clean(adapters_dir: Path) -> bool:
    ck = adapters_dir / "adapter_model.safetensors"
    if not ck.exists():
        ck = adapters_dir / "adapter_model.bin"
        if not ck.exists():
            return False
    try:
        sd = _load_sd(ck)
    except Exception:
        return False
    keys = list(sd.keys())
    # Must look like LoRA (contain lora_A/B) and must NOT contain base 4-bit weights
    has_lora = any((".lora_A." in k) or (".lora_B." in k) for k in keys)
    bad_base = any(k.endswith("base_layer.weight") for k in keys)
    return has_lora and not bad_base

def adapters_present(adapters_dir: Path) -> bool:
    if not adapters_dir.exists():
        return False
    cfg_path = adapters_dir / "adapter_config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return False
    if str(cfg.get("peft_type", "")).lower() != "lora":
        return False
    if not any((adapters_dir / f).exists() for f in ["adapter_model.safetensors","adapter_model.bin"]):
        return False
    if not adapter_checkpoint_is_clean(adapters_dir):
        print(f"[warn] {adapters_dir} doesn’t look like a clean LoRA adapter. Skipping eval-only path.")
        return False
    return True

def adapter_config(adapters_dir: Path) -> dict:
    try:
        return json.loads((adapters_dir / "adapter_config.json").read_text())
    except Exception:
        return {}
    
def _patch_bnb_loader_for_lora():
    try:
        import bitsandbytes as bnb, torch
        Linear4bit = bnb.nn.modules.Linear4bit
        if getattr(Linear4bit, "_hvd_patched", False):
            return
        _orig = Linear4bit._load_from_state_dict
        def _safe(self, state_dict, prefix, *args, **kwargs):
            # If there is no 4-bit base weight in the incoming state_dict,
            # fall back to the parent behavior (ignore missing keys).
            key = f"{prefix}base_layer.weight"
            if key not in state_dict:
                return torch.nn.Module._load_from_state_dict(self, state_dict, prefix, *args, **kwargs)
            return _orig(self, state_dict, prefix, *args, **kwargs)
        Linear4bit._load_from_state_dict = _safe
        Linear4bit._hvd_patched = True
        print("[patch] Patched bitsandbytes Linear4bit loader for LoRA-only checkpoints")
    except Exception as e:
        print(f"[patch] Couldn’t patch bitsandbytes: {e}")

def load_peft_for_inference(adapters_dir: Path, hf_token: str | None = None):
    _patch_bnb_loader_for_lora()
    if AutoPeftModelForCausalLM is None:
        raise RuntimeError("peft.AutoPeftModelForCausalLM not available; please update `peft`.")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )

    # Load the exact base + adapters stack recorded in adapter_config.json
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapters_dir),
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        # attn_implementation="eager",
        low_cpu_mem_usage=False,
        token=hf_token,
    )
    peft_model.eval()
    peft_model.config.use_cache = True

    # Prefer the *base* tokenizer if recorded in the adapter config; else fall back to the local copy.
    cfg = adapter_config(Path(adapters_dir))
    base = (cfg or {}).get("base_model_name_or_path")
    tok_src = base or str(adapters_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class _InferWrap: pass
    inf = _InferWrap()
    inf.model = peft_model
    inf.tokenizer = tokenizer
    inf.generation_cfg = GenerationConfig(
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # no_repeat_ngram_size=32,
        # repetition_penalty=1.05,
    )
    inf.use_autocast_fp16 = False
    return inf

def run_qlora(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    prompt_template: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    epochs: int = 3,
    seed: int = 42,
    output_dir: Path = Path("qlora_output"),
    eval_cache_path: Path | None = None,
    hf_token: str | None = None,
):
    set_seed(seed)

    adapters_dir = output_dir / "lora_adapters"

    # If adapters already exist → evaluate only, no training ──
    if adapters_present(adapters_dir):
        cfg = adapter_config(adapters_dir)
        if cfg:
            # optional: warn if CLI hyperparams differ from saved adapters
            if "r" in cfg and int(cfg["r"]) != lora_r:
                print(f"[warn] Using existing adapters r={cfg['r']} != CLI r={lora_r}")
            if "lora_alpha" in cfg and int(cfg["lora_alpha"]) != lora_alpha:
                print(f"[warn] Using existing adapters alpha={cfg['lora_alpha']} != CLI alpha={lora_alpha}")
        print(f"[qlora] Found trained adapters at {adapters_dir}. Skipping training; running evaluation only.")
        inf = load_peft_for_inference(adapters_dir, hf_token=hf_token)
        val_preds = run_zero_or_few_shot(
            df=val_df,
            model=inf,
            prompt_template=prompt_template,
            df_for_exemplars=None,
            fewshot_k=0,
            seed=seed,
            cache_path=eval_cache_path,
        )
        torch.cuda.empty_cache()
        return val_preds

    # ── SLOW PATH: no adapters saved → train (resumes if checkpoints exist) ──
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 4-bit weights, **fp16 compute** (consistent with inference) ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        # device_map={"": 0}, # force everything on GPU0
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # attn_implementation="eager",
        low_cpu_mem_usage=False,
        token=hf_token
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    # tiny warm call
    with torch.inference_mode():
        _ = base_model(**tokenizer("hello", return_tensors="pt").to(base_model.device))

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    try:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            lora_dropout=0.05,
            lora_dtype=torch.float16,          # match compute dtype
        )
    except TypeError:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            lora_dropout=0.05,
        )

    def make_examples(df: pd.DataFrame):
        for _, r in tqdm(df.iterrows(), total=len(df)):
            gold = {v: float(r[v]) for v in VALUES}
            yield {
                "text": (
                    prompt_template.format(
                        sentence=r["Text"],
                        example_sentence="",
                        example_labels=""
                    ).rstrip()
                    + "\nOUTPUT: "
                    + json.dumps(gold, ensure_ascii=False)
                ),
                "label": json.dumps(gold, ensure_ascii=False),
            }

    train_ds = Dataset.from_list(list(make_examples(train_df)))
    val_ds   = Dataset.from_list(list(make_examples(val_df)))

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        packing=False,
        dataset_text_field="text",
        max_length=512,
        bf16=False,
        fp16=False,                 # we keep fp32 loss; model compute is fp16
        optim="adamw_torch",
    )

    class FP32LossSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_logits.numel() == 0 or shift_labels.numel() == 0:
                loss = logits.sum() * 0.0
                return (loss, outputs) if return_outputs else loss
            B, Tm1, V = shift_logits.shape
            flat_logits = shift_logits.view(B * Tm1, V)
            flat_labels = shift_labels.view(B * Tm1).to(device=flat_logits.device, dtype=torch.long)
            valid = flat_labels != -100
            if valid.any():
                sel_logits = flat_logits[valid]
                sel_labels = flat_labels[valid]
                loss = F.nll_loss(F.log_softmax(sel_logits, dim=-1), sel_labels, reduction="mean")
            else:
                loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

    trainer = FP32LossSFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        args=sft_config,
    )

    _cast_lora_to_dtype(base_model, torch.float16)
    trainer.train(resume_from_checkpoint=True)

        # ---- save adapters only (optional; for later reuse) ----
    adapters_dir = output_dir / "lora_adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(trainer.model, PeftModel)
    trainer.model.save_pretrained(adapters_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapters_dir)

    # ---- use the already loaded, LoRA-wrapped model for inference ----
    peft_model = trainer.model          # already quantized + adapters injected
    peft_model.eval()
    peft_model.config.use_cache = True
    try:
        peft_model.gradient_checkpointing_disable()
    except Exception:
        pass
    _cast_lora_to_dtype(peft_model, torch.float16)  # keep everything in fp16

    # free trainer scaffolding but keep the model in memory
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    # ---- inference via your existing path ----
    class _InferWrap: pass
    inf_wrapper = _InferWrap()
    inf_wrapper.model = peft_model
    inf_wrapper.tokenizer = tokenizer
    inf_wrapper.generation_cfg = GenerationConfig(
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # no_repeat_ngram_size=32,
        # repetition_penalty=1.05,
    )
    # inf_wrapper.use_autocast_fp16 = True   # make generate run under fp16 autocast
    inf_wrapper.use_autocast_fp16 = False   # disable autocast for PEFT eval
    peft_model.config.use_cache = False
    inf_wrapper.generation_cfg.use_cache = False

    val_preds = run_zero_or_few_shot(
        df=val_df,
        model=inf_wrapper,
        prompt_template=prompt_template,
        df_for_exemplars=None,
        fewshot_k=0,
        seed=seed,
        cache_path=eval_cache_path,
    )

    torch.cuda.empty_cache()
    return val_preds

# ---------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human value detection with LLMs (direct approach)")
    p.add_argument("--data_dir", type=Path, default="../../data/", help="Directory containing train/val/test TSV files")
    p.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which split to run inference on")
    p.add_argument("--mode", choices=["zero-shot", "few-shot", "qlora"], default="zero-shot")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model repo or local path")
    p.add_argument("--prompt_id", default="direct", help="Choose the prompt to use")
    p.add_argument("--k", type=int, default=0, help="k exemplars per value in few‑shot mode (ignored otherwise)")
    p.add_argument("--seed", type=int, default=42)
    # QLoRA specific
    p.add_argument("--qlora_lr", type=float, default=2e-4)
    p.add_argument("--qlora_r", type=int, default=16)
    p.add_argument("--qlora_alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--run_name", default="", help="Sub-folder inside output to store predictions, cache, adapters")
    p.add_argument("--max_sentences", type=int, default=0, help="If >0, process only the first N sentences of the chosen split")
    p.add_argument("--hf_token", default=None, help="Hugging Face token for gated models")
    p.add_argument("--log_raw", type=int, default=0, help="Log the first N generations (prompt + raw + parsed) to a JSONL file")
    p.add_argument("--log_file", type=str, default="", help="Optional path for the JSONL (defaults to output/<run_name>/debug_samples.jsonl)")
    # Retrofit hierarchy (hard-mask)
    p.add_argument("--gate_mode", choices=["none", "sbert-hard"], default="none", help="Retrofit hierarchy: SBERT presence gate with hard mask.")
    p.add_argument("--gate_scores", type=Path, default=None, help="TSV with SBERT scores. Must have Text-ID, Sentence-ID and --gate_col.")
    p.add_argument("--gate_col", type=str, default="Presence", help="Column in --gate_scores with the gate score (0..1).")
    p.add_argument("--gate_tau", type=float, default=None, help="Optional manual τ. If not set: auto-tune on val, load on test.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ----------------- load data -----------------
    train_file = args.data_dir / "training-english/sentences.tsv"
    val_file   = args.data_dir / "validation-english/sentences.tsv"
    test_file  = args.data_dir / "test-english/sentences.tsv"
    df_train = load_split(train_file) if train_file.exists() else None
    df_val   = load_split(val_file)   if val_file.exists()   else None
    df_test  = load_split(test_file)  if test_file.exists()  else None

    split_map = {"train": df_train, "val": df_val, "test": df_test}
    df_target = split_map[args.split]

    output_dir = Path("output") / args.run_name
    # ------------- choose (and create) output location -------------
    out_root = output_dir.resolve() if args.run_name else "output"
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- debug logging setup ----
    global DEBUG_N, DEBUG_FILE
    DEBUG_N = getattr(args, "log_raw", 0) or 0
    log_file_cli = getattr(args, "log_file", "") or ""
    if DEBUG_N > 0:
        DEBUG_FILE = str((out_root / (log_file_cli if log_file_cli else "debug_samples.jsonl")))
        # start fresh
        try:
            Path(DEBUG_FILE).write_text("", encoding="utf-8")
        except Exception:
            pass
        print(f"[debug] logging first {DEBUG_N} generations to {DEBUG_FILE}")


    # Sanitize model name for file safety
    safe_model = re.sub(r"[^\w\-]+", "-", args.model)

    active_split = args.split
    cache_filename = f"{safe_model}-{active_split}-cache.json"
    cache_path = out_root / cache_filename

    prompt_template = PROMPTS[args.prompt_id]

    if args.mode in {"zero-shot", "few-shot"}:
        if df_target is None:
            sys.exit(f"[{args.split}] split not found under {args.data_dir}")

        if args.max_sentences > 0:
            df_target = df_target.head(args.max_sentences).reset_index(drop=True)
            
        model = ValueLLM(args.model, hf_token=args.hf_token)
        preds_df = run_zero_or_few_shot(
            df_target,
            model,
            prompt_template,
            df_for_exemplars=df_train,
            fewshot_k=args.k if args.mode == "few-shot" else 0,
            seed=args.seed,
            cache_path=cache_path,
        )
    elif args.mode == "qlora":
        # ---- Eval-only path if adapters already exist ----
        adapters_dir = (out_root / "qlora_output" / "lora_adapters")
        if adapters_present(adapters_dir) and args.split in {"val", "test"}:
            if df_target is None:
                sys.exit(f"[{args.split}] split not found under {args.data_dir}")
            if args.max_sentences > 0:
                df_target = df_target.head(args.max_sentences).reset_index(drop=True)

            # load adapters and run on the requested split (incl. test)
            values_model = load_peft_for_inference(adapters_dir=adapters_dir, hf_token=args.hf_token)
            preds_df = run_zero_or_few_shot(
                df=df_target,
                model=values_model,
                prompt_template=prompt_template,
                df_for_exemplars=None,
                fewshot_k=0,
                seed=args.seed,
                cache_path=cache_path,   # cache filename already uses args.split
            )
            # keep active_split = args.split so the output filename is “…-test.tsv”
        else:
            if df_train is None or df_val is None:
                sys.exit("QLoRA mode requires train and val splits available.")
            if args.max_sentences > 0:
                df_train = df_train.head(args.max_sentences).reset_index(drop=True)
                df_val   = df_val.head(args.max_sentences).reset_index(drop=True)

            active_split = "val"    # we evaluate on val after training

            qlora_cache_filename = (
                f"{safe_model}-qlora-{args.run_name or 'run'}-"
                f"{args.prompt_id}-r{args.qlora_r}-a{args.qlora_alpha}-"
                f"lr{args.qlora_lr}-ep{args.epochs}-seed{args.seed}-"
                f"{active_split}-cache.json"
            )
            eval_cache_path = out_root / qlora_cache_filename

            preds_df = run_qlora(
                train_df=df_train,
                val_df=df_val,
                model_name=args.model,
                prompt_template=prompt_template,
                lora_r=args.qlora_r,
                lora_alpha=args.qlora_alpha,
                lr=args.qlora_lr,
                epochs=args.epochs,
                seed=args.seed,
                output_dir=out_root / "qlora_output",
                eval_cache_path=eval_cache_path,
                hf_token=args.hf_token,
            )
    else:
        raise ValueError(args.mode)

    # after run_zero_or_few_shot / run_qlora returns preds_df
    preds_df.iloc[:, 2:] = preds_df.iloc[:, 2:].round(3)

    # ----------------- retrofit hierarchy (hard-mask) -----------------
    gate_cfg_path = Path(out_root) / f"gate_{args.gate_col}_config.json"

    if args.gate_mode == "sbert-hard":
        if args.split == "val":
            if df_val is None:
                sys.exit("Gate requires a validation split with gold labels.")
            if args.gate_scores is None:
                sys.exit("--gate_scores TSV is required when --gate_mode sbert-hard.")
            gate_df = load_gate_scores(args.gate_scores)

            # make sure gold labels exist for F1 on val
            if not all(v in df_val.columns for v in VALUES):
                sys.exit("[val] gold label columns missing; cannot tune τ.")
            # Align gold rows to preds_df rows (defensive)
            gold = preds_df.merge(df_val[["Text-ID","Sentence-ID"] + VALUES], on=["Text-ID","Sentence-ID"], how="left")
            if gold[VALUES].isna().any().any():
                sys.exit("Gold labels missing for some val items after merge.")

            # τ: use user-provided or auto-tune
            if args.gate_tau is not None:
                best = {"tau": float(args.gate_tau)}
                gated = apply_hard_gate(preds_df, gate_df, args.gate_col, best["tau"], VALUES)
                best["macro_f1"] = macro_f1_from_frames(gold, gated, VALUES)
                best["coverage"] = float(
                    load_gate_scores(args.gate_scores)[args.gate_col].pipe(_ensure_01).ge(best["tau"]).mean()
                )
            else:
                best = tune_tau_on_val(preds_df, gold, gate_df, args.gate_col, VALUES)
                gated = apply_hard_gate(preds_df, gate_df, args.gate_col, best["tau"], VALUES)

            # Save both raw and gated val predictions
            gated_filename = f"{safe_model}-{active_split}-gated_{args.gate_col}.tsv"
            (out_root / "gate_metrics").mkdir(exist_ok=True, parents=True)
            (out_root / gated_filename).write_text(gated.to_csv(sep="\t", index=False))
            # Save config for reuse on test
            cfg = {
                "gate_mode": args.gate_mode,
                "gate_col": args.gate_col,
                "tau": best["tau"],
                "val_macro_f1": best.get("macro_f1", None),
                "val_coverage": best.get("coverage", None),
            }
            (out_root / "gate_metrics" / "README.txt").write_text(
                "Gate results live here. τ chosen on validation to maximise end-to-end macro-F1."
            )
            gate_cfg_path.write_text(json.dumps(cfg, indent=2))
            print(f"[gate] val: τ={best['tau']:.2f}, macro-F1={best['macro_f1']:.5f}, coverage={best['coverage']:.3f}")
            # Replace preds_df so the normal writer below emits the gated file too (optional)
            preds_df = gated

        elif args.split == "test":
            # Load τ picked on val
            if not gate_cfg_path.exists():
                sys.exit(f"Missing {gate_cfg_path}. Run val with --gate_mode first, or provide --gate_tau.")
            cfg = json.loads(gate_cfg_path.read_text())
            tau = float(args.gate_tau) if args.gate_tau is not None else float(cfg["tau"])
            if args.gate_scores is None:
                sys.exit("--gate_scores TSV is required when --gate_mode sbert-hard.")
            gate_df = load_gate_scores(args.gate_scores)
            preds_df = apply_hard_gate(preds_df, gate_df, args.gate_col, tau, VALUES)
            print(f"[gate] test: applied hard mask with τ={tau:.2f} (gate={args.gate_col}).")
        else:
            print("[gate] Skipping gate in train split.")

    # Compose new filename
    filename = f"{safe_model}-{active_split}.tsv"

    # Write predictions
    out_file = out_root / filename
    preds_df.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {out_file}")

    try:
        col_sums = preds_df.iloc[:, 2:].sum(0)
        nonzero = {k: int(v) for k, v in col_sums[col_sums > 0].to_dict().items()}
        print(f"[summary] non-zero counts per value (this run): {nonzero}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
