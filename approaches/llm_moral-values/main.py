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
import re, ast

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    generation,
    logging as hf_logging,
)
# tell Transformers to log only errors
hf_logging.set_verbosity_error()
# hf_logging.set_verbosity_info()
from tqdm.auto import tqdm

try:
    # Optional PEFT imports – only required in qlora mode
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError:
    # Delay failure until user actually requests qlora
    pass

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

MAX_BATCH = 4

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
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=hf_token,
        )
        self.model.eval()
        self.generation_cfg = GenerationConfig(
            max_new_tokens  = 128,
            do_sample       = False,
            pad_token_id    = self.tokenizer.eos_token_id
        )

        # ---------- one-time CUDA / Triton warm-up --------------
        with torch.inference_mode():
            dummy = "Warm-up"
            toks  = self.tokenizer(dummy, return_tensors="pt").to(self.model.device)
            _ = self.model.generate(**toks, generation_config=GenerationConfig(max_new_tokens=1,do_sample=False))
            torch.cuda.synchronize() # make sure kernels finish
    
# ---------------------------------------------------------------------
# BATCHED INFERENCE HELPER
# ---------------------------------------------------------------------
def extract_json(text: str) -> dict[str, float] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return None

def extract_present_values(text: str) -> set[str]:
    """
    Pull the JSON array (last {...] or [...] block) from the model output
    and return a *set* of value names.
    """
    obj = extract_json(text)            # re-use the generic helper above
    if obj is None:                     # maybe the model emitted a plain list
        try:
            obj = json.loads(text[text.rfind("[") : text.rfind("]")+1])
        except Exception:
            return set()
    # normalise / defensive strip
    return {str(v).strip() for v in obj if isinstance(v, str)}


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
    toks = model.tokenizer(chat_prompts, padding=True, return_tensors="pt").to(model.model.device)
    out_ids = model.model.generate(**toks,generation_config=model.generation_cfg)
    # ─── DEBUG ───
    # print("\n\n---- RAW DECODED ----")
    # for t in model.tokenizer.batch_decode(out_ids, skip_special_tokens=False):
    #     print(repr(t));  print("---------------")
    # print("---- END ----\n\n")

    # strip off the input tokens
    gen_texts = model.tokenizer.batch_decode(out_ids[:, toks.input_ids.shape[1]:], skip_special_tokens=True)

    out = []
    for g in gen_texts:
        present = extract_present_values(g)
        out.append(
            {v: 1.0 if v in present else 0.0 for v in VALUES}
        )
    return out

# ---------------------------------------------------------------------
# INFERENCE MODES
# ---------------------------------------------------------------------

def run_zero_or_few_shot(
    df: pd.DataFrame,
    model: ValueLLM,
    prompt_template: str,
    df_for_exemplars,
    exemplar_dict: Dict[str, tuple] | None = None,
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
    pending_meta: List[tuple[int,str]] = []

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
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                for i, score, pr in zip(pending_meta, scores, pending_prompts):
                    cache[pr] = score
                    rows[i] = score
                pending_prompts.clear()
                pending_meta.clear()
                save_cache(cache_path, cache)

    # flush any remaining prompts
    if pending_prompts:
        scores = infer_batch(pending_prompts, model)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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

def run_qlora(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    prompt_template: str,
    exemplar_dict: Dict[str, tuple] | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    epochs: int = 3,
    seed: int = 42,
    output_dir: Path = Path("qlora_output"),
):
    """Fine‑tune with QLoRA and return val predictions as DataFrame."""

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, lora_cfg)

    # Build regression‑style training samples
    def make_examples(df: pd.DataFrame):
        for _, r in tqdm(df.iterrows(), total=len(df)):
            gold = {v: r[v] for v in VALUES}           # 0/1 floats from label cols
            yield {
                "text": prompt_template.format(
                    sentence=r["Text"],
                    example_sentence="",
                    example_labels=""
                ) + json.dumps(gold, ensure_ascii=False),
                "label": json.dumps(gold, ensure_ascii=False)
            }

    train_examples = list(make_examples(train_df))
    val_examples = list(make_examples(val_df))

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        dataset_text_field="text",
        max_seq_length=512,
        peft_config=lora_cfg,
        args=dict(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=50,
            evaluation_strategy="epoch",
            output_dir=str(output_dir),
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
        ),
    )
    trainer.train()

    # Save only adapters for reproduction
    model.save_pretrained(output_dir / "lora_adapters")

    # ---- inference on val
    peft_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = get_peft_model(peft_model, lora_cfg)
    peft_model.load_adapter(str(output_dir / "lora_adapters"))
    inf_wrapper = ValueLLM(model_name)  # uses 4‑bit wrapper above
    inf_wrapper.model = peft_model  # swap underlying model

    val_preds = run_zero_or_few_shot(val_df, inf_wrapper, fewshot_k=0)
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
    if df_target is None:
        sys.exit(f"[{args.split}] split not found under {args.data_dir}")

    # optional quick slice
    if args.max_sentences > 0:
        df_target = df_target.head(args.max_sentences).reset_index(drop=True)

    output_dir = Path("output") / args.run_name
    # ------------- choose (and create) output location -------------
    out_root = output_dir.resolve() if args.run_name else "output"
    out_root.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for file safety
    safe_model = re.sub(r"[^\w\-]+", "-", args.model)

    cache_filename = f"{safe_model}-{args.split}-cache.json"
    cache_path = out_root / cache_filename

    prompt_template = PROMPTS[args.prompt_id]

    # -----------------------------------------------------------------
    # build ONE exemplar per value from the training split (if available)
    # -----------------------------------------------------------------
    exemplar_by_value = {}
    if "{example_sentence}" in PROMPTS[args.prompt_id] and df_train is not None:
        for v in VALUES:
            # pick the first positive example; if none exist, take any negative
            pos = df_train[df_train[v] == 1.0]
            row = pos.iloc[0] if len(pos) else df_train.iloc[0]
            exemplar_by_value[v] = (row["Text"], float(row[v]))

    if args.mode in {"zero-shot", "few-shot"}:
        model = ValueLLM(args.model, hf_token=args.hf_token)
        preds_df = run_zero_or_few_shot(
            df_target,
            model,
            prompt_template,
            df_for_exemplars=df_train,
            exemplar_dict=exemplar_by_value,
            fewshot_k=args.k if args.mode == "few-shot" else 0,
            seed=args.seed,
            cache_path=cache_path,
        )
    elif args.mode == "qlora":
        if df_train is None or df_val is None:
            sys.exit("QLoRA mode requires train and val splits available.")
        preds_df = run_qlora(
            train_df=df_train,
            val_df=df_val,
            model_name=args.model,
            prompt_template=prompt_template,
            exemplar_dict=exemplar_by_value,
            lora_r=args.qlora_r,
            lora_alpha=args.qlora_alpha,
            lr=args.qlora_lr,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=out_root / "qlora_output",
        )
    else:
        raise ValueError(args.mode)

    # after run_zero_or_few_shot / run_qlora returns preds_df
    preds_df.iloc[:, 2:] = preds_df.iloc[:, 2:].round(3)

    # Compose new filename
    filename = f"{safe_model}-{args.split}.tsv"

    # Write predictions
    out_file = out_root / filename
    preds_df.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
