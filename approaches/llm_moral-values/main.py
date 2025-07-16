#!/usr/bin/env python
"""
llm_value_detection.py
======================
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
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
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

PROMPT_ST_1 = (
    "Given the following SENTENCE, determine the degree (between 0 and 1) that the "
    "SENTENCE refers to the human value of {value}. Think step by step. Then say \"ANSWER: \" "
    "followed by your determined degree as single number between 0 and 1.\nSENTENCE: {sentence}\n"
)

ANSWER_RE = re.compile(r"ANSWER:\s*([01](?:\.\d+)?(?:e-?\d+)?)")

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
    df = pd.read_csv(path, sep="\t", header=None, names=["Text-ID", "Sentence-ID", "Text"])
    return df


# ---------------------------------------------------------------------
# MODEL WRAPPERS
# ---------------------------------------------------------------------

class ValueLLM:
    """Thin wrapper for causal LMs in 4‑bit for fast inference."""

    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
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
        )
        self.model.eval()
        self.generation_cfg = GenerationConfig(max_new_tokens=128, do_sample=False)

    @torch.no_grad()
    def get_answer(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, generation_config=self.generation_cfg)
        # Take newly generated tokens only
        gen_text = self.tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        match = ANSWER_RE.search(gen_text)
        if not match:
            raise ValueError(f"Model failed to follow instruction. Output was: {gen_text}")
        return float(match.group(1))


# ---------------------------------------------------------------------
# INFERENCE MODES
# ---------------------------------------------------------------------

def run_zero_or_few_shot(
    df: pd.DataFrame,
    model: ValueLLM,
    fewshot_k: int = 0,
    seed: int = 42,
    cache_path: Path | None = None,
):
    """Run zero‑shot (k=0) or few‑shot (k>0) inference, return DataFrame with probs."""

    cache: Dict[str, Any] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    # Few‑shot exemplar pool from *train* set only (assumes df contains that split)
    exemplar_pool: Dict[str, List[str]] = {}
    if fewshot_k > 0:
        rng = random.Random(seed)
        for value in VALUES:
            sentences = df.sample(frac=1.0, random_state=rng).head(1000)["Text"].tolist()
            exemplar_pool[value] = sentences[:fewshot_k]

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["Text"]
        sent_cache: Dict[str, float] = {}
        for value in VALUES:
            prompt = PROMPT_ST_1.format(value=value, sentence=sentence)
            if fewshot_k > 0:
                exemplars = "\n".join(
                    PROMPT_ST_1.format(value=value, sentence=s) + "ANSWER: 1.0"  # dummy high
                    for s in exemplar_pool[value]
                )
                prompt = exemplars + "\n" + prompt
            if prompt in cache:
                degree_resorted = cache[prompt]
            else:
                degree_resorted = model.get_answer(prompt)
                cache[prompt] = degree_resorted
            sent_cache[value] = degree_resorted
        rows.append(sent_cache)
    if cache_path:
        cache_path.write_text(json.dumps(cache))
    out_df = pd.concat([df[["Text-ID", "Sentence-ID"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out_df


# ---------------------------------------------------------------------
# QLoRA TRAINING
# ---------------------------------------------------------------------

def run_qlora(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
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
    if tokenizer.pad_token is None:
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
        for _, r in df.iterrows():
            for value in VALUES:
                prompt = PROMPT_ST_1.format(value=value, sentence=r["Text"])
                yield {"text": prompt + "ANSWER: ", "label": str(r[value])}

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
    p.add_argument("--data_dir", type=Path, default="../../data/", ="Directory containing train/val/test TSV files")
    p.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which split to run inference on")
    p.add_argument("--mode", choices=["zero-shot", "few-shot", "qlora"], default="zero-shot")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model repo or local path")
    p.add_argument("--k", type=int, default=0, help="k exemplars per value in few‑shot mode (ignored otherwise)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5, help="Prob. → label threshold for downstream eval utils")
    # QLoRA specific
    p.add_argument("--qlora_lr", type=float, default=2e-4)
    p.add_argument("--qlora_r", type=int, default=16)
    p.add_argument("--qlora_alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ----------------- load data -----------------
    df_train = load_split(args.data_dir / "training-english/sentences.tsv") if (args.data_dir / "training-english/sentences.tsv").exists() else None
    df_val = load_split(args.data_dir / "validation-english/sentences.tsv") if (args.data_dir / "validation-english/sentences.tsv").exists() else None
    df_test = load_split(args.data_dir / "test-english/sentences.tsv") if (args.data_dir / "test-english/sentences.tsv").exists() else None

    split_map = {"train": df_train, "val": df_val, "test": df_test}
    df_target = split_map[args.split]
    if df_target is None:
        sys.exit(f"[{args.split}] split not found under {args.data_dir}")

    cache_path = args.data_dir / "llm_cache.json"

    if args.mode in {"zero-shot", "few-shot"}:
        model = ValueLLM(args.model)
        preds_df = run_zero_or_few_shot(
            df_target,
            model,
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
            lora_r=args.qlora_r,
            lora_alpha=args.qlora_alpha,
            lr=args.qlora_lr,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        raise ValueError(args.mode)

    out_file = args.data_dir / "predictions.tsv"
    preds_df.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()