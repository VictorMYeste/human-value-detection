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

PROMPTS = {
    "base": (
        "Given the following SENTENCE, determine the degree (between 0 and 1) "
        "that the SENTENCE refers to the human value of {value}. Think step by "
        "step. Then say \"ANSWER: \" followed by your determined degree as single "
        "number between 0 and 1 using use up to 3 decimal places.\nSENTENCE: {sentence}\n"
    ),

    # 1 ───────────────────────────────────────────────────────────────
    # terse, no chain-of-thought
    "concise": (
        "Rate from 0 (not present) to 1 (clearly present) using use up to 3 decimal places how much the value "
        "\"{value}\" appears in this sentence. Respond with "
        "\"ANSWER: <number>\" and nothing else.\nSENTENCE: {sentence}\n"
    ),

    # 2 ───────────────────────────────────────────────────────────────
    # persona + CoT + emphasis on citing cues
    "philosopher_cot": (
        "You are a moral philosopher analysing text. For the value "
        "\"{value}\", think aloud about keywords, actors, and implications you "
        "see in the sentence, then conclude. Finish with "
        "\"ANSWER: <number between 0 and 1 using use up to 3 decimal places>\" on a new line.\nSENTENCE: {sentence}\n"
    ),

    # 3 ───────────────────────────────────────────────────────────────
    # includes short definition (you pass it via .format(definition=…))
    "definition": (
        "VALUE: {value} — {definition}\n"
        "Task: Score how strongly the sentence expresses this value on a 0-1 "
        "scale using use up to 3 decimal places). Think step by step, then output only "
        "\"ANSWER: <number>\".\nSENTENCE: {sentence}\n"
    ),

    # 4 ───────────────────────────────────────────────────────────────
    # JSON output to test structured parsing
    "json_out": (
        "Evaluate the presence of the human value \"{value}\" in the sentence "
        "on a scale 0–1 using use up to 3 decimal places. Think briefly, then respond with a JSON object exactly "
        "like {\"score\": 0.0}.\nSENTENCE: {sentence}\n"
    ),

    # 5 ───────────────────────────────────────────────────────────────
    # one-shot exemplar (supply {example_sentence} & {example_score})
    "one_shot": (
        "Example → \"{example_sentence}\" VALUE: {value} ⇒ ANSWER: {example_score}\n"
        "Now evaluate the next sentence and give ANSWER likewise using use up to 3 decimal places.\n"
        "SENTENCE: {sentence}\n"
    ),
}

# accept either the strict form (“ANSWER: 0.7”) or any standalone 0–1 number
ANSWER_RE = re.compile(r"(?:ANSWER:\s*)?([01](?:\.\d+)?(?:e-?\d+)?)")

MAX_BATCH = 64

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
        self.generation_cfg = GenerationConfig(max_new_tokens=8,do_sample=False)

        # ---------- one-time CUDA / Triton warm-up --------------
        with torch.inference_mode():
            dummy = "Warm-up"
            toks  = self.tokenizer(dummy, return_tensors="pt").to(self.model.device)
            _ = self.model.generate(**toks, generation_config=GenerationConfig(max_new_tokens=1,do_sample=False))
            torch.cuda.synchronize() # make sure kernels finish

    @torch.no_grad()
    def get_answer(self, prompt: str) -> float:
        for _ in range(2):                                      # 1 × normal, 1 × strict retry
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs,
                                             generation_config=self.generation_cfg)
            gen_text = self.tokenizer.decode(
                output_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            match = ANSWER_RE.search(gen_text)
            if match:
                return round(float(match.group(1)), 3)
            # first try failed → add an explicit instruction and retry
            prompt += ("\nRespond **only** with \"ANSWER: <number between 0 and 1>\""
                       " on one line.")
        # still no usable number – fall back to 0.0 instead of crashing
        return 0.0
    
# ---------------------------------------------------------------------
# BATCHED INFERENCE HELPER
# ---------------------------------------------------------------------
def infer_batch(prompts: List[str], model: ValueLLM) -> List[float]:
    """
    Tokenize a list of prompts, run one .generate() and return list of floats.
    """
    toks = model.tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.model.device)
    out_ids = model.model.generate(
        **toks,
        generation_config=model.generation_cfg,
    )
    # strip off the input tokens
    gen_texts = model.tokenizer.batch_decode(
        out_ids[:, toks.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    # extract numbers (fallback to 0.0)
    return [
        round(float(m.group(1)), 3) if (m := ANSWER_RE.search(txt)) else 0.0
        for txt in gen_texts
    ]


# ---------------------------------------------------------------------
# INFERENCE MODES
# ---------------------------------------------------------------------

def run_zero_or_few_shot(
    df: pd.DataFrame,
    model: ValueLLM,
    prompt_template: str,
    exemplar_dict: Dict[str, tuple] | None = None,
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

    # prepare storage
    rows: List[Dict[str,float]] = [{} for _ in range(len(df))]
    pending_prompts: List[str] = []
    pending_meta: List[tuple[int,str]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["Text"]
        for value in VALUES:
            # Add extra information to the prompt if needed
            extra: Dict[str, Any] = {}
            if "{definition}" in prompt_template:
                extra["definition"] = VALUE_DEFINITIONS.get(value, "")
            if "{example_sentence}" in prompt_template and exemplar_dict:
                ex_s, ex_sc = exemplar_dict[value]
                extra["example_sentence"] = ex_s
                extra["example_score"] = ex_sc
            prompt = prompt_template.format(value=value, sentence=sentence, **extra)
            if fewshot_k > 0:
                exemplars = "\n".join(
                    prompt_template.format(value=value, sentence=s) + "ANSWER: 1.0"  # dummy high
                    for s in exemplar_pool[value]
                )
                prompt = exemplars + "\n" + prompt
            if prompt in cache:
                # immediate hit
                rows[idx][value] = cache[prompt]
            else:
                # schedule for batch
                pending_prompts.append(prompt)
                pending_meta.append((idx, value))

                # once we reach BATCH, fire it off
                if len(pending_prompts) >= MAX_BATCH:
                    scores = infer_batch(pending_prompts, model)
                    for (r_i, val), score, pr in zip(pending_meta, scores, pending_prompts):
                        cache[pr] = score
                        rows[r_i][val] = score
                    pending_prompts.clear()
                    pending_meta.clear()

        # flush any remaining prompts
        if pending_prompts:
            scores = infer_batch(pending_prompts, model)
            for (r_i, val), score, pr in zip(pending_meta, scores, pending_prompts):
                cache[pr] = score
                rows[r_i][val] = score

    # write back cache            
    if cache_path:
        cache_path.write_text(json.dumps(cache))
    
    # assemble final DataFrame
    out_df = pd.concat([df[["Text-ID", "Sentence-ID"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
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
            for value in VALUES:
                # Add extra information to the prompt if needed
                extra = {}
                if "{definition}" in prompt_template:
                    extra["definition"] = VALUE_DEFINITIONS.get(value, "")
                if "{example_sentence}" in prompt_template and exemplar_dict:
                    ex_sent, ex_score = exemplar_dict.get(value, ("", 0.0))
                    extra["example_sentence"] = ex_sent
                    extra["example_score"] = ex_score
                prompt = prompt_template.format(
                    value=value,
                    sentence=r["Text"],
                    **extra,
                )
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
    p.add_argument("--data_dir", type=Path, default="../../data/", help="Directory containing train/val/test TSV files")
    p.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which split to run inference on")
    p.add_argument("--mode", choices=["zero-shot", "few-shot", "qlora"], default="zero-shot")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model repo or local path")
    p.add_argument("--prompt_id", default="base", help="Choose the prompt to use")
    p.add_argument("--k", type=int, default=0, help="k exemplars per value in few‑shot mode (ignored otherwise)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5, help="Prob. → label threshold for downstream eval utils")
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

    cache_path = out_root / "llm_cache.json"

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

    out_file = out_root / "predictions.tsv"
    preds_df.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()