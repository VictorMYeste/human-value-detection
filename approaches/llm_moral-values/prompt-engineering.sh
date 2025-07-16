#!/usr/bin/env bash
export HF_HOME="$HOME/.cache/huggingface"
for p in base concise philosopher_cot definition json_out one_shot; do
  python main.py \
      --mode zero-shot \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --prompt_id $p \
      --split val \
      --run_name zs_llama8b_$p
done