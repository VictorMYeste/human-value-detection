#!/usr/bin/env bash

echo "=========="
echo "===== Train gate adapter ====="
echo "=========="

python3 main.py --model google/gemma-2-9b-it --run_name qlora-hier-gemma2-gate --mode qlora --qlora_task gate --qlora_lr 2e-4 --qlora_r 16 --qlora_alpha 32 --epochs 3 --max_sentences 64 --hf_token HF_TOKEN 2>&1 | tee qlora-hier-gate-run.log

echo "=========="
echo "===== Train values adapter (on Presence==1 rows) ====="
echo "=========="

python3 main.py --model google/gemma-2-9b-it --prompt_id definition --run_name qlora-hier-gemma2-values --mode qlora --qlora_task values --qlora_lr 2e-4 --qlora_r 16 --qlora_alpha 64 --epochs 3 --max_sentences 64 --hf_token HF_TOKEN 2>&1 | tee qlora-hier-values-run.log

echo "=========="
echo "===== Hierarchical inference (gate→values) ====="
echo "=========="

python3 main.py --model google/gemma-2-9b-it --prompt_id definition --run_name qlora-hier-gemma2 --mode hier --gate_adapter output/qlora-hier-gemma2-gate/qlora_output/lora_adapters_gate --values_adapter output/qlora-hier-gemma2-values/qlora_output/lora_adapters_values --max_sentences 32 --hf_token HF_TOKEN 2>&1 | tee qlora-hier-run.log