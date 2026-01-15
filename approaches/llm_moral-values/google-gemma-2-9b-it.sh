#!/usr/bin/env bash

# echo "=========="
# echo "=========="
# echo "=========="
# echo "===== google-gemma-2-9b-it ====="
# echo "=========="
# echo "=========="
# echo "=========="
# 
# echo "=========="
# echo "===== direct ====="
# echo "=========="
# 
# mkdir output/direct
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id direct --run_name direct --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/direct | tee output/direct/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== cot_hidden ====="
# echo "=========="
# 
# mkdir output/cot_hidden
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id cot_hidden --run_name cot_hidden --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/cot_hidden | tee output/cot_hidden/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== qa ====="
# echo "=========="
# 
# mkdir output/qa
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id qa --run_name qa --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/qa | tee output/qa/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition ====="
# echo "=========="
# 
# mkdir output/definition
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition | tee output/definition/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 1 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-1 --mode few-shot --k 1 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-1 | tee output/definition-1/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 2 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-2 --mode few-shot --k 2 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-2 | tee output/definition-2/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 4 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-4 --mode few-shot --k 4 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-4 | tee output/definition-4/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 8 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-8 --mode few-shot --k 8 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-8 | tee output/definition-8/google-gemma-2-9b-it-val-results.txt
#
# echo "=========="
# echo "===== definition - 12 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-12 --mode few-shot --k 12 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-12 | tee output/definition-12/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 16 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-16 --mode few-shot --k 16 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-16 | tee output/definition-16/google-gemma-2-9b-it-val-results.txt
# 
# echo "=========="
# echo "===== definition - 20 ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-20 --mode few-shot --k 20 --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-20 | tee output/definition-20/google-gemma-2-9b-it-val-results.txt

# echo "=========="
# echo "===== definition (test) ====="
# echo "=========="
# 
# mkdir output/definition
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name definition --hf_token HF_TOKEN
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/definition | tee output/definition/google-gemma-2-9b-it-test-results.txt
# 
# echo "=========="
# echo "===== definition - 20 (test) ====="
# echo "=========="
# 
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name definition-20 --mode few-shot --k 20 --hf_token HF_TOKEN
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/definition-20 | tee output/definition-20/google-gemma-2-9b-it-test-results.txt

# echo "=========="
# echo "===== QLoRA direct ====="
# echo "=========="
# echo "Validation:"
# python3 -u main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name qlora-direct-lr-2e-4 --mode qlora --qlora_lr 2e-4 --qlora_r 16 --qlora_alpha 32 --epochs 3 --hf_token HF_TOKEN 2>&1 | tee run.log
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/qlora-direct-lr-2e-4 | tee output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-val-results.txt
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name qlora-direct-lr-2e-4 --mode qlora --hf_token HF_TOKEN 2>&1 | tee run.log
# python3 eval.py --test-dataset ../../data/test-english --model-name google-gemma-2-9b-it --output-directory output/qlora-direct-lr-2e-4 | tee output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-test-results.txt

# echo "=========="
# echo "===== Zero-shot (SBERT gate) ====="
# echo "=========="

# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition --preds_tsv output/definition/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col Presence --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/definition | tee output/definition/google-gemma-2-9b-it-gated_Presence-val-results.txt

# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name definition --preds_tsv output/definition/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col Presence --gate_tau 0.29 --hf_token HF_TOKEN
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/definition | tee output/definition/google-gemma-2-9b-it-gated_Presence-test-results.txt

# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="

# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name definition-20 --mode few-shot --k 20 --preds_tsv output/definition-20/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col Presence --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/definition-20 | tee output/definition-20/google-gemma-2-9b-it-gated_Presence-val-results.txt

# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name definition-20 --mode few-shot --k 20 --preds_tsv output/definition-20/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col Presence --gate_tau 0.29 --hf_token HF_TOKEN
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/definition-20 | tee output/definition-20/google-gemma-2-9b-it-gated_Presence-test-results.txt

# echo "=========="
# echo "===== QLoRA (SBERT gate) ====="
# echo "=========="

# echo "Validation:"
# python main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name qlora-direct-lr-2e-4 --mode qlora --preds_tsv output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col Presence --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run.log
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/qlora-direct-lr-2e-4 | tee output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-gated_Presence-val-results.txt

# echo "Test:"
# python main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name qlora-direct-lr-2e-4 --mode qlora --preds_tsv output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col Presence --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run.log
# python3 eval.py --test-dataset ../../data/test-english --model-name google-gemma-2-9b-it-gated_Presence --output-directory output/qlora-direct-lr-2e-4 | tee output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-gated_Presence-test-results.txt