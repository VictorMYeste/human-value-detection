#!/usr/bin/env bash

# 1) Change values and value definitions at main.py.
# 2) Change MODEL_GROUP at eval.py.
# 3) Execute.

echo "=========="
echo "===== Zero-shot (adapted prompt) ====="
echo "=========="

mkdir output/openness_definition

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition --hf_token HF_TOKEN 2>&1 | tee run_zero-shot-adapted-val.txt 
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition | tee output/openness_definition/zero-shot-adapted-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition --hf_token HF_TOKEN 2>&1 | tee run_zero-shot-adapted-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition | tee output/openness_definition/zero-shot-adapted-test-results.txt

echo "=========="
echo "===== Few-shot (adapted prompt) ====="
echo "=========="

mkdir output/openness_definition-5

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-adapted-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-adapted-test-results.txt

echo "=========="
echo "===== Zero-shot (SBERT gate) ====="
echo "=========="

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition --preds_tsv output/openness_definition/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --hf_token HF_TOKEN 2>&1 | tee run_zero-shot-gated-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition | tee output/openness_definition/zero-shot-gated-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition --preds_tsv output/openness_definition/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_zero-shot-gated-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition | tee output/openness_definition/zero-shot-gated-test-results.txt

echo "=========="
echo "===== Few-shot (SBERT gate) ====="
echo "=========="

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --preds_tsv output/openness_definition-5/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-gated-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --preds_tsv output/openness_definition-5/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-gated-test-results.txt

echo "=========="
echo "===== QLoRA (SBERT gate) ====="
echo "=========="

mkdir output/openness_qlora

echo "Validation:"
python main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_qlora --mode qlora --preds_tsv output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-val.tsv  --gate_mode sbert-hard --gate_col "Openness to Change" --hf_token HF_TOKEN 2>&1 | tee run_qlora-gated-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_qlora | tee output/openness_qlora/qlora-gated-val-results.txt

echo "Test:"
python main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_qlora --mode qlora --preds_tsv output/qlora-direct-lr-2e-4/google-gemma-2-9b-it-test.tsv  --gate_mode sbert-hard --gate_col "Openness to Change" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_qlora-gated-test.txt
python3 eval.py --test-dataset ../../data/test-english --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_qlora | tee output/openness_qlora/qlora-gated-test-results.txt