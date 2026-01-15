#!/usr/bin/env bash

# 1) Change values and value definitions at main.py.
# 2) Change MODEL_GROUP at eval.py.
# 3) Execute.

# echo "=========="
# echo "===== Self-Protection ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/self-protection_definition-11
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-protection_definition-11 --mode few-shot --k 11 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/self-protection_definition-11 | tee output/self-protection_definition-11/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-protection_definition-11 --mode few-shot --k 11 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/self-protection_definition-11 | tee output/self-protection_definition-11/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-protection_definition-11 --mode few-shot --k 11 --preds_tsv output/self-protection_definition-11/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Self-Protection Anxiety-Avoidance" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance" --output-directory output/self-protection_definition-11 | tee output/self-protection_definition-11/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-protection_definition-11 --mode few-shot --k 11 --preds_tsv output/self-protection_definition-11/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Self-Protection Anxiety-Avoidance" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Self-Protection Anxiety-Avoidance" --output-directory output/self-protection_definition-11 | tee output/self-protection_definition-11/few-shot-gated-test-results.txt

# echo "=========="
# echo "===== Social Focus ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/social-focus_definition-11
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name social-focus_definition-11 --mode few-shot --k 11 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/social-focus_definition-11 | tee output/social-focus_definition-11/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name social-focus_definition-11 --mode few-shot --k 11 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/social-focus_definition-11 | tee output/social-focus_definition-11/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name social-focus_definition-11 --mode few-shot --k 11 --preds_tsv output/social-focus_definition-11/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Social Focus" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Social Focus" --output-directory output/social-focus_definition-11 | tee output/social-focus_definition-11/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name social-focus_definition-11 --mode few-shot --k 11 --preds_tsv output/social-focus_definition-11/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Social Focus" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Social Focus" --output-directory output/social-focus_definition-11 | tee output/social-focus_definition-11/few-shot-gated-test-results.txt

# echo "=========="
# echo "===== Personal Focus ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/personal-focus_definition-10
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name personal-focus_definition-10 --mode few-shot --k 10 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/personal-focus_definition-10 | tee output/personal-focus_definition-10/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name personal-focus_definition-10 --mode few-shot --k 10 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/personal-focus_definition-10 | tee output/personal-focus_definition-10/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name personal-focus_definition-10 --mode few-shot --k 10 --preds_tsv output/personal-focus_definition-10/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Personal Focus" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Personal Focus" --output-directory output/personal-focus_definition-10 | tee output/personal-focus_definition-10/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name personal-focus_definition-10 --mode few-shot --k 10 --preds_tsv output/personal-focus_definition-10/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Personal Focus" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Personal Focus" --output-directory output/personal-focus_definition-10 | tee output/personal-focus_definition-10/few-shot-gated-test-results.txt

# echo "=========="
# echo "===== Openness ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/openness_definition-5
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --preds_tsv output/openness_definition-5/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name openness_definition-5 --mode few-shot --k 5 --preds_tsv output/openness_definition-5/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Openness to Change" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Openness to Change" --output-directory output/openness_definition-5 | tee output/openness_definition-5/few-shot-gated-test-results.txt

# echo "=========="
# echo "===== Conservation ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/conservation_definition-8
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name conservation_definition-8 --mode few-shot --k 8 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/conservation_definition-8 | tee output/conservation_definition-8/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name conservation_definition-8 --mode few-shot --k 8 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/conservation_definition-8 | tee output/conservation_definition-8/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name conservation_definition-8 --mode few-shot --k 8 --preds_tsv output/conservation_definition-8/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Conservation" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Conservation" --output-directory output/conservation_definition-8 | tee output/conservation_definition-8/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name conservation_definition-8 --mode few-shot --k 8 --preds_tsv output/conservation_definition-8/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Conservation" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Conservation" --output-directory output/conservation_definition-8 | tee output/conservation_definition-8/few-shot-gated-test-results.txt

# echo "=========="
# echo "===== Self-Transcendence ====="
# echo "=========="
# 
# echo "=========="
# echo "===== Few-shot (adapted prompt) ====="
# echo "=========="
# 
# mkdir output/self-transcendence_definition-7
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-transcendence_definition-7 --mode few-shot --k 7 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/self-transcendence_definition-7 | tee output/self-transcendence_definition-7/few-shot-adapted-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-transcendence_definition-7 --mode few-shot --k 7 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/self-transcendence_definition-7 | tee output/self-transcendence_definition-7/few-shot-adapted-test-results.txt
# 
# echo "=========="
# echo "===== Few-shot (SBERT gate) ====="
# echo "=========="
# 
# echo "Validation:"
# python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-transcendence_definition-7 --mode few-shot --k 7 --preds_tsv output/self-transcendence_definition-7/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Self-Transcendence" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Self-Transcendence" --output-directory output/self-transcendence_definition-7 | tee output/self-transcendence_definition-7/few-shot-gated-val-results.txt
# 
# echo "Test:"
# python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-transcendence_definition-7 --mode few-shot --k 7 --preds_tsv output/self-transcendence_definition-7/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Self-Transcendence" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
# python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Self-Transcendence" --output-directory output/self-transcendence_definition-7 | tee output/self-transcendence_definition-7/few-shot-gated-test-results.txt

echo "=========="
echo "===== Self-Enhancement ====="
echo "=========="

echo "=========="
echo "===== Few-shot (adapted prompt) ====="
echo "=========="

mkdir output/self-enhancement_definition-6

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-enhancement_definition-6 --mode few-shot --k 6 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name google-gemma-2-9b-it --output-directory output/self-enhancement_definition-6 | tee output/self-enhancement_definition-6/few-shot-adapted-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-enhancement_definition-6 --mode few-shot --k 6 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-adapted-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name google-gemma-2-9b-it --output-directory output/self-enhancement_definition-6 | tee output/self-enhancement_definition-6/few-shot-adapted-test-results.txt

echo "=========="
echo "===== Few-shot (SBERT gate) ====="
echo "=========="

echo "Validation:"
python3 main.py --model google/gemma-2-9b-it --split val --prompt_id definition --run_name self-enhancement_definition-6 --mode few-shot --k 6 --preds_tsv output/self-enhancement_definition-6/google-gemma-2-9b-it-val.tsv --gate_mode sbert-hard --gate_col "Self-Enhancement" --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-val.txt
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name "google-gemma-2-9b-it-gated_Self-Enhancement" --output-directory output/self-enhancement_definition-6 | tee output/self-enhancement_definition-6/few-shot-gated-val-results.txt

echo "Test:"
python3 main.py --model google/gemma-2-9b-it --split test --prompt_id definition --run_name self-enhancement_definition-6 --mode few-shot --k 6 --preds_tsv output/self-enhancement_definition-6/google-gemma-2-9b-it-test.tsv --gate_mode sbert-hard --gate_col "Self-Enhancement" --gate_tau 0.5 --hf_token HF_TOKEN 2>&1 | tee run_few-shot-gated-test.txt
python3 eval.py --test-dataset ../../data/test-english/ --model-name "google-gemma-2-9b-it-gated_Self-Enhancement" --output-directory output/self-enhancement_definition-6 | tee output/self-enhancement_definition-6/few-shot-gated-test-results.txt