#!/usr/bin/env bash

# echo "=========="
# echo "=========="
# echo "=========="
# echo "===== Qwen/Qwen2.5-7B-Instruct ====="
# echo "=========="
# echo "=========="
# echo "=========="
# 
# echo "=========="
# echo "===== direct ====="
# echo "=========="
# 
# mkdir output/direct
# python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct | tee output/direct/Qwen-Qwen2-5-7B-Instruct-val-results.txt
# 
# echo "=========="
# echo "===== cot_hidden ====="
# echo "=========="
# 
# mkdir output/cot_hidden
# python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id cot_hidden --run_name cot_hidden --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/cot_hidden | tee output/cot_hidden/Qwen-Qwen2-5-7B-Instruct-val-results.txt
# echo "=========="
# echo "===== qa ====="
# echo "=========="
# 
# mkdir output/qa
# python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id qa --run_name qa --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/qa | tee output/qa/Qwen-Qwen2-5-7B-Instruct-val-results.txt
# 
# echo "=========="
# echo "===== definition ====="
# echo "=========="
# 
# mkdir output/definition
# python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id definition --run_name definition --hf_token HF_TOKEN
# python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/definition | tee output/definition/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 1 ====="
echo "=========="

mkdir output/direct-1
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-1 --mode few-shot --k 1 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-1 | tee output/direct-1/Qwen-Qwen2-5-7B-Instruct-val-results.txt
echo "=========="
echo "===== direct - 2 ====="
echo "=========="

mkdir output/direct-2
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-2 --mode few-shot --k 2 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-2 | tee output/direct-2/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 4 ====="
echo "=========="

mkdir output/direct-4
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-4 --mode few-shot --k 4 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-4 | tee output/direct-4/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 8 ====="
echo "=========="

mkdir output/direct-8
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-8 --mode few-shot --k 8 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-8 | tee output/direct-8/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 12 ====="
echo "=========="

mkdir output/direct-12
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-12 --mode few-shot --k 12 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-12 | tee output/direct-12/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 16 ====="
echo "=========="

mkdir output/direct-16
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-16 --mode few-shot --k 16 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-16 | tee output/direct-16/Qwen-Qwen2-5-7B-Instruct-val-results.txt

echo "=========="
echo "===== direct - 20 ====="
echo "=========="

mkdir output/direct-20
python3 main.py --model Qwen/Qwen2.5-7B-Instruct --split val --prompt_id direct --run_name direct-20 --mode few-shot --k 20 --hf_token HF_TOKEN
python3 eval.py --validation-dataset ../../data/validation-english/ --model-name Qwen-Qwen2-5-7B-Instruct --output-directory output/direct-20 | tee output/direct-20/Qwen-Qwen2-5-7B-Instruct-val-results.txt