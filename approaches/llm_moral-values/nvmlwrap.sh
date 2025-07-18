#!/bin/bash
export LD_PRELOAD=$HOME/fake_nvml/libnvidia-ml.so.1
export LD_LIBRARY_PATH="$HOME/fake_nvml:$LD_LIBRARY_PATH"
export PYTORCH_NVML_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
# rm output/base/llm_cache.json
exec "$@"