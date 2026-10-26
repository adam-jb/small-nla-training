#!/bin/bash
# Qwen3.6-27B judge for Stage B chunk summaries (thinking on, reasoning parser). Long context: 60k-char chunks + 16k reply.
export HF_HOME=/workspace/hf HF_TOKEN=$(cat /workspace/.hf_token)
exec /workspace/vvenv/bin/vllm serve Qwen/Qwen3.6-27B --served-model-name qwen --port 8000 --max-model-len 65536 \
  --gpu-memory-utilization 0.92 --max-num-seqs 64 --reasoning-parser qwen3 --enable-prefix-caching --max-num-batched-tokens 32768 --kv-cache-dtype fp8
