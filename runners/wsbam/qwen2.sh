#!/bin/bash
# Second WSB judge box (26 Oct): vLLM Qwen3.6-27B only. setsid nohup bash /workspace/qwen2.sh > /workspace/logs/qwen2.out 2>&1 &
W=/workspace; L=$W/logs; Q=$L/queue.log; mkdir -p $L
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
say START
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv procps curl > $L/apt.log 2>&1 || { say "FAIL apt"; exit 1; }
rm -rf $W/vvenv
python3 -m venv $W/vvenv && $W/vvenv/bin/pip install -q vllm==0.19.0 > $L/vllm_pip.log 2>&1 || { say "FAIL pip"; exit 1; }
$W/vvenv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)' > $L/dl.log 2>&1 || { say "FAIL dl"; exit 1; }
say "DONE setup"
(setsid nohup bash $W/serve.sh > $L/serve.out 2>&1 < /dev/null &)
for i in $(seq 1 90); do curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && break; sleep 20; done
curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && say QWEN_READY || say "FAIL serve"
