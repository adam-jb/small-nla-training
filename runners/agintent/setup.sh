#!/bin/bash
# Agentic-intent GRPO box set-up (24 Oct 2026). nohup bash /workspace/setup.sh data|train > /workspace/ai_logs/setup.out 2>&1 &
# data : venv + vLLM env + Qwen3.6-27B + AR files + GRPO526 student + web vec sets (rollout/extract/mix, then arm r100)
# train: venv + AR files + GRPO526 student + vec_ai from HF (arm only)
MODE=$1; W=/workspace; L=$W/ai_logs; mkdir -p $L; Q=$L/queue.log
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
say "START setup $MODE"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential procps > $L/apt.log 2>&1 || fail apt
python3 -m venv --system-site-packages $W/venv || fail venv
P=$W/venv/bin/pip
$P install -q -U "transformers==5.17.*" "huggingface_hub>=1.0" pyyaml pyarrow pandas hf_xet "jinja2>=3.1" safetensors accelerate > $L/pip.log 2>&1 || fail pip
( $P install -q flash-linear-attention || echo fla_failed
  MAX_JOBS=16 $P install -q --no-build-isolation causal-conv1d || echo conv1d_failed ) > $L/pip2.log 2>&1 &
P2=$!
if [ "$MODE" = data ]; then
  ( U=$(command -v uv || { $P install -q uv >/dev/null; echo $W/venv/bin/uv; }); $U venv -q $W/vllm-env --python 3.12 && VIRTUAL_ENV=$W/vllm-env $U pip install -q "vllm==0.19.0" \
    && echo vllm_ok || echo vllm_failed ) > $L/vllm_pip.log 2>&1 &
  P3=$!
fi
$W/venv/bin/python - <<PY || fail download
import os
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
S = "nld27_grpo_qwen35-2b_from-nla-qwen36-27b-L42-rl400_pairs526k-op100k_ar-reward_20261012"
jobs = [dict(repo_id="ceselder/qwen3.6-27b-nla-rl", local_dir="/workspace/teacher",
             allow_patterns=["ar_reconstructor/*", "av_base/*.json", "av_base/*.txt", "av_base/*.jinja", "nla_meta.yaml", "run_config.yaml"]),
        dict(repo_id="adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013",
             local_dir="/workspace/lenses/runs/" + S),
        dict(repo_id="adamjb111/nla-qwen36-27b-L42-distill-data", repo_type="dataset", local_dir="/workspace/hfdata",
             allow_patterns=["work/vec27b/heldout/*", "work/vec27b/inject/*", "work/vec27b/rawvar.json", "work/vec27b/train/L42_000.npy"])]
if "$MODE" == "data":
    jobs.append(dict(repo_id="Qwen/Qwen3.6-27B"))
else:
    import time
    from huggingface_hub import HfApi
    R = "adamjb111/agintent-vec-qwen36-27b-L42-ladder-20261024"
    while True:  # wait for run_data's upload to finish (its last file is rollouts/vec_agent/rows.jsonl)
        try:
            if HfApi().file_exists(R, "rollouts/vec_agent/rows.jsonl", repo_type="dataset"): break
        except Exception as e: print("wait", e)
        time.sleep(120)
    jobs.append(dict(repo_id=R, repo_type="dataset", local_dir="/workspace/vec_ai_dl", ignore_patterns=["rollouts/*"]))
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(lambda k: snapshot_download(**k, max_workers=16), jobs): print(r)
PY
say "DONE download"
mkdir -p $W/vec_ai
for s in heldout inject; do ln -sfn $W/hfdata/work/vec27b/$s $W/vec_ai/$s; done
cp $W/hfdata/work/vec27b/rawvar.json $W/vec_ai/rawvar.json
mkdir -p $W/vec_ai/web_train && ln -sf $W/hfdata/work/vec27b/train/L42_000.npy $W/vec_ai/web_train/L42_000.npy
if [ "$MODE" = train ]; then
  for d in $W/vec_ai_dl/*; do b=$(basename $d); [ -e $W/vec_ai/$b ] || ln -sfn $d $W/vec_ai/$b; done
fi
wait $P2; cat $L/pip2.log
if [ "$MODE" = data ]; then wait $P3; cat $L/vllm_pip.log; grep -q vllm_ok $L/vllm_pip.log || fail vllm_install
  $W/vllm-env/bin/python -c "import vllm; print('vllm', vllm.__version__)" || fail vllm_env; fi
$W/venv/bin/python -c "import torch, transformers, fla; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'tf', transformers.__version__)" || fail env
ls $W/vec_ai >> $L/setup.out
touch $L/done_setup; say "DONE setup"
