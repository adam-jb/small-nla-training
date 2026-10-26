#!/bin/bash
# WSB agentic_misalignment, L42 readers (25 Oct 2026). setsid nohup bash /workspace/run.sh > /workspace/logs/run.out 2>&1 &
# setup -> capture -> AGI-r100 + GRPO526 notes -> export -> start Qwen vLLM (teacher runs later via teacher.sh).
W=/workspace; L=$W/logs; Q=$L/queue.log; mkdir -p $L; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
say "START"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential procps > $L/apt.log 2>&1 || fail apt
python3 -m venv --system-site-packages $W/venv || fail venv
P=$W/venv/bin/pip; PY=$W/venv/bin/python
$P install -q -U "transformers==5.17.*" "huggingface_hub>=1.0" pyyaml peft hf_xet "jinja2>=3.1" safetensors accelerate pyarrow > $L/pip.log 2>&1 || fail pip
( $P install -q flash-linear-attention || echo fla_failed; MAX_JOBS=16 $P install -q --no-build-isolation causal-conv1d || echo conv1d_failed ) > $L/pip2.log 2>&1 &
P2=$!
( python3 -m venv $W/vvenv && $W/vvenv/bin/pip install -q vllm==0.19.0; echo rc=$? ) > $L/vllm_pip.log 2>&1 &
$PY - <<PY || fail download
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
snapshot_download("ceselder/qwen3.6-27b-nla-rl", local_dir="$W/teacher", max_workers=16,
                  allow_patterns=["av_base/*", "av_rl_adapters/iter_000400/*", "nla_meta.yaml", "run_config.yaml"])
snapshot_download("adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013",
                  local_dir="$W/m/grpo526", allow_patterns=["final_lm/*", "final_adapter.pt"])
snapshot_download("adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r100-rows24k-20261024",
                  local_dir="$W/m/agi_r100", allow_patterns=["final_lm/*", "final_adapter.pt"])
PY
wait $P2; say "DONE setup $(tr '\n' ' ' < $L/pip2.log)"
S="$PY scripts/wsbam_box_20261025.py"; D=$W/out
$S capture --items data/wsbam_20261025/items.jsonl --d $D >> $L/capture.log 2>&1 || fail capture
say "DONE capture $(tail -1 $L/capture.log | cut -c1-200)"
for n in agi_r100 grpo526; do
  $S student --d $D --k 1 --bs 512 --model $W/m/$n --name $n >> $L/student.log 2>&1 || fail student_$n
  say "DONE student $n $(tail -1 $L/student.log | cut -c1-160)"
  $S export --d $D --out $W/readouts >> $L/export.log 2>&1 || fail export
done
say "STUDENTS_DONE"
until grep -q "rc=" $L/vllm_pip.log; do sleep 20; done
say "vllm $(tail -1 $L/vllm_pip.log)"
(setsid nohup bash $W/serve.sh > $L/serve.out 2>&1 < /dev/null &)
for i in $(seq 1 90); do curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && break; sleep 20; done
curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && say "QWEN_READY" || fail qwen_serve
