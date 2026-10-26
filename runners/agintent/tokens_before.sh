#!/bin/bash
# Tokens before the decision point (25 Oct 2026): extract L42 at 12 offsets for 100 units, then teacher + GRPO526 + AGI-r25 notes, 27B AR.
# setsid nohup bash /workspace/tokens_before.sh > /workspace/ai_logs/tb.out 2>&1 &
W=/workspace; L=$W/ai_logs; Q=$L/queue.log; mkdir -p $L; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
say "START tokens_before"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential procps > $L/apt.log 2>&1 || fail apt
python3 -m venv --system-site-packages $W/venv || fail venv
P=$W/venv/bin/pip; PY=$W/venv/bin/python
$P install -q -U "transformers==5.17.*" "huggingface_hub>=1.0" pyyaml peft hf_xet "jinja2>=3.1" safetensors accelerate pyarrow pandas > $L/pip.log 2>&1 || fail pip
( $P install -q flash-linear-attention || echo fla_failed; MAX_JOBS=16 $P install -q --no-build-isolation causal-conv1d || echo conv1d_failed ) > $L/pip2.log 2>&1 &
P2=$!
$PY - <<PY || fail download
from huggingface_hub import snapshot_download
snapshot_download("ceselder/qwen3.6-27b-nla-rl", local_dir="$W/teacher", max_workers=16,
                  allow_patterns=["ar_reconstructor/*", "av_base/*", "av_rl_adapters/iter_000400/*", "nla_meta.yaml", "run_config.yaml"])
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
snapshot_download("adamjb111/agintent-vec-qwen36-27b-L42-ladder-20261024", repo_type="dataset", local_dir="$W/hfdata", allow_patterns=["rollouts/rollouts.jsonl"])
for r, d in [("adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013", "GRPO526"),
             ("adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r25-rows6k-20261024", "AGI-r25")]:
    snapshot_download(r, local_dir="$W/models/" + d, allow_patterns=["final_lm/*", "final_adapter.pt"])
PY
wait $P2; say "DONE setup $(cat $L/pip2.log | tr '\n' ' ')"
V=$W/vec; mkdir -p $V; cp $W/rawvar_agent.json $V/rawvar.json
$PY scripts/agintent_tokens_before_20261025.py extract $W/hfdata/rollouts/rollouts.jsonl $W/before_units100.json $V >> $L/tb_extract.log 2>&1 || fail extract
say "DONE extract"
$PY scripts/agintent_tokens_before_20261025.py score2b $V $W/tb $W/models/GRPO526 $W/models/AGI-r25 >> $L/tb_2b.log 2>&1 || fail score2b
say "DONE score2b"
$PY scripts/nladistil27b_teacher_20261010.py --vec $V --set before --out $W/tc --backend hf --merged $W/teacher_merged --hf-bs 200 --shard 400 >> $L/tb_gen.log 2>&1 || fail gen_teacher
say "DONE gen_teacher"
$PY scripts/nladistil27b_ar_score_20261010.py --vec $V --set before $W/tc/before/T_r*.jsonl >> $L/tb_ar.log 2>&1 || fail ar_teacher
say "TB_DONE"
