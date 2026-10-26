#!/bin/bash
# One ladder arm: bash /workspace/run_arm.sh NAME POOL [extra trainer args]   (nohup setsid ... &)
# Trains, then uploads the run dir (best LoRA merged into final_lm, evals, metrics, gens) to private HF, then ARM_DONE.
NAME=$1; POOL=$2; shift 2; W=/workspace; L=$W/ai_logs; Q=$L/queue.log; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=/workspace/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
PY=$W/venv/bin/python
until [ -f $L/done_setup ]; do grep -q "FAIL" $Q && exit 1; sleep 20; done
until [ -e $W/vec_ai/$POOL/rows.jsonl ] && [ -e $W/vec_ai/agent_dev/rows.jsonl ] && [ -e $W/vec_ai/rawvar_agent.json ]; do grep -q "FAIL" $Q && exit 1; sleep 30; done
S=runs/nld27_grpo_qwen35-2b_from-nla-qwen36-27b-L42-rl400_pairs526k-op100k_ar-reward_20261012
say "START arm $NAME"
$PY scripts/agintent_grpo_20261024.py --vec $W/vec_ai --init $S --pool $POOL --out runs/$NAME "$@" >> $L/arm_$NAME.log 2>&1 || fail "arm $NAME"
say "DONE arm $NAME"
$PY - <<PY >> $L/arm_$NAME.log 2>&1 || fail "upload $NAME"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN']); n = "$NAME"; r = 'adamjb111/' + n.replace('_', '-')[:90]
api.create_repo(r, private=True, exist_ok=True)
api.upload_folder(folder_path='runs/' + n, repo_id=r, ignore_patterns=['ckpt.pt', 'best.pt', '*.tmp'])
print('uploaded', r)
PY
say "ARM_DONE $NAME"
