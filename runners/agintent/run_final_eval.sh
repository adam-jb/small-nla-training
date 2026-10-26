#!/bin/bash
# After ARM_DONE: paired final eval (start GRPO526 vs arm best) -> upload to the arm's HF repo -> EVAL_DONE.
# bash /workspace/run_final_eval.sh NAME   (setsid nohup ... &)
NAME=$1; W=/workspace; L=$W/ai_logs; Q=$L/queue.log; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=/workspace/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
until grep -qE "ARM_DONE $NAME|[0-9] FAIL " $Q; do sleep 30; done
grep -q "ARM_DONE $NAME" $Q || exit 1
S=runs/nld27_grpo_qwen35-2b_from-nla-qwen36-27b-L42-rl400_pairs526k-op100k_ar-reward_20261012
say "START final_eval $NAME"
$W/venv/bin/python scripts/agintent_final_eval_20261024.py --vec $W/vec_ai --arm runs/$NAME --start $S --out runs/$NAME/final_eval >> $L/final_eval.log 2>&1 || { say "FAIL final_eval $NAME"; exit 1; }
$W/venv/bin/python - <<PY >> $L/final_eval.log 2>&1 || { say "FAIL eval_upload $NAME"; exit 1; }
import os
from huggingface_hub import HfApi
n = "$NAME"; r = 'adamjb111/' + n.replace('_', '-')[:90]
HfApi(token=os.environ['HF_TOKEN']).upload_folder(folder_path='runs/' + n + '/final_eval', path_in_repo='final_eval', repo_id=r)
print('uploaded final_eval', r)
PY
say "EVAL_DONE $NAME"
