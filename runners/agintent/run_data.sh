#!/bin/bash
# Data steps on box D: rollout -> extract -> mix -> upload vec_ai (private HF) -> DATA_DONE. Resumable.
W=/workspace; L=$W/ai_logs; Q=$L/queue.log; O=$W/ai; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
PY=$W/venv/bin/python; VP=$W/vllm-env/bin/python; B=scripts/agintent_box_20261024.py
until [ -f $L/done_setup ]; do grep -q "FAIL" $Q && exit 1; sleep 20; done
mkdir -p $O
step() { local n=$1; shift; [ -f $O/ok_$n ] && return 0; say "START $n"; "$@" >> $L/$n.log 2>&1 || fail $n; touch $O/ok_$n; say "DONE $n"; }
step rollout $VP $B rollout ${ROLL_ARGS}
step extract $PY $B extract
step mix $PY $B mix
step upload $PY - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN']); r = 'adamjb111/agintent-vec-qwen36-27b-L42-ladder-20261024'
api.create_repo(r, repo_type='dataset', private=True, exist_ok=True)
for s in ['train_r25', 'train_r50', 'train_r100', 'agent_dev', 'agent_hold']:
    api.upload_folder(folder_path=f'/workspace/vec_ai/{s}', path_in_repo=s, repo_id=r, repo_type='dataset')
for f in ['rawvar_agent.json', 'mix_report.json']:
    api.upload_file(path_or_fileobj=f'/workspace/vec_ai/{f}', path_in_repo=f, repo_id=r, repo_type='dataset')
api.upload_folder(folder_path='/workspace/ai', path_in_repo='rollouts', repo_id=r, repo_type='dataset',
                  allow_patterns=['rollouts.jsonl', 'vec_agent/rows.jsonl'])
print('uploaded', r)
PY
say "DATA_DONE"
