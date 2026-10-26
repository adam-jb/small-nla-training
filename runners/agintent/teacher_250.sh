#!/bin/bash
# Teacher on 250 per row type + 250 web rows (25 Oct 2026); runs after TB_DONE on the tokens-before box.
W=/workspace; L=$W/ai_logs; Q=$L/queue.log; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher; PY=$W/venv/bin/python
say() { echo "$(date -u +%F_%T) $1" >> $Q; }; fail() { say "FAIL $1"; exit 1; }
until grep -qE "TB_DONE|FAIL" $Q; do sleep 30; done; grep -q TB_DONE $Q || exit 1
V=$W/vec250; mkdir -p $V; cp -r $W/agent_s750 $V/; cp $W/agent_s750/rawvar.json $V/rawvar.json
$PY - <<PY || fail web_dl
from huggingface_hub import snapshot_download
snapshot_download("adamjb111/nla-qwen36-27b-L42-distill-data", repo_type="dataset", local_dir="$W/hfdata", allow_patterns=["work/vec27b/heldout/*", "work/vec27b/rawvar.json"])
PY
mkdir -p $V/w; cp -r $W/hfdata/work/vec27b/heldout $V/w/; cp $W/hfdata/work/vec27b/rawvar.json $V/w/rawvar.json
$PY scripts/nladistil27b_teacher_20261010.py --vec $V --set agent_s750 --out $W/t250 --backend hf --merged $W/teacher_merged --hf-bs 150 --shard 150 >> $L/t250.log 2>&1 || fail t250_agent
$PY scripts/nladistil27b_teacher_20261010.py --vec $V/w --set heldout --rows 0:250 --out $W/t250 --backend hf --merged $W/teacher_merged --hf-bs 125 --shard 125 >> $L/t250.log 2>&1 || fail t250_web
$PY scripts/nladistil27b_ar_score_20261010.py --vec $V --set agent_s750 $W/t250/agent_s750/T_r*.jsonl >> $L/t250.log 2>&1 || fail t250_ar_agent
$PY scripts/nladistil27b_ar_score_20261010.py --vec $V/w --set heldout $W/t250/heldout/T_r*.jsonl >> $L/t250.log 2>&1 || fail t250_ar_web
say "T250_DONE"
