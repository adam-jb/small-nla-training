#!/bin/bash
# Teacher check (25 Oct 2026): 27B NLA teacher (HF backend) writes notes for 150 agent_hold rows + 51 web held-out rows,
# then the 27B AR scores them. setsid nohup bash /workspace/teacher_check.sh > /workspace/ai_logs/tc.out 2>&1 &
W=/workspace; L=$W/ai_logs; Q=$L/queue.log; mkdir -p $L; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
say "START teacher_check"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential procps > $L/apt.log 2>&1 || fail apt
python3 -m venv --system-site-packages $W/venv || fail venv
P=$W/venv/bin/pip; PY=$W/venv/bin/python
$P install -q -U "transformers==5.17.*" "huggingface_hub>=1.0" pyyaml peft hf_xet "jinja2>=3.1" safetensors accelerate > $L/pip.log 2>&1 || fail pip
( $P install -q flash-linear-attention || echo fla_failed; MAX_JOBS=16 $P install -q --no-build-isolation causal-conv1d || echo conv1d_failed ) > $L/pip2.log 2>&1 &
P2=$!
$PY - <<PY || fail download
from huggingface_hub import snapshot_download
snapshot_download("ceselder/qwen3.6-27b-nla-rl", local_dir="$W/teacher", max_workers=16,
                  allow_patterns=["ar_reconstructor/*", "av_base/*", "av_rl_adapters/iter_000400/*", "nla_meta.yaml", "run_config.yaml"])
snapshot_download("adamjb111/nla-qwen36-27b-L42-distill-data", repo_type="dataset", local_dir="$W/hfdata",
                  allow_patterns=["work/vec27b/heldout/*", "work/vec27b/rawvar.json"])
PY
wait $P2; say "DONE setup $(cat $L/pip2.log | tr '\n' ' ')"
V=$W/vec; mkdir -p $V; cp -r $W/agent_s150 $V/; cp -r $W/hfdata/work/vec27b/heldout $V/; cp $W/hfdata/work/vec27b/rawvar.json $V/web_rawvar.json
$PY scripts/nladistil27b_teacher_20261010.py --vec $V --set agent_s150 --out $W/tc --backend hf --merged $W/teacher_merged --hf-bs 150 >> $L/tc_gen.log 2>&1 || fail gen_agent
$PY scripts/nladistil27b_teacher_20261010.py --vec $V --set heldout --rows 0:51 --out $W/tc --backend hf --merged $W/teacher_merged --hf-bs 51 >> $L/tc_gen.log 2>&1 || fail gen_web
say "DONE gen"
cp $V/agent_s150/rawvar.json $V/rawvar.json
$PY scripts/nladistil27b_ar_score_20261010.py --vec $V --set agent_s150 $W/tc/agent_s150/T_r0000000.jsonl >> $L/tc_ar.log 2>&1 || fail ar_agent
mkdir -p $V/w && cp -r $V/heldout $V/w/ && cp $V/web_rawvar.json $V/w/rawvar.json
$PY scripts/nladistil27b_ar_score_20261010.py --vec $V/w --set heldout $W/tc/heldout/T_r0000000.jsonl >> $L/tc_ar.log 2>&1 || fail ar_web
say "TC_DONE"
