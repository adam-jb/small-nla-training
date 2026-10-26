#!/bin/bash
# Stop Qwen, write teacher notes, export, restart Qwen. setsid nohup bash /workspace/teacher.sh > /workspace/logs/teacher.out 2>&1 &
W=/workspace; L=$W/logs; Q=$L/queue.log; cd $W/lenses
export HF_HOME=$W/hf HF_TOKEN=$(cat $W/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
fail() { say "FAIL $1"; exit 1; }
ps -eo pid,args | grep -E "[v]llm serve|[V]LLM|EngineCore" | awk '{print $1}' | xargs -r kill; sleep 15
PY=$W/venv/bin/python; S="$PY scripts/wsbam_box_20261025.py"; D=$W/out
$S teacher --d $D --k 1 --bs 160 >> $L/teacher.log 2>&1 || fail teacher
say "DONE teacher $(tail -1 $L/teacher.log | cut -c1-160)"
$S export --d $D --out $W/readouts >> $L/export.log 2>&1 || fail export
(setsid nohup bash $W/serve.sh > $L/serve.out 2>&1 < /dev/null &)
for i in $(seq 1 90); do curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && break; sleep 20; done
curl -s -m 5 127.0.0.1:8000/v1/models | grep -q qwen && say "TEACHER_DONE QWEN_READY" || fail qwen_serve2
