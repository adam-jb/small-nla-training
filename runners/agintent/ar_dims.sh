#!/bin/bash
# after T250_DONE: big-dimension check on the tokens-before notes (teacher, GRPO526, AGI-r25)
W=/workspace; L=$W/ai_logs; Q=$L/queue.log; cd $W/lenses
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false NLA_TEACHER_DIR=$W/teacher; PY=$W/venv/bin/python
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
until grep -qE "T250_DONE" $Q; do sleep 20; done
cat $W/tc/before/T_r*.jsonl | grep -v _scored > /dev/null; cat $(ls $W/tc/before/T_r*.jsonl | grep -v _scored) > $W/tc/before_teacher_all.jsonl
$PY scripts/agintent_ar_dims_check_20261025.py $W/vec before $W/ar_dims_before.json teacher=$W/tc/before_teacher_all.jsonl GRPO526=$W/tb/notes_GRPO526.jsonl AGI-r25=$W/tb/notes_AGI-r25.jsonl >> $L/ar_dims.log 2>&1 && say "DIMS_DONE" || say "FAIL dims"
