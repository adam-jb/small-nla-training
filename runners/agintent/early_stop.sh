#!/bin/bash
# Budget early stop (25 Oct): at AT_EPOCH, stop the round-2 trainer cleanly (SIGTERM saves ckpt.pt), then resume with
# --hours 0 so it exports the best checkpoint + uploads (ARM_DONE), then the paired final eval (EVAL_DONE).
# bash /workspace/early_stop.sh NAME POOL AT_EPOCH
NAME=$1; POOL=$2; AT=$3; W=/workspace; L=$W/ai_logs; Q=$L/queue.log
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
while [ $(date +%s) -lt $AT ]; do pgrep -f "agintent_grpo_20261024.py.*$NAME" > /dev/null || { say "EARLYSTOP skipped $NAME (trainer gone)"; exit 0; }; sleep 30; done
pkill -f "continue_arm.sh $NAME" ; pkill -f "run_arm.sh $NAME"
P=$(pgrep -f "agintent_grpo_20261024.py.*$NAME")
[ -z "$P" ] && { say "EARLYSTOP no trainer $NAME"; exit 1; }
say "EARLYSTOP SIGTERM $NAME pid $P"
kill -TERM $P; while kill -0 $P 2>/dev/null; do sleep 5; done
bash $W/run_arm.sh $NAME $POOL --hours 0 --kl-max 1.0
bash $W/run_final_eval.sh $NAME
