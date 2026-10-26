#!/bin/bash
# Resume an arm that stopped on the KL guard: wait for its first ARM_DONE (and any final eval) -> resume from ckpt.pt with
# --kl-max 1.0 until END_UTC (epoch secs) -> second ARM_DONE -> paired final eval -> EVAL_DONE.
# bash /workspace/continue_arm.sh NAME POOL END_EPOCH [extra trainer args]
NAME=$1; POOL=$2; END=$3; shift 3; W=/workspace; L=$W/ai_logs; Q=$L/queue.log
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
until grep -q "ARM_DONE $NAME" $Q; do grep -qE "[0-9] FAIL " $Q && exit 1; sleep 30; done
while pgrep -f agintent_final_eval_20261024.py > /dev/null; do sleep 30; done
R=$W/lenses/runs/$NAME
if ! grep -q '"KL too large"' $R/STOPPED.json 2>/dev/null; then say "NOCONT $NAME (stop was not KL)"; bash $W/run_final_eval.sh $NAME; exit 0; fi
mv $R/STOPPED.json $R/STOPPED_round1.json; cp $R/done.json $R/done_round1.json
H=$(python3 -c "import time; print(round(max(0.2, ($END - time.time()) / 3600), 2))")
say "CONTINUE $NAME kl-max 1.0 hours $H"
bash $W/run_arm.sh $NAME $POOL --hours $H --kl-max 1.0 "$@"
bash $W/run_final_eval.sh $NAME
