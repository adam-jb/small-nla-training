#!/bin/zsh
# Poll every 2 min; destroy a box when a 25 Oct EVAL_DONE appears; stop on FAIL or after 40 polls.
R=<repo root>/.runtime/agintent; left=(data r25 r50 j100)
for i in $(seq 1 40); do
  for t in $left; do
    q=$(perl -e 'alarm 50; exec @ARGV' zsh $R/vs.sh $t 'grep -E "^2026-10-25_0[1-9].*(EVAL_DONE|FAIL)" /workspace/ai_logs/queue.log' < /dev/null 2>/dev/null)
    if echo "$q" | grep -q FAIL; then echo "FAIL on $t: $q $(date -u +%H:%M)"; exit 1; fi
    if echo "$q" | grep -q EVAL_DONE; then id=$(cat <repo root>/.runtime/vast_instance_id_ai_${t}); echo "$t EVAL_DONE $(date -u +%H:%M) -> destroy $id"; echo y | vastai destroy instance $id; left=(${left:#$t}); fi
  done
  [ ${#left} -eq 0 ] && { echo "ALL DONE $(date -u +%H:%M)"; exit 0; }
  sleep 120
done
echo "TIMEOUT left: $left"
