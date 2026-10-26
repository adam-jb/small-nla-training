#!/bin/zsh
# wait for TC_DONE / FAIL on the teacher-check box (poll 2 min, 40 polls)
for i in $(seq 1 40); do
  q=$(perl -e 'alarm 50; exec @ARGV' zsh <repo root>/.runtime/agintent/vs.sh tc 'cat /workspace/ai_logs/queue.log' < /dev/null 2>/dev/null)
  echo "$q" | grep -vE "08:31:29 FAIL" | grep -qE "TC_DONE|FAIL" && { echo "$q"; exit 0; }
  sleep 120
done; echo TIMEOUT; echo "$q"
