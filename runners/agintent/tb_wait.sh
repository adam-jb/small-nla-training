#!/bin/zsh
# wait for TB_DONE / FAIL on the tokens-before box (poll 2 min, 45 polls)
for i in $(seq 1 45); do
  q=$(perl -e 'alarm 50; exec @ARGV' zsh <repo root>/.runtime/agintent/vs.sh tb 'cat /workspace/ai_logs/queue.log' < /dev/null 2>/dev/null)
  echo "$q" | grep -qE "TB_DONE|FAIL" && { echo "$q"; exit 0; }
  sleep 120
done; echo TIMEOUT; echo "$q"
