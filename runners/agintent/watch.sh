#!/bin/zsh
# Poll agintent boxes every 5 min: zsh watch.sh "data r25 r50" [polls]. Exit when any queue.log changes, a newest log is
# silent > 20 min, ssh fails twice for a box, or after N polls. Prints a tail per box.
R=<repo root>/.runtime/agintent; TAGS=(${=1}); N=${2:-12}; why=""
typeset -A q0 bad
probe='wc -l < /workspace/ai_logs/queue.log; f=$(ls -t /workspace/ai_logs/*.log /workspace/ai_logs/*.out 2>/dev/null | grep -v queue | head -1); echo $(( $(date +%s) - $(stat -c %Y ${f:-/workspace/ai_logs/queue.log}) ))'
for i in $(seq 1 $N); do
  for t in $TAGS; do
    r=$(zsh $R/vs.sh $t "$probe" < /dev/null 2>/dev/null)
    if [ -z "$r" ]; then nb=${bad[$t]}; nb=$(( ${nb:-0} + 1 )); bad[$t]=$nb; [ $nb -ge 2 ] && why="$why ${t}:ssh-failed-twice"
    else bad[$t]=0; q=$(echo "$r" | sed -n 1p); age=$(echo "$r" | sed -n 2p); [ -z "${q0[$t]}" ] && q0[$t]=$q
      [ "$q" != "${q0[$t]}" ] && why="$why ${t}:queue-changed"; [ "${age:-0}" -gt 2400 ] && why="$why ${t}:silent-${age}s"; fi
  done
  [ -n "$why" ] && break
  [ $i -lt $N ] && sleep 300
done
echo "WATCH:${why:- polls done} $(date -u +%H:%M)"
for t in $TAGS; do echo "== $t"; zsh $R/vs.sh $t 'tail -n 3 /workspace/ai_logs/queue.log; f=$(ls -t /workspace/ai_logs/*.log | grep -v queue | head -1); echo "[$f]"; tail -n 2 $f | cut -c1-400; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' < /dev/null 2>&1 | tail -n 8; done
