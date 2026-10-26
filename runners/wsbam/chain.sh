#!/bin/bash
# WSB misalignment full run, unattended (Adam 26 Oct: all three readers). Mac side; run with bash (zsh won't split $J).
# Box 1 (wsbam): teacher.sh already running on the box (vLLM down) -> pull teacher notes -> Qwen: AGI premerge + GRPO526 and teacher
# partials together -> premerge -> Sonnet 5.5 batch merge + score per reader -> report.
cd <repo root>
W=/private/tmp/claude-501/-Users-adambricknail-Desktop-lenses/135ed19d-3558-4795-baca-a7bedd0b47a3/scratchpad/wsb
export WSB_REPO=$W/repo
J="$W/wvenv/bin/python scripts/wsbam_judge_20261025.py"
L=data/wsbam_20261025/logs; Q=$L/chain.log
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
vs() { perl -e 'alarm 600; exec @ARGV' zsh .runtime/agintent/vs.sh wsbam "$@"; }
sonnet() {  # merge + score one reader via batch (Sonnet 5.5)
  $J consolidate --reader $1 --tag $1 --grouped --allow-missing >> $L/bc_$1.log 2>&1 \
    && $J score --tag $1 >> $L/bc_$1.log 2>&1 && say "SONNET_DONE $1" || say "FAIL sonnet $1"
}
reader() {  # $1 reader, $2 workers: Qwen partials (skipped if cached) -> premerge -> Sonnet
  QWEN_WORKERS=$2 $J partials --reader $1 --tag $1 --backend qwen --no-think >> $L/partials_$1.log 2>&1 || { say "FAIL partials $1"; return 1; }
  say "DONE partials $1 $(wc -l < data/wsbam_20261025/judge/partials_$1.jsonl)"
  QWEN_WORKERS=$2 $J premerge --reader $1 --tag $1 --no-think >> $L/premerge_$1.log 2>&1 || { say "FAIL premerge $1"; return 1; }
  say "DONE premerge $1"
  sonnet $1
}

say "START3 box1 (waiting for teacher.sh)"
until vs 'tail -1 /workspace/logs/queue.log' | grep -qE "TEACHER_DONE|FAIL"; do sleep 300; done
vs 'tail -1 /workspace/logs/queue.log' | grep -q TEACHER_DONE || { say "FAIL teacher $(vs 'tail -1 /workspace/logs/queue.log')"; exit 1; }
vs 'cat /workspace/readouts/teacher.jsonl' > data/wsbam_20261025/readouts/teacher.jsonl
say "pulled teacher $(wc -l < data/wsbam_20261025/readouts/teacher.jsonl) rows"
reader teacher 48
say "BOX1_DONE"
