#!/bin/bash
# WSB misalignment full run, unattended (Adam 26 Oct: all three readers). Mac side; run with bash (zsh won't split $J).
# Box 2 (wsbam2, Qwen on Mac port 18001): AGI-r100 premerge + GRPO526 partials/premerge; meanwhile box 1 runs teacher.sh (vLLM down) -> pull teacher notes -> Qwen: AGI premerge + GRPO526 and teacher
# partials together -> premerge -> Sonnet 5.5 batch merge + score per reader -> report.
cd <repo root>
W=/private/tmp/claude-501/-Users-adambricknail-Desktop-lenses/135ed19d-3558-4795-baca-a7bedd0b47a3/scratchpad/wsb
export WSB_REPO=$W/repo WSB_QWEN_URL=http://127.0.0.1:18001/v1
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

say "START box2"
reader agi_r100 12 &
reader grpo526 36 &
wait
say "BOX2_DONE"
