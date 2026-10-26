#!/bin/bash
# Teacher, split across both boxes (26 Oct 10:55): half the scenarios each for partials and premerge, then Sonnet.
cd <repo root>
W=/private/tmp/claude-501/-Users-adambricknail-Desktop-lenses/135ed19d-3558-4795-baca-a7bedd0b47a3/scratchpad/wsb
export WSB_REPO=$W/repo QWEN_WORKERS=48
J="$W/wvenv/bin/python scripts/wsbam_judge_20261025.py"
L=data/wsbam_20261025/logs; Q=$L/chain.log
say() { echo "$(date -u +%F_%T) $1" >> $Q; }
A=$(sed -n 1p data/wsbam_20261025/teacher_split.txt); B=$(sed -n 2p data/wsbam_20261025/teacher_split.txt)
half() {  # $1 port, $2 ids
  WSB_QWEN_URL=http://127.0.0.1:$1/v1 $J partials --reader teacher --tag teacher --backend qwen --no-think --ids $2 >> $L/partials_teacher.log 2>&1 \
  && WSB_QWEN_URL=http://127.0.0.1:$1/v1 $J premerge --reader teacher --tag teacher --no-think --ids $2 >> $L/premerge_teacher.log 2>&1 \
  && say "DONE teacher half port $1" || say "FAIL teacher half port $1"
}
say "START teacher split"
half 18000 $A &
half 18001 $B &
wait
$J consolidate --reader teacher --tag teacher --grouped --allow-missing >> $L/bc_teacher.log 2>&1 \
  && $J score --tag teacher >> $L/bc_teacher.log 2>&1 && say "SONNET_DONE teacher" || say "FAIL sonnet teacher"
say "QWEN_IDLE TEACHER_ALL_DONE"
