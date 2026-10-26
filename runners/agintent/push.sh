#!/bin/zsh
# Push code + units to a box: zsh push.sh BOXTAG   (BOXTAG -> .runtime/vast_instance_id_ai_<tag>)
T=$1; cd <repo root>
COPYFILE_DISABLE=1 tar --no-xattrs -czf - scripts/__init__.py scripts/agintent_*.py scripts/antischem_box_20261017.py scripts/antischem_env_20261017.py scripts/antischem_envs \
  scripts/nladistil27b_common_20261010.py scripts/nladistil27b_extract_20261010.py scripts/nladistil27b_ar_score_20261010.py \
  .runtime/agintent/*.sh $( [ -f data/agintent_units_20261024/units_split.jsonl ] && echo data/agintent_units_20261024/units_split.jsonl ) \
  | zsh .runtime/agintent/vs.sh $T 'mkdir -p /workspace/lenses /workspace/ai_logs && cd /workspace/lenses && tar xzf - && cp .runtime/agintent/*.sh /workspace/ && ls /workspace/*.sh | tr "\n" " "'
