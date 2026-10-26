"""Build the teacher-check sample (25 Oct 2026): 50 agent_hold rows per row type (P_last, P_mid, G) from the SAME rows the
final eval scored (pick_rows, per-source 400), so teacher notes line up with GRPO526 / arm notes. Writes OUT/agent_s150/
{rows.jsonl, L42_000.npy, rawvar.json (agent rawvar)} and OUT/agent_s150_join.json (sample row -> index in notes_hold.jsonl).
Usage: .venv/bin/python scripts/agintent_teacher_sample_20261025.py VEC_DIR OUT_DIR [PER_TYPE=50]"""
import json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.agintent_final_eval_20261024 import pick_rows

VEC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 50
NAME = f'agent_s{3 * K}'
hrows = pick_rows(VEC, 400)
rng = random.Random(25)
pick = []
for t in ['P_last', 'P_mid', 'G']:
    ix = [j for j, r in enumerate(hrows) if r['rtype'] == t]
    pick += sorted(rng.sample(ix, K))
raw = np.load(VEC / 'agent_hold' / 'L42_000.npy')
d = OUT / NAME; d.mkdir(parents=True, exist_ok=True)
np.save(d / 'L42_000.npy', raw[[hrows[j]['i'] for j in pick]])
with open(d / 'rows.jsonl', 'w') as f:
    for n, j in enumerate(pick):
        r = {k: v for k, v in hrows[j].items() if k != 'ctx'}
        f.write(json.dumps({**r, 'hold_i': r['i'], 'i': n, 'doc_id': r['group']}) + '\n')
json.dump({'rawvar': json.load(open(VEC / 'rawvar_agent.json'))['rawvar'], 'n': 'agent rawvar (train_r100 agent rows)'}, open(d / 'rawvar.json', 'w'))
json.dump(pick, open(OUT / f'{NAME}_join.json', 'w'))
print(len(pick), raw.dtype, raw.shape, [hrows[j]['source'] for j in pick[:5]])
