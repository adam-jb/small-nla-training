"""Tokens before the decision point (25 Oct 2026). Adam: the decision point (last prompt token, the "\\n" after <think>)
reads badly; can the NLA read the tokens leading up to it instead?
  pick    (Mac): 100 holdout units from the final-eval decision-point rows, spread evenly over prompt length -> UNITS_JSON
  extract (box): for each unit's k=0 prompt, L42 vectors at offsets OFFS back from the end (1 = decision point) from one
                 Qwen3.6-27B forward pass (same States helper as the ladder) -> VEC/before/{rows.jsonl, L42_000.npy}
  score2b (box): GRPO526 and AGI-r25 write notes (same sampler/seed as the final eval), 27B AR scores -> OUT/notes_<model>.jsonl
  (teacher notes: scripts/nladistil27b_teacher_20261010.py --set before, scored by nladistil27b_ar_score_20261010.py)
Usage: python scripts/agintent_tokens_before_20261025.py pick VEC_LOCAL UNITS_JSON
       python scripts/agintent_tokens_before_20261025.py extract ROLLOUTS UNITS_JSON VEC
       python scripts/agintent_tokens_before_20261025.py score2b VEC OUT MODEL_DIR [MODEL_DIR ...]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OFFS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24]


def pick(vec, out):
    from scripts.agintent_final_eval_20261024 import pick_rows
    P = sorted([r for r in pick_rows(vec, 400) if r['rtype'] == 'P_last'], key=lambda r: r['pos'])
    step = len(P) / 100
    units = [P[int(i * step)]['unit_id'] for i in range(100)]
    json.dump(units, open(out, 'w'))
    print(len(units), 'prompt tokens', P[0]['pos'] + 1, '...', P[-1]['pos'] + 1)


def extract(rollouts, units_json, vec):
    import torch
    from transformers import AutoTokenizer
    from scripts.nladistil27b_common_20261010 import SUBJECT
    from scripts.antischem_box_20261017 import States, save_set
    want = set(json.load(open(units_json))); tok = AutoTokenizer.from_pretrained(SUBJECT); st = States()
    rows, Hs = [], []
    for l in open(rollouts):
        r = json.loads(l)
        if r['k'] != 0 or r['unit_id'] not in want:
            continue
        ids = r['prompt_ids']; n = len(ids)
        ps = [n - o for o in OFFS if n - o >= 0]
        for o, p, h in zip(OFFS, ps, st.at(ids, ps)):
            rows.append({'unit_id': r['unit_id'], 'source': r['source'], 'slice': r['slice'], 'group': r['group'], 'doc_id': r['group'],
                         'off': o, 'pos': p, 'n_prompt': n, 'tok': tok.decode([ids[p]]),
                         'ctx': tok.decode(ids[max(0, p - 30):p + 1])[-160:]})
            Hs.append(h)
    save_set(vec, 'before', rows, torch.stack(Hs))
    print(json.dumps({'units': len({r['unit_id'] for r in rows}), 'rows': len(rows),
                      'toks_by_off': {o: sorted({r['tok'] for r in rows if r['off'] == o})[:4] for o in OFFS}}), flush=True)


def score2b(vec, out, models):
    import torch
    from scripts.nladistil27b_common_20261010 import AR, LAYER, rl
    from scripts.nladistil27b_extract_20261010 import load_set
    from scripts.agintent_final_eval_20261024 import score_model
    rows = rl(Path(vec) / 'before' / 'rows.jsonl'); V = load_set(vec, 'before', LAYER)
    ar = AR('cuda'); Path(out).mkdir(parents=True, exist_ok=True)
    for m in models:
        res = score_model(m, {'before': (V, list(range(len(rows))))}, ar, 'cuda', 0, 512)['before']
        with open(Path(out) / f'notes_{Path(m).name}.jsonl', 'w') as f:
            for r, s in zip(rows, res):
                f.write(json.dumps({**{k: r[k] for k in ('unit_id', 'source', 'off', 'pos', 'n_prompt', 'tok')}, **s}) + '\n')


if __name__ == '__main__':
    s = sys.argv[1]
    if s == 'pick':
        pick(Path(sys.argv[2]), sys.argv[3])
    elif s == 'extract':
        extract(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        score2b(sys.argv[2], sys.argv[3], sys.argv[4:])
