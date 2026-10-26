"""WorkspaceBench agentic_misalignment at L42 (25 Oct 2026): render the 32 scenarios with the bench's own render() and
position rule (every prompt position, chat render, thinking off), so the box only runs one forward pass per scenario.
One row per scenario: {id, ids (full render), pos (read positions), arm, group}.
Usage (bench venv): wvenv/bin/python scripts/wsbam_render_items_20261025.py OUT.jsonl"""
import json, sys
from transformers import AutoTokenizer
from wsbench import readplan
from pathlib import Path
import wsbench
from wsbench.produce.producer import _positions
from wsbench.produce.render import render

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B')
meta = {i['id']: i for i in json.load(open(Path(wsbench.__file__).parents[2] / 'evals/agentic_misalignment/items.json'))['items']}
n = 0
with open(sys.argv[1], 'w') as f:
    for spec in readplan.plan('agentic_misalignment'):
        r = render(spec, tok)
        pos = _positions(spec.positions, r.tokens)
        m = meta[spec.id]
        f.write(json.dumps({'id': spec.id, 'ids': list(r.ids), 'pos': pos, 'arm': m['arm'], 'group': m['group']}) + '\n')
        n += len(pos)
print(n, 'positions')
