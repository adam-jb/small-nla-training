"""WorkspaceBench tiers 1+2 (25 Oct 2026): render every plan row with the bench's own render() + position rules and the
Qwen3.6-27B tokenizer, so the box only has to run forward passes. One output row per (family, item, read position):
{family, id, pos (as the bench reports it), idx (absolute), token, ids}.
Usage (bench venv): wvenv/bin/python scripts/wsbt12_render_cells_20261025.py FAMILIES OUT.jsonl"""
import json, sys
from transformers import AutoTokenizer
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render

fams, out = sys.argv[1].split(','), sys.argv[2]
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B')
n = 0
with open(out, 'w') as f:
    for fam in fams:
        for spec in readplan.plan(fam):
            r = render(spec, tok)
            pos = _positions(spec.positions, r.tokens)
            for p in pos:
                j = p % len(r)
                f.write(json.dumps({'family': fam, 'id': spec.id, 'pos': p, 'idx': j, 'token': r.decoded[j],
                                    'ids': list(r.ids)}, ensure_ascii=False) + '\n')
                n += 1
print(n, 'cells')
