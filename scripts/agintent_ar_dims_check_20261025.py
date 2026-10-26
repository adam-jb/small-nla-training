"""Big-dimension check (25 Oct 2026). One L42 dimension (3994) holds ~57% of a normal vector's squared length; at <think>
(and at 'assistant' / the final '\\n' in long prompts) it is small, and rebuild cosine tracks this (corr 0.91 over 1,198 rows).
Re-run the 27B AR on saved notes, keep the rebuilt vectors, and score cosine (a) in full, (b) with dim 3994 dropped from both,
(c) with the top-5 big dims dropped. If (b)/(c) stay high at the decision point, the notes carry the content and the low
full cosine is the big dimension.
Usage (box): python scripts/agintent_ar_dims_check_20261025.py VEC SET OUT NAME=notes.jsonl [NAME=notes.jsonl ...]
  (each notes file: one row per set row in order, or rows with 'i'; text in 'expl')"""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import AR, LAYER, rl
from scripts.nladistil27b_extract_20261010 import load_set

BIG = [3994, 310, 253, 4110, 3842]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main():
    vec, set_, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    V = torch.tensor(load_set(vec, set_, LAYER)).float(); ar = AR('cuda'); res = {}
    keep1 = torch.ones(V.shape[1], dtype=torch.bool); keep1[BIG[0]] = False
    keep5 = torch.ones(V.shape[1], dtype=torch.bool); keep5[BIG] = False
    for spec in sys.argv[4:]:
        name, f = spec.split('=', 1)
        rows = rl(f); idx = [r.get('i', j) for j, r in enumerate(rows)]
        R = ar.reconstruct([r['expl'] for r in rows]).float().cpu(); G = V[idx]
        res[name] = {'i': idx, 'full': cos(R, G).tolist(), 'no3994': cos(R[:, keep1], G[:, keep1]).tolist(),
                     'no_top5': cos(R[:, keep5], G[:, keep5]).tolist(),
                     'R_big': R[:, BIG[0]].div(R.norm(dim=1)).tolist(), 'G_big': G[:, BIG[0]].div(G.norm(dim=1)).tolist()}
        print(name, {k: round(float(np.mean(v)), 4) for k, v in res[name].items() if k != 'i'}, flush=True)
    json.dump(res, open(out, 'w'))


if __name__ == '__main__':
    main()
