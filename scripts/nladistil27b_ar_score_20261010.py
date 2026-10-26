"""Score explanation files with the 27B teacher's AR (10 Oct 2026).
Each jsonl row needs 'i' (row in the set) and 'expl'. Writes *_ar.json: FVE with a doc-bootstrap CI, shuffled control,
parse rate, mean length; and *_scored.jsonl with per-row cos. rawvar comes from the first 50k train vectors (cached).
Usage: python scripts/nladistil27b_ar_score_20261010.py --vec DIR --set heldout f1.jsonl [f2.jsonl ...]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import AR, LAYER, fve_stats, rawvar_baseline, rl
from scripts.nladistil27b_extract_20261010 import load_set


def rawvar_for(vec):
    p = Path(vec) / 'rawvar.json'
    if not p.exists():
        V = load_set(vec, 'train')
        if V is None:  # before the train set exists (pre-flight test): estimate from the teacher-eval set, not cached
            return rawvar_baseline(torch.tensor(load_set(vec, 'teval')))
        V = V[:50000]
        json.dump({'rawvar': rawvar_baseline(torch.tensor(V)), 'n': len(V)}, open(p, 'w'))
    return json.load(open(p))['rawvar']


def score_files(ar, vec, set_name, files, rawvar):
    V = torch.tensor(load_set(vec, set_name, LAYER))
    srows = rl(Path(vec) / set_name / 'rows.jsonl')
    out = []
    for f in files:
        rows = rl(f)
        P = ar.reconstruct([r['expl'] for r in rows])
        G = V[[r['i'] for r in rows]]
        summ, c = fve_stats(P, G, rawvar, groups=[srows[r['i']].get('doc_id', r['i']) for r in rows])
        for r, x in zip(rows, c.tolist()):
            r['ar_cos'] = x
        with open(str(f).replace('.jsonl', '_scored.jsonl'), 'w') as fo:
            for r in rows:
                fo.write(json.dumps(r) + '\n')
        summ.update({'file': str(f), 'set': set_name, 'rawvar': rawvar,
                     'parsed': float(np.mean([r.get('parsed', True) for r in rows])),
                     'mean_tok': float(np.mean([r.get('n_tok', 0) for r in rows]))})
        json.dump(summ, open(str(f).replace('.jsonl', '_ar.json'), 'w'), indent=1)
        print(json.dumps(summ), flush=True)
        out.append(summ)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--set', default='heldout')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    score_files(AR(dev), a.vec, a.set, a.files, rawvar_for(a.vec))


if __name__ == '__main__':
    main()
