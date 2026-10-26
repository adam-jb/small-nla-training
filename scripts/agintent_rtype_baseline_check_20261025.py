"""Check (25 Oct 2026): is decision-point (P_last) FVE < 0 a yardstick artefact?
FVE = 1 - mean(2(1-cos(AR(note), v))) / rawvar. Final eval used one agent rawvar (all agent row types).
Here: rawvar per row type (spread of that type's unit vectors around that type's own mean), and per type+source,
then re-score the saved notes' cos with each yardstick.
Usage: .venv/bin/python scripts/agintent_rtype_baseline_check_20261025.py VEC_DIR RESULTS_DIR"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_extract_20261010 import load_set

VEC, RES = Path(sys.argv[1]), Path(sys.argv[2])


def tagged(name):
    rows = [json.loads(l) for l in open(VEC / name / 'rows.jsonl')]
    V = load_set(VEC, name)
    last = {}
    for r in rows:
        if r['type'] == 'P':
            last[r['unit_id']] = max(last.get(r['unit_id'], -1), r['pos'])
    rt = [r['type'] if r['type'] not in ('P',) else ('P_last' if r['pos'] == last[r['unit_id']] else 'P_mid') for r in rows]
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    return rows, rt, U


def rawvar(U):
    return float(((U - U.mean(0)) ** 2).sum(1).mean())


out = {}
for name in ['train_r100', 'agent_hold']:
    rows, rt, U = tagged(name)
    ag = np.array([r['source'] != 'web' for r in rows])
    rt = np.array(rt); res = {'all_agent': rawvar(U[ag]), 'n_agent': int(ag.sum())}
    gm = U[ag].mean(0)
    for t in ['G', 'P_mid', 'P_last']:
        m = ag & (rt == t)
        res[t] = {'n': int(m.sum()), 'rawvar_own_mean': rawvar(U[m]),
                  'rawvar_vs_all_agent_mean': float(((U[m] - gm) ** 2).sum(1).mean()),
                  'norm_of_type_mean': float(np.linalg.norm(U[m].mean(0)))}
    # per type + source (lazy guess that also knows the source)
    srcs = np.array([r['source'] for r in rows])
    for t in ['P_last']:
        tot, n = 0.0, 0
        for s in sorted(set(srcs[ag])):
            m = ag & (rt == t) & (srcs == s)
            if m.sum() > 1:
                tot += ((U[m] - U[m].mean(0)) ** 2).sum(1).sum(); n += m.sum()
        res[t]['rawvar_own_mean_by_source'] = float(tot / n)
    out[name] = res
print(json.dumps(out, indent=1))

hold = out['agent_hold']
yard = {'agent_rawvar_used': json.load(open(VEC / 'rawvar_agent.json'))['rawvar']}
for arm in sorted(RES.glob('agi_*')):
    notes = [json.loads(l) for l in open(arm / 'final_eval' / 'notes_hold.jsonl')]
    by = defaultdict(lambda: {'start': [], 'arm': []})
    for r in notes:
        for w in ['start', 'arm']:
            by[r['rtype']][w].append(2 * (1 - r[w]['cos']))
            if r['rtype'] == 'P_last' and r['slice'] == 'A':
                by['P_last_A'][w].append(2 * (1 - r[w]['cos']))
    print('\n==', arm.name)
    for t in ['G', 'P_mid', 'P_last', 'P_last_A']:
        b = by[t]; tt = 'P_last' if t == 'P_last_A' else t
        for lab, rv in [('agent rawvar (used)', yard['agent_rawvar_used']), ('own-type rawvar, hold', hold[tt]['rawvar_own_mean']),
                        ('own-type+source rawvar, hold', hold[tt].get('rawvar_own_mean_by_source'))]:
            if rv:
                print(f"  {t:9s} n={len(b['arm']):4d} {lab:30s} rv={rv:.3f}  FVE start {1-np.mean(b['start'])/rv:+.3f} -> arm {1-np.mean(b['arm'])/rv:+.3f}"
                      f"   (mean cos start {1-np.mean(b['start'])/2:.3f} arm {1-np.mean(b['arm'])/2:.3f})")
