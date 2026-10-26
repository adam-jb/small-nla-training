"""Collect every 27B->small student run (metrics.jsonl, done.json, config.json) into one table (11 Oct 2026).
Usage: python scripts/nladistil27b_collect_results_20261011.py runs/nladistil27b_results_20261011
Expects <dir>/<run>/{metrics.jsonl,config.json,done.json?} (rsynced from the boxes). Writes summary.md + ledger rows."""
import json, sys
from pathlib import Path

d = Path(sys.argv[1]); TF = 0.7545
rows, curves = [], {}
for r in sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith('nld27')):
    m = [json.loads(l) for l in open(r / 'metrics.jsonl')] if (r / 'metrics.jsonl').exists() else []
    cfg = json.load(open(r / 'config.json')) if (r / 'config.json').exists() else {}
    done = json.load(open(r / 'done.json')) if (r / 'done.json').exists() else None
    stop = json.load(open(r / 'STOPPED.json')) if (r / 'STOPPED.json').exists() else None
    ev = {e['eval']: e for e in m if 'eval' in e}
    qk = {e['quick']: e['fve'] for e in m if 'quick' in e}
    curves[r.name] = {'full': {k: round(v['heldout']['fve'], 4) for k, v in ev.items() if 'heldout' in v}, 'quick': qk}
    last = ev[max(ev)] if ev else {}
    b = [e for e in m if e.get('stage') == 'B']
    rows.append({'run': r.name, 'items': next((e['items'] for e in m if e.get('data')), None),
                 'student': cfg.get('student'), 'mode': cfg.get('mode'), 'lr': cfg.get('lr_lm'), 'epochs': cfg.get('epochs'),
                 'final_seen': max(ev) if ev else None, 'fve': last.get('heldout', {}).get('fve'),
                 'ci': last.get('heldout', {}).get('ci95'), 'share': (last.get('heldout', {}).get('fve') or 0) / TF,
                 'parsed': last.get('heldout', {}).get('parsed'), 'teval': last.get('teval', {}).get('fve'),
                 'inject': last.get('inject'), 'notes': last.get('notes'),
                 'ce_end': b[-1]['ce'] if b else None, 'soft_end': b[-1]['soft'] if b else None,
                 'tok_s': sorted(e['tok_s'] for e in b)[len(b) // 2] if b else None,
                 'hours': round(m[-1]['sec'] / 3600, 2) if m else None, 'done': bool(done), 'stopped': stop})
json.dump({'rows': rows, 'curves': curves}, open(d / 'summary.json', 'w'), indent=1)
f = lambda x, n=3: '-' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))
out = ['| run | items | seen | held-out FVE [95% CI] | share of teacher | teacher-eval FVE | inject (α→hit) | notes city/job | CE/soft end | h | status |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
for x in rows:
    ci = f"[{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]" if x['ci'] else ''
    inj = ' '.join(f"{k}:{v:.2f}" for k, v in (x['inject'] or {}).items())
    nt = f"{x['notes']['city']:.2f}/{x['notes']['job']:.2f}" if x['notes'] else '-'
    st = 'done' if x['done'] else ('STOPPED ' + x['stopped'].get('why', '') if x['stopped'] else 'running/cut')
    out.append(f"| {x['run'].replace('nld27_', '').replace('_from-nla-qwen36-27b-L42-rl400', '')} | {x['items']} | {x['final_seen']} | "
               f"{f(x['fve'])} {ci} | {f(x['share'], 2)} | {f(x['teval'])} | {inj or '-'} | {nt} | {f(x['ce_end'], 2)}/{f(x['soft_end'], 2)} | {x['hours']} | {st} |")
out += ['', '## Held-out FVE by examples seen (full 1,020 set)', '']
for k, c in curves.items():
    out.append(f"- {k.replace('nld27_', '')}: " + ', '.join(f'{s // 1000}k {v:.3f}' for s, v in sorted(c['full'].items())))
out += ['', '## Quick FVE (256-vector subset) by examples seen', '']
for k, c in curves.items():
    out.append(f"- {k.replace('nld27_', '')}: " + ', '.join(f'{s // 1000}k {v:.3f}' for s, v in sorted(c['quick'].items())))
(d / 'summary.md').write_text('\n'.join(out) + '\n'); print('\n'.join(out))
