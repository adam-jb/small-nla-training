"""Analyse the tokens-before run (25 Oct 2026): rebuild cosine / FVE by offset from the end of the prompt (1 = decision point),
split short (< 256 tokens) vs long prompts, for teacher 27B, GRPO526, AGI-r25; plus a "lazy guess" per offset
(leave-one-out mean of the other units' vectors at that offset) and a chart.
Usage: .venv/bin/python scripts/agintent_tokens_before_analyse_20261025.py DIR   (DIR holds rows.jsonl, L42_000.npy,
       notes_GRPO526.jsonl, notes_AGI-r25.jsonl, teacher_scored.jsonl)"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_extract_20261010 import from_bf16_u16

D = Path(sys.argv[1]); RV = 0.426
rows = [json.loads(l) for l in open(D / 'rows.jsonl')]
V = from_bf16_u16(np.load(D / 'L42_000.npy')).float().numpy(); U = V / np.linalg.norm(V, axis=1, keepdims=True)
cos = {'Teacher 27B': {r['i']: r['ar_cos'] for r in map(json.loads, open(D / 'teacher_scored.jsonl'))}}
for m in ['GRPO526', 'AGI-r25']:
    cos[m] = {i: json.loads(l)['cos'] for i, l in enumerate(open(D / f'notes_{m}.jsonl'))}
off = np.array([r['off'] for r in rows]); long_ = np.array([r['n_prompt'] >= 256 for r in rows])
# lazy guess: mean of the OTHER units' vectors at the same offset and same length group
lazy = np.zeros(len(rows))
for o in set(off):
    for lg in (True, False):
        m = np.where((off == o) & (long_ == lg))[0]
        S = U[m].sum(0)
        for j in m:
            g = (S - U[j]) / max(len(m) - 1, 1); lazy[j] = U[j] @ g / np.linalg.norm(g)
cos['lazy guess (same offset)'] = dict(enumerate(lazy))
fve = lambda c: 1 - np.mean(2 * (1 - np.array(c))) / RV
offs = sorted(set(off))
tab = defaultdict(dict)
for lg, name in ((False, 'short'), (True, 'long')):
    for o in offs:
        ix = [j for j in range(len(rows)) if off[j] == o and long_[j] == lg]
        toks = sorted({rows[j]['tok'] for j in ix}, key=lambda t: -sum(rows[j]['tok'] == t for j in ix))[:2]
        tab[(name, o)] = {'n': len(ix), 'tok': toks, **{m: (float(np.mean([c[j] for j in ix])), float(fve([c[j] for j in ix]))) for m, c in cos.items()}}
for (name, o), t in tab.items():
    print(f"{name:5s} off {o:2d} n={t['n']:3d} tok {str(t['tok'])[:28]:28s} " +
          ' | '.join(f"{m[:8]} {t[m][0]:.3f} ({t[m][1]:+.2f})" for m in cos))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor='#fcfcfb', sharey=True)
col = {'Teacher 27B': '#1baf7a', 'GRPO526': '#eb6834', 'AGI-r25': '#2a78d6', 'lazy guess (same offset)': '#52514e'}
for a, name in zip(ax, ('short', 'long')):
    a.set_facecolor('#fcfcfb'); a.grid(axis='y', color='#e6e5e0'); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for m, c in col.items():
        a.plot(range(len(offs)), [tab[(name, o)][m][0] for o in offs], marker='o', color=c, lw=2, ms=6, label=m, ls='--' if 'lazy' in m else '-', mec='#fcfcfb')
    a.axhline(0.787, color='#52514e', lw=0.8, ls=':'); a.invert_xaxis()
    names = {1: '1\n\\n', 2: '2\n<think>', 3: '3\n\\n', 4: '4\nasst', 5: '5\nim_start', 6: '6\n\\n', 7: '7\nim_end', 8: '8\nend text'}
    a.set_xticks(range(len(offs))); a.set_xticklabels([names.get(o, str(o)) for o in offs], fontsize=8)
    n = tab[(name, offs[0])]['n']
    a.set_title(f"{'Short prompts (< 256 tokens)' if name == 'short' else 'Long prompts (>= 256 tokens)'}, {n} prompts", loc='left', fontsize=11)
    a.set_xlabel('tokens back from the end of the prompt (1 = decision point)', color='#52514e')
ax[0].set_ylabel('mean cosine: rebuilt vs true vector', color='#52514e'); ax[0].legend(frameon=False, fontsize=8.5, loc='lower left')
plt.tight_layout(); plt.savefig('docs/agintent_tokens_before_decision_point_cos_by_offset_2026-10-25.png', dpi=140, facecolor='#fcfcfb')
json.dump({f'{k[0]}_{k[1]}': v for k, v in tab.items()}, open(D / 'by_offset.json', 'w'), indent=1)
