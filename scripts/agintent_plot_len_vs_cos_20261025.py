"""Chart (25 Oct 2026): decision-point rebuild cosine vs prompt length (GRPO526, AGI-r25, 27B teacher) and AGI-r25 by row type.
Inputs: final-eval notes (runs/agintent_results_20261025), teacher check (data/agintent_teacher_check_20261025), hold rows (pick_rows).
Usage: .venv/bin/python scripts/agintent_plot_len_vs_cos_20261025.py"""
import json, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from scripts.agintent_final_eval_20261024 import pick_rows

V = 'data/agintent_vec_check_20261025'; R = 'runs/agintent_results_20261025'; D = 'data/agintent_teacher_check_20261025'
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e0'
hrows = pick_rows(V, 400)
L = {r['unit_id']: r['pos'] + 1 for r in hrows if r['rtype'] == 'P_last'}
N = [json.loads(l) for l in open(f'{R}/agi_grpo_qwen35-2b_from-grpo526_ar_agentmix-r25-rows6k_20261024/final_eval/notes_hold.jsonl')]
assert all(n['unit_id'] == h['unit_id'] for n, h in zip(N, hrows))
T = [json.loads(l) for l in open(f'{D}/teacher_agent_s150_scored.jsonl')]; join = json.load(open(f'{D}/agent_s150_join.json'))
edges = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def binned(lens, cos):
    lens, cos = np.array(lens), np.array(cos); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lens >= a) & (lens < b)
        if m.sum() >= 5:
            out.append((np.sqrt(a * b), cos[m].mean(), m.sum()))
    return np.array(out)


P = [(L[n['unit_id']], n) for n in N if n['rtype'] == 'P_last']
ser = {'GRPO526 (start)': ('#eb6834', binned([l for l, _ in P], [n['start']['cos'] for _, n in P])),
       'AGI-r25 (trained)': ('#2a78d6', binned([l for l, _ in P], [n['arm']['cos'] for _, n in P]))}
tp = [(L[N[j]['unit_id']], T[i]['ar_cos']) for i, j in enumerate(join) if N[j]['rtype'] == 'P_last']
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF); a.set_xscale('log'); a.grid(axis='y', color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ['top', 'right']:
        a.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    a.axvspan(200, 600, color='#f0efec', zorder=0); a.text(346, 0.985, 'almost no prompts\nthis long', ha='center', va='top', fontsize=8, color=INK2)
    a.axhline(0.787, color=INK2, lw=1, ls='--'); a.text(7800, 0.775, 'FVE = 0: as good as guessing the average agent vector', ha='right', va='top', fontsize=8, color=INK2)
    a.set_xlabel('prompt length (tokens, log scale)', color=INK2); a.set_ylim(0.38, 1.0); a.set_xlim(16, 8192)
    a.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096]); a.set_xticklabels(['32', '64', '128', '256', '512', '1k', '2k', '4k'])
ax[0].axhline(0.85, color=INK2, lw=1, ls=':'); ax[0].text(7800, 0.856, 'guessing the average decision-point vector', ha='right', fontsize=8, color=INK2)
for k, (c, s) in ser.items():
    ax[0].plot(s[:, 0], s[:, 1], '-o', color=c, lw=2, ms=7, label=k + ', n=345', mec=SURF, mew=1.5, zorder=3)
tl, tc = zip(*tp)
ax[0].scatter(tl, tc, s=20, color='#1baf7a', alpha=0.75, label='Teacher 27B, each row (n=50)', edgecolor='none', zorder=2)
for x, y, n in ser['AGI-r25 (trained)'][1]:
    ax[0].text(x, 0.395, f'n={int(n)}', ha='center', fontsize=7.5, color=INK2)
ax[0].set_title('Decision point (last prompt token)', fontsize=11.5, loc='left', color=INK)
ax[0].set_ylabel('mean cosine: vector rebuilt from note vs true vector', color=INK2)
ax[0].legend(frameon=False, fontsize=8.5, loc='lower left', bbox_to_anchor=(0, 0.07), labelcolor=INK)
for t, (c, lab) in {'G': ('#1baf7a', "agent's reply"), 'P_mid': ('#6250d6', 'mid-prompt'), 'P_last': ('#2a78d6', 'decision point')}.items():
    pr = [(L[n['unit_id']], n['arm']['cos']) for n in N if n['rtype'] == t and n['unit_id'] in L]
    b = binned(*zip(*pr)); ax[1].plot(b[:, 0], b[:, 1], '-o', color=c, lw=2, ms=7, label=lab, mec=SURF, mew=1.5)
ax[1].set_title('AGI-r25: every row type', fontsize=11.5, loc='left', color=INK)
ax[1].legend(frameon=False, fontsize=8.5, loc='lower left', bbox_to_anchor=(0, 0.07), labelcolor=INK)
plt.tight_layout()
plt.savefig('docs/agintent_decision_point_cos_vs_prompt_length_2026-10-25.png', dpi=140, facecolor=SURF)
