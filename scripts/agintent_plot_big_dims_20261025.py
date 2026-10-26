"""Chart (25 Oct 2026): Qwen3.6-27B L42 dimension 3994 vs the next biggest dimensions: size (box plot of |value|, log scale)
and how much of the across-token variance each dimension carries (unit-length vectors, as in the FVE/cosine scores).
Data: train_r100 (web + agent rows) and the tokens-before set (template tokens). Usage: .venv/bin/python scripts/agintent_plot_big_dims_20261025.py"""
import json, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from scripts.nladistil27b_extract_20261010 import from_bf16_u16

INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e0'
R = [json.loads(l) for l in open('data/agintent_vec_check_20261025/train_r100/rows.jsonl')]
V = from_bf16_u16(np.load('data/agintent_vec_check_20261025/train_r100/L42_000.npy')).numpy()
web = np.array([r['source'] == 'web' for r in R])
U = V / np.linalg.norm(V, axis=1, keepdims=True)
order = np.argsort(-np.abs(V).mean(0))
top = list(order[:10]); med = order[len(order) // 2]
var = U.var(0); vshare = var / var.sum()
print('mean |v| top10', [(int(d), round(float(np.abs(V[:, d]).mean()), 1)) for d in top], 'median dim', int(med), round(float(np.abs(V[:, med]).mean()), 2))
print('variance share (unit vectors) top10', [(int(d), round(float(vshare[d]), 4)) for d in top], 'rank of 3994 by variance', int((vshare > vshare[3994]).sum()) + 1)
print('mean share of squared length', [(int(d), round(float((U[:, d] ** 2).mean()), 3)) for d in top[:3]])
cv = V[:, 3994].std() / np.abs(V[:, 3994]).mean(); print('dim 3994 std/mean', round(float(cv), 3), 'typical dim std/mean', round(float(np.median(V.std(0) / (np.abs(V).mean(0) + 1e-6))), 2))
# template-token set
T = [json.loads(l) for l in open('data/agintent_tokens_before_20261025/rows.jsonl')]
VT = from_bf16_u16(np.load('data/agintent_tokens_before_20261025/L42_000.npy')).numpy()
think = np.array([t['off'] == 2 for t in T])

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURF, gridspec_kw={'width_ratios': [1.45, 1]})
a = ax[0]
labels = [f'{d}' for d in top] + [f'typical\n({med})']
data_web = [np.abs(V[web, d]) for d in top] + [np.abs(V[web, med])]
data_ag = [np.abs(V[~web, d]) for d in top] + [np.abs(V[~web, med])]
x = np.arange(len(labels))
for off_, data, c, lab in ((-0.2, data_web, '#86b6ef', 'web text'), (0.2, data_ag, '#2a78d6', 'agent text')):
    bp = a.boxplot(data, positions=x + off_, widths=0.34, patch_artist=True, showfliers=False, whis=(5, 95),
                   medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for b in bp['boxes']:
        b.set(facecolor=c, edgecolor=INK2, lw=0.8)
    a.plot([], [], 's', color=c, label=lab)
a.scatter(np.zeros(think.sum()) + 0.45, np.abs(VT[think, 3994]) + 1e-2, s=10, color='#eb6834', zorder=4, label='dim 3994 at <think> tokens')
a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(labels, fontsize=8.5)
a.set_ylabel('|value| (log scale)', color=INK2); a.set_xlabel('dimension (10 biggest by mean |value|, then a typical one)', color=INK2)
a.set_title('Size: dimension 3994 vs the next biggest (box 25-75%, whiskers 5-95%)', loc='left', fontsize=11, color=INK)
a.legend(frameon=False, fontsize=8.5, loc='upper right')
b_ = ax[1]
vs = [vshare[d] * 100 for d in top]
b_.bar(np.arange(10), vs, color=['#eb6834'] + ['#2a78d6'] * 9, width=0.6)
b_.axhline(100 / V.shape[1], color=INK2, ls=':', lw=1); b_.text(9.4, 100 / V.shape[1] * 1.15, 'average dimension', ha='right', fontsize=8, color=INK2)
b_.set_yscale('log'); b_.set_xticks(np.arange(10)); b_.set_xticklabels([str(d) for d in top], fontsize=8.5)
b_.set_ylabel('% of across-token variance (log scale)', color=INK2); b_.set_xlabel('dimension', color=INK2)
b_.set_title('Information: share of the variance each one carries', loc='left', fontsize=11, color=INK)
for a in ax:
    a.set_facecolor(SURF); a.grid(axis='y', color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('docs/agintent_qwen27b_L42_dim3994_size_vs_variance_2026-10-25.png', dpi=140, facecolor=SURF)
