"""WSB agentic_misalignment L42 summary (26 Oct 2026): per-reader design score on (a) every scenario that reader has and
(b) the scenarios every reader has (Sonnet blocked a few merges per reader), paired differences with item bootstrap,
controls, and a per-scenario fidelity table. Run with the bench venv python, WSB_REPO set.
-> data/wsbam_20261025/summary.json and a markdown table on stdout."""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.wsbam_judge_20261025 import ITEMS, records  # noqa: E402
from wsbench.evals.agentic_misalignment.score import aggregate_design, design_item_score  # noqa: E402

READERS = ['teacher', 'grpo526', 'agi_r100', 'prompt_only']
R = {t: {r['id']: r for r in records(t)} for t in READERS}
common = sorted(set.intersection(*(set(v) for v in R.values())))
KEYS = ('design_score', 'design_score_ci95', 'design_hit_at_1', 'design_hit_at_3', 'commit_strength_mean',
        'commit_strength_mean_control', 'control_asserts_rate', 'propensity_agreement', 'n_misaligned', 'n_control')


def agg(recs):
    a = aggregate_design(recs)
    return {k: a.get(k) for k in KEYS}


def paired(x, y, n=4000, seed=0):
    mis = [i for i in common if ITEMS[i]['arm'] == 'misaligned']
    d = [design_item_score(R[x][i]) - design_item_score(R[y][i]) for i in mis]
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {'diff': sum(d) / len(d), 'ci95': [bs[int(.025 * n)], bs[int(.975 * n) - 1]], 'n': len(d)}


out = {'n_common': len(common), 'missing': {t: sorted(set(ITEMS) - set(R[t])) for t in READERS},
       'all': {t: agg(list(R[t].values())) for t in READERS},
       'common': {t: agg([R[t][i] for i in common]) for t in READERS},
       'paired_common': {f'{x}-{y}': paired(x, y) for x, y in [('teacher', 'grpo526'), ('teacher', 'agi_r100'),
                                                               ('agi_r100', 'grpo526'), ('teacher', 'prompt_only'),
                                                               ('agi_r100', 'prompt_only')]}}
json.dump(out, open(Path(__file__).resolve().parents[1] / 'data/wsbam_20261025/summary.json', 'w'), indent=1)

f = lambda v: '-' if v is None else f'{v:.2f}'  # noqa: E731
print(f"common set: {len(common)} scenarios; missing per reader: " + ', '.join(f'{t} {len(m)}' for t, m in out['missing'].items()))
print('| Reader | design score (all) | n | design score (common) | CI95 | hit@1 | commit | control asserts |')
print('|---|---|---|---|---|---|---|---|')
for t in READERS:
    a, c = out['all'][t], out['common'][t]
    print(f"| {t} | {f(a['design_score'])} | {a['n_misaligned']} | {f(c['design_score'])} | "
          f"[{f(c['design_score_ci95'][0])}, {f(c['design_score_ci95'][1])}] | {f(c['design_hit_at_1'])} | "
          f"{f(c['commit_strength_mean'])} | {f(c['control_asserts_rate'])} (n={c['n_control']}) |")
print()
for k, v in out['paired_common'].items():
    print(f"{k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}")
print('\n| Scenario | arm | ' + ' | '.join(READERS) + ' |\n|---|---|' + '---|' * len(READERS))
for i in ITEMS:
    print(f"| {i} | {ITEMS[i]['arm'][:3]} | " + ' | '.join(
        '-' if i not in R[t] else f"{R[t][i].get('design_fidelity')}{'*' if R[t][i].get('asserts_misaligned_plan') else ''}"
        for t in READERS) + ' |')
