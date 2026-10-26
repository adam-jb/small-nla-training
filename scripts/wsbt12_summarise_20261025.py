"""Summarise WorkspaceBench tiers 1+2 (25 Oct 2026): per family x reader, mean pass rate over the K single-sample runs
(each run = the bench's k=1 protocol at layer 42), the spread across runs, a paired item-bootstrap CI of reader
differences (items x samples), and the bench's prompt-only floor. Usage: python3 scripts/wsbt12_summarise_20261025.py
JUDGED_DIR BENCH_REPO OUT.json"""
import json, random, sys
from pathlib import Path

J, BR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
floor = json.load(open(BR / 'evals/baselines/prompt_only.json'))
PUB = {'basic_readout_mt': 0.53, 'multilingual_mt': 0.44, 'multilingual_multihop': 0.34}  # nla-rl-L42, family READMEs
TIER1 = ['association', 'basic_readout', 'multihop', 'multilingual', 'poetry', 'typo', 'wrong_bridge', 'wrong_bridge_true']
readers = sorted(p.name for p in J.iterdir() if p.is_dir())
fams = sorted({f.parent.name for f in J.glob('*/s*/*/results.json')}, key=lambda f: (f not in TIER1, f))

# passes[reader][family][item] = list of 0/1 over samples
P, spend = {}, 0.0
for r in readers:
    for sd in sorted((J / r).glob('s*')):
        for f in fams:
            p = sd / f / 'results.json'
            if not p.exists():
                continue
            d = json.load(open(p)); spend += d['counts'].get('spend_usd', 0) or 0
            for row in d['rows']:
                P.setdefault(r, {}).setdefault(f, {}).setdefault(row['id'], []).append(int(bool(row['pass'])))


def mean_rate(it):
    return sum(sum(v) / len(v) for v in it.values()) / len(it)


def boot_diff(a, b, n=2000, seed=0):
    ids = sorted(set(a) & set(b)); rng = random.Random(seed); ds = []
    da = {i: sum(a[i]) / len(a[i]) - sum(b[i]) / len(b[i]) for i in ids}
    for _ in range(n):
        s = [da[rng.choice(ids)] for _ in ids]
        ds.append(sum(s) / len(s))
    ds.sort()
    return round(sum(da.values()) / len(ids), 3), [round(ds[int(.025 * n)], 3), round(ds[int(.975 * n)], 3)]


res = {'spend_usd': round(spend, 2), 'families': {}}
for f in fams:
    e = {'tier': 1 if f in TIER1 else 2, 'prompt_only_floor': (floor.get(f) or {}).get('rate'), 'published_teacher': PUB.get(f)}
    for r in readers:
        it = P.get(r, {}).get(f)
        if not it:
            continue
        k = len(next(iter(it.values())))
        per = [sum(v[s] for v in it.values()) / len(it) for s in range(k)]
        e[r] = {'mean': round(mean_rate(it), 3), 'per_sample': [round(x, 2) for x in per], 'any_of_k': round(sum(max(v) for v in it.values()) / len(it), 3)}
    for a in readers:
        if a != 'teacher' and 'teacher' in P and f in P.get(a, {}):
            e[f'{a}_minus_teacher'] = boot_diff(P[a][f], P['teacher'][f])
    if 'agi_r100' in P and 'grpo526' in P and f in P['agi_r100']:
        e['agi_r100_minus_grpo526'] = boot_diff(P['agi_r100'][f], P['grpo526'][f])
    res['families'][f] = e
for t in (1, 2):
    fs = [f for f in fams if res['families'][f]['tier'] == t]
    if not fs:
        continue
    res[f'tier{t}_mean'] = {r: round(sum(res['families'][f][r]['mean'] for f in fs) / len(fs), 3) for r in readers
                            if all(r in res['families'][f] for f in fs)}
json.dump(res, open(OUT, 'w'), indent=1)
print(f"{'family':24s} {'floor':>6s} " + ' '.join(f'{r:>9s}' for r in readers) + '   ag-T     ag-g526')
for f in fams:
    e = res['families'][f]
    fl = e['prompt_only_floor']
    print(f"{f:24s} {fl if fl is not None else '-':>6} " + ' '.join(f"{e[r]['mean'] if r in e else '-':>9}" for r in readers),
          e.get('agi_r100_minus_teacher', ''), e.get('agi_r100_minus_grpo526', ''))
print('tier means', res.get('tier1_mean'), res.get('tier2_mean'), 'spend $', res['spend_usd'])
