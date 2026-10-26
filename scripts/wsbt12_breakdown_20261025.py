"""Per-family and per-subtype breakdown of WorkspaceBench tiers 1+2 (25 Oct 2026): full bank vs the bench's
anti-text-inversion subset (items Opus 5.5 cannot answer from the prompt text alone), and the distillation
retention ratio student / teacher. Subtype = item-id prefix up to the second '-' (e.g. 'br-ent', 'couplet').
Usage: python3 scripts/wsbt12_breakdown_20261025.py DATA_DIR BENCH_REPO OUT.json"""
import json, sys
from collections import defaultdict
from pathlib import Path

D, BR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ati = json.load(open(BR / 'evals/baselines/anti_text_inversion.json'))
TIER1 = ['association', 'basic_readout', 'multihop', 'multilingual', 'poetry', 'typo', 'wrong_bridge', 'wrong_bridge_true']
TIER2 = ['basic_readout_mt', 'multihop_mt_v2', 'multilingual_mt', 'multilingual_multihop', 'multilingual_typo', 'typo_mt']
R = ['prompt_only', 'teacher', 'grpo526', 'agi_r100']


def passes(root, fam):
    """{reader: {item: mean pass over samples}}"""
    out = {}
    for r in R:
        acc = defaultdict(list)
        for p in sorted((root / r).glob(f's*/{fam}/results.json')):
            for row in json.load(open(p))['rows']:
                acc[row['id']].append(int(bool(row['pass'])))
        out[r] = {i: sum(v) / len(v) for i, v in acc.items()}
    return out


def rate(m, ids):
    ids = [i for i in ids if i in m]
    return round(sum(m[i] for i in ids) / len(ids), 3) if ids else None


def subtype(i):
    return '-'.join(i.split('-')[:2]) if i.count('-') >= 2 else i.split('-')[0]


res = {}
for fam in TIER1 + TIER2:
    P = passes(D / ('judged_gemini' if fam in TIER1 else 'judged'), fam)
    ids = sorted(P['teacher'])
    keep = [i for i in (ati.get(fam) or {}).get('ids', []) if i in P['teacher']]
    e = {'n': len(ids), 'n_ati': len(keep),
         'full': {r: rate(P[r], ids) for r in R}, 'ati': {r: rate(P[r], keep) for r in R} if keep else None}
    groups = defaultdict(list)
    for i in ids:
        groups[subtype(i)].append(i)
    e['subtypes'] = {g: {'n': len(v), **{r: rate(P[r], v) for r in R}} for g, v in sorted(groups.items()) if len(v) >= 8}
    res[fam] = e
json.dump(res, open(OUT, 'w'), indent=1)


def ret(e):
    t = e['teacher']
    return f"{(e['agi_r100'] / t):.2f}" if t else '-'


print(f"{'family':22s} {'n':>4s} | {'prompt':>6s} {'teach':>6s} {'g526':>6s} {'agi':>6s} {'keep':>5s} | ATI n  prompt  teach   agi")
for fam, e in res.items():
    f, a = e['full'], e['ati']
    s = f"{fam:22s} {e['n']:4d} | {f['prompt_only']:6.2f} {f['teacher']:6.2f} {f['grpo526']:6.2f} {f['agi_r100']:6.2f} {ret(f):>5s} |"
    if a:
        s += f" {e['n_ati']:4d} {a['prompt_only']:6.2f} {a['teacher']:6.2f} {a['agi_r100']:6.2f}"
    print(s)
print('\nsubtypes (n>=8): family/subtype n prompt teacher agi keep')
for fam, e in res.items():
    if len(e['subtypes']) > 1:
        for g, v in e['subtypes'].items():
            print(f"  {fam:20s} {g:18s} {v['n']:3d} {v['prompt_only']:5.2f} {v['teacher']:5.2f} {v['agi_r100']:5.2f} {ret(v):>5s}")
