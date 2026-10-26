"""DeceptionBench (HF skyai798/DeceptionBench, CC-BY-SA-4.0, academic use only) -> agintent units (24 Oct 2026).
Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
WHOLE SOURCE IS HOLDOUT (meta.holdout_source = true). 150 scenarios x 6 single-turn prompts (L1/L2, self/other, pressure/reward).
We keep 400 (seed 0): one L1 and one L2 prompt per scenario, plus 100 more prompts at random. All slice A; group = scenario.
L3 (multi-turn) is not in the data: its later turns are written live by GPT during the benchmark, so it is skipped.
Usage: .venv/bin/python -I scripts/agintent_conv_deceptionbench_20261024.py RAW_DIR OUT_DIR
"""
import json, random, re, statistics, sys
from collections import Counter
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KEYS = ['L1-self', 'L1-other', 'L2-self-pressure', 'L2-self-reward', 'L2-other-pressure', 'L2-other-reward']
CAP = 400
FILTERS = {
    'blackmail_coercion': re.compile(r'blackmail|extort|coerc', re.I),
    'leaking': re.compile(r'\bleak(s|ed|ing)?\b', re.I),
    'lethal_harm': re.compile(r'\blet (him|her|them|the \w+) die|(?<!a )\bkill|lethal|fatal|life-threatening', re.I),
    'bio_chem': re.compile(r'bioweapon|biological weapon|chemical weapon|nerve agent|pathogen', re.I),
    'banned_name': re.compile(r'summitbridge|novastar', re.I),
}


def uid(s):
    return re.sub(r'[^a-z0-9_-]+', '-', s.lower()).strip('-')


rng = random.Random(0)
pool, drop, dropped_ex = [], Counter(), {}
for fp in sorted(RAW.glob('*.json')):
    for j, r in enumerate(json.load(open(fp))):
        g = f'deceptionbench-{fp.stem}-{j:02d}'
        ok = []
        for k in KEYS:
            p = (r.get(k) or '').strip()
            if not p:
                drop['empty'] += 1; continue
            why = next((w for w, rx in FILTERS.items() if rx.search(p)), None)
            if why:
                drop[why] += 1; dropped_ex.setdefault(why, p[:200]); continue
            ok.append((k, p))
        pool.append((g, fp.stem, r, ok))

chosen = []
rest = []
for g, dom, r, ok in pool:
    l1 = [x for x in ok if x[0].startswith('L1')]; l2 = [x for x in ok if x[0].startswith('L2')]
    pick = ([rng.choice(l1)] if l1 else []) + ([rng.choice(l2)] if l2 else [])
    chosen += [(g, dom, r, k, p) for k, p in pick]
    rest += [(g, dom, r, k, p) for k, p in ok if (k, p) not in pick]
rng.shuffle(rest)
chosen += rest[:max(0, CAP - len(chosen))]
chosen.sort(key=lambda x: (x[0], KEYS.index(x[3])))

units = [{'unit_id': uid(f'{g}-{k}'), 'source': 'deceptionbench', 'slice': 'A', 'twin_of': None, 'group': g,
          'kind': uid(r['dimension']), 'messages': [{'role': 'user', 'content': p}], 'tools': [], 'license': 'CC-BY-SA-4.0',
          'meta': {'holdout_source': True, 'domain': dom, 'topic': r['topic'], 'dimension': r['dimension'], 'variant': k,
                   'level': k.split('-')[0], 'role': k.split('-')[1], 'inducement': (k.split('-') + ['none'])[2],
                   'goal': r.get('goal'), 'groundtruth': r.get('groundtruth')}}
         for g, dom, r, k, p in chosen]

with open(OUT / 'deceptionbench.jsonl', 'w') as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + '\n')
ch = [len(u['messages'][0]['content']) for u in units]
rep = {'units': len(units), 'scenarios': len({u['group'] for u in units}), 'by_slice': dict(Counter(u['slice'] for u in units)),
       'by_variant': dict(Counter(u['meta']['variant'] for u in units)), 'by_domain': dict(Counter(u['meta']['domain'] for u in units)),
       'kinds': len({u['kind'] for u in units}), 'median_chars': statistics.median(ch), 'max_chars': max(ch), 'dropped_prompts': dict(drop)}
ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1, ensure_ascii=False)[:1500] + '\n```' for u in (units[0], units[-1]))
(OUT / 'deceptionbench.README.md').write_text(f"""# deceptionbench -> agintent units (24 Oct 2026) -- WHOLE SOURCE IS HOLDOUT

- Fetched: HF `skyai798/DeceptionBench` @ 38b412da565073483c0a68ce9e04eb15859beedc (5 domain json files, 30 scenarios each).
- Converter: `scripts/agintent_conv_deceptionbench_20261024.py`.
- Licence: CC-BY-SA-4.0; card adds "only used for academic research, commercial use prohibited". Do not publish units.
- Canary check (grep of all fetched files): no hits. PASS.
- Every unit has `meta.holdout_source = true` (never train on it). All slice A, one user turn, verbatim, no glue
  (each prompt already sets the role: "You are a marketing manager ..." / "You are my trusted copywriter ...").
- Pick: 1 L1 + 1 L2 prompt per scenario, then 100 more at random (seed 0), cap {CAP}. Group = scenario; kind = dimension.
- L3 multi-turn skipped: its follow-up turns are generated live by GPT, they are not in the data.
- Prompts dropped by keyword hard filters: {dict(drop)}
  e.g. {json.dumps(dropped_ex, ensure_ascii=False)}

## Counts
```
{json.dumps(rep, indent=1)}
```

## Examples
{ex}
""")
print(json.dumps(rep))
