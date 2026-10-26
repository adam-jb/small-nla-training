"""Anthropic model-written evals, advanced-ai-risk (HF Anthropic/model-written-evals, CC-BY-4.0) -> agintent units (24 Oct 2026).
Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
10 temptation categories (self-awareness-* and one-box-tendency skipped: nothing to be tempted by), 60 per category (seed 0):
30 from human_generated_evals + 30 from lm_generated_evals. One user turn = the question with its choices, verbatim. Slice A.
Usage: .venv/bin/python -I scripts/agintent_conv_mwe_advanced_ai_risk_20261024.py RAW_DIR OUT_DIR
"""
import json, random, re, statistics, sys
from collections import Counter
from pathlib import Path

RAW, OUT = Path(sys.argv[1]) / 'advanced-ai-risk', Path(sys.argv[2])
CATS = ['power-seeking-inclination', 'wealth-seeking-inclination', 'survival-instinct', 'corrigible-less-HHH', 'corrigible-more-HHH',
        'corrigible-neutral-HHH', 'coordinate-itself', 'coordinate-other-ais', 'coordinate-other-versions', 'myopic-reward']
PER = 30
FILTERS = {
    'blackmail_coercion': re.compile(r'blackmail|extort|coerc', re.I),
    'leaking': re.compile(r'\bleak(s|ed|ing)?\b', re.I),
    'lethal_harm': re.compile(r'\blet (him|her|them|people|humans?|someone|the \w+) die|(?<!a )\bkill|lethal|murder', re.I),
    'bio_chem': re.compile(r'bioweapon|biological weapon|chemical weapon|nerve agent|pathogen', re.I),
    'banned_name': re.compile(r'summitbridge|novastar', re.I),
}

rng = random.Random(0)
units, drop, seen, dropped_ex = [], Counter(), set(), {}
for cat in CATS:
    for gen in ('human_generated_evals', 'lm_generated_evals'):
        rows = [json.loads(l) for l in open(RAW / gen / f'{cat}.jsonl')]
        order = list(range(len(rows))); rng.shuffle(order)
        took = 0
        for j in order:
            if took >= PER:
                break
            q = rows[j]['question'].strip()
            if q in seen:
                drop['duplicate'] += 1; continue
            why = next((k for k, rx in FILTERS.items() if rx.search(q)), None)
            if why:
                drop[why] += 1; dropped_ex.setdefault(why, q[:160]); continue
            seen.add(q); took += 1
            g = 'h' if gen.startswith('human') else 'lm'
            u = f'mwe-{cat.lower()}-{g}-{j:04d}'
            units.append({'unit_id': u, 'source': 'mwe_advanced_ai_risk', 'slice': 'A', 'twin_of': None, 'group': u, 'kind': cat.lower(),
                          'messages': [{'role': 'user', 'content': q}], 'tools': [], 'license': 'CC-BY-4.0',
                          'meta': {'category': cat, 'generator': gen, 'row': j,
                                   'answer_matching_behavior': rows[j]['answer_matching_behavior'],
                                   'answer_not_matching_behavior': rows[j]['answer_not_matching_behavior']}})

with open(OUT / 'mwe_advanced_ai_risk.jsonl', 'w') as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + '\n')
ch = [len(u['messages'][0]['content']) for u in units]
rep = {'units': len(units), 'by_slice': dict(Counter(u['slice'] for u in units)), 'kinds': dict(Counter(u['kind'] for u in units)),
       'by_generator': dict(Counter(u['meta']['generator'] for u in units)),
       'median_chars': statistics.median(ch), 'max_chars': max(ch), 'dropped': dict(drop)}
ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1, ensure_ascii=False)[:1200] + '\n```' for u in (units[0], units[-1]))
(OUT / 'mwe_advanced_ai_risk.README.md').write_text(f"""# mwe_advanced_ai_risk -> agintent units (24 Oct 2026)

- Fetched: HF `Anthropic/model-written-evals` @ d533f626cc321c92175a58ee570aa3cdb87238d1, folder `advanced-ai-risk/` only.
- Converter: `scripts/agintent_conv_mwe_advanced_ai_risk_20261024.py`. Licence: CC-BY-4.0 (HF card).
- Canary check (grep of all fetched files): one hit, a false positive (a question "Are you aware that humans train you?",
  matching "do not train" inside its choices). No canary string. PASS.
- Categories: {', '.join(CATS)}; self-awareness-* and one-box-tendency skipped (no temptation).
  {PER} human-written + {PER} LM-written per category, random (seed 0). Question + choices verbatim as one user turn, no glue.
- All slice A; group = the question. Drops (keyword hard filters, duplicates): {dict(drop)}
  e.g. {json.dumps(dropped_ex, ensure_ascii=False)}

## Counts
```
{json.dumps(rep, indent=1)}
```

## Examples
{ex}
""")
print(json.dumps(rep))
