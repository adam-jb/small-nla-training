"""DarkBench (HF anonymous152311/darkbench, Apache-2.0) -> agintent units (24 Oct 2026).
Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
Every prompt is one user turn, verbatim, slice A, kind = dark-pattern category. "Harmful generation" kept (meta.harmful_generation).
Prompts that hit the hard filters (blackmail / coercion, leaking, lethal harm, bio/chem weapons, banned names) are dropped.
Usage: .venv/bin/python -I scripts/agintent_conv_darkbench_20261024.py RAW_DIR OUT_DIR
"""
import csv, json, re, statistics, sys
from collections import Counter
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
FILTERS = {
    'blackmail_coercion': re.compile(r'blackmail|extort|coerc', re.I),
    'leaking': re.compile(r'\bleak(s|ed|ing)?\b', re.I),
    'lethal_harm': re.compile(r'ending their life|suicide|assassinating|exterminat|\bkill', re.I),
    'bio_chem': re.compile(r'bioweapon|biological weapon|chemical weapon|nerve agent|spreading a virus|pathogen', re.I),
    'banned_name': re.compile(r'summitbridge|novastar', re.I),
}


def uid(s):
    return re.sub(r'[^a-z0-9_-]+', '-', s.lower()).strip('-')


units, drop, dropped_ex = [], Counter(), {}
with open(RAW / 'darkbench.tsv', newline='') as f:
    for i, r in enumerate(csv.DictReader(f, delimiter='\t')):
        p, cat = r['Example'].strip(), r['Deceptive Pattern'].strip()
        why = next((k for k, rx in FILTERS.items() if rx.search(p)), None)
        if why:
            drop[why] += 1; dropped_ex.setdefault(why, p[:160]); continue
        units.append({'unit_id': f'darkbench-{i:04d}', 'source': 'darkbench', 'slice': 'A', 'twin_of': None, 'group': f'darkbench-{i:04d}',
                      'kind': uid(cat), 'messages': [{'role': 'user', 'content': p}], 'tools': [], 'license': 'Apache-2.0',
                      'meta': {'pattern': cat, 'row': i, 'harmful_generation': cat == 'Harmful generation'}})

with open(OUT / 'darkbench.jsonl', 'w') as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + '\n')
ch = [len(u['messages'][0]['content']) for u in units]
rep = {'units': len(units), 'by_slice': dict(Counter(u['slice'] for u in units)), 'kinds': dict(Counter(u['kind'] for u in units)),
       'median_chars': statistics.median(ch), 'max_chars': max(ch), 'dropped': dict(drop)}
ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1, ensure_ascii=False)[:1000] + '\n```' for u in (units[0], units[-1]))
(OUT / 'darkbench.README.md').write_text(f"""# darkbench -> agintent units (24 Oct 2026)

- Fetched: HF `anonymous152311/darkbench` @ a3c639ad48a5f36df36c6d5fead78a08382b568f (`darkbench.tsv`, 660 prompts, 6 categories x 110).
- Converter: `scripts/agintent_conv_darkbench_20261024.py`. Licence: Apache-2.0 (HF card).
- Canary check (grep of all fetched files): no hits. PASS.
- One user turn per prompt, verbatim, no glue. All slice A; kind = dark-pattern category; group = the prompt itself.
- "Harmful generation" kept and tagged (`meta.harmful_generation`). Hard-filter drops (keyword): {dict(drop)}
  e.g. {json.dumps(dropped_ex, ensure_ascii=False)}

## Counts
```
{json.dumps(rep, indent=1)}
```

## Examples
{ex}
""")
print(json.dumps(rep))
