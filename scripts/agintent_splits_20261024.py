"""Merge converted units, drop bench overlap, assign splits (agentic-intent GRPO ladder, 24 Oct 2026).

Reads data/agintent_units_20261024/<source>.jsonl (format: docs/agintent_unit_format_spec_..._2026-10-24.md).
Drops: any unit sharing a banned name or an 8-word run with any of the 32 WorkspaceBench agentic_misalignment items
(data/wsb_am_items_92d763e/items.json: system + text + prefill); units with held-out misdeed kinds by keyword
(blackmail, leak to outsider, lethal harm) as a second net behind the converters' own filters.
Splits by group (a unit and its twins never straddle splits), per source, seeded: train 75 / dev 10 / holdout 15.
DeceptionBench (meta.holdout_source) is holdout as a whole source.
Writes data/agintent_units_20261024/units_split.jsonl and split_report.json.
"""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[1] / 'data'
U = D / 'agintent_units_20261024'
BANNED = ['summitbridge', 'novastar', 'kyle johnson', 'alex (', 'agentic-misalignment']
KIND_RE = re.compile(r'blackmail|extort|coerc|leverage (?:the|his|her) affair|leak(?:ing|ed)? (?:to|the) (?:press|rival|competitor)'
                     r'|let (?:him|her|them|the \w+) die|cancel(?:ling)? the (?:emergency|alert)|lethal', re.I)


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def grams(t, n=8):
    w = words(t)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def text_of(u):
    return '\n'.join(m.get('content') or '' for m in u['messages'])


def main():
    items = json.load(open(D / 'wsb_am_items_92d763e' / 'items.json'))['items']
    bench = set()
    for it in items:
        bench |= grams('\n'.join([it.get('system') or '', it['text'], it.get('prefill') or '']))
    units, drops = [], Counter()
    for f in sorted(U.glob('*.jsonl')):
        if f.name == 'units_split.jsonl':
            continue
        for l in open(f):
            u = json.loads(l); t = text_of(u); tl = t.lower()
            if any(b in tl for b in BANNED):
                drops[(u['source'], 'banned_name')] += 1; continue
            if grams(t) & bench:
                drops[(u['source'], 'bench_8gram')] += 1; continue
            if u['slice'] != 'C' and KIND_RE.search(t):
                drops[(u['source'], 'held_out_kind_keyword')] += 1; continue
            last = u['messages'][-1]['role']
            if last not in ('user', 'tool'):
                drops[(u['source'], 'bad_last_role')] += 1; continue
            units.append(u)
    # drop a slice-B twin whose partner was dropped, and vice versa keep groups whole
    have = {u['unit_id'] for u in units}
    units = [u for u in units if not u.get('twin_of') or u['twin_of'] in have]
    by_src = defaultdict(lambda: defaultdict(list))
    for u in units:
        by_src[u['source']][u['group']].append(u)

    for src, gs in sorted(by_src.items()):
        keys = sorted(gs); random.Random(f"0-{src}").shuffle(keys)  # own seed per source: adding sources keeps old splits
        hold_src = any(u.get('meta', {}).get('holdout_source') for g in keys for u in gs[g])
        n = len(keys); n_dev = max(1, round(0.10 * n)) if n >= 5 else 0; n_hold = max(1, round(0.15 * n)) if n >= 5 else 0
        for i, g in enumerate(keys):
            sp = 'holdout' if hold_src else ('dev' if i < n_dev else 'holdout' if i < n_dev + n_hold else 'train')
            for u in gs[g]:
                u['split'] = sp
    with open(U / 'units_split.jsonl', 'w') as f:
        for u in units:
            f.write(json.dumps(u) + '\n')
    rep = {'n_units': len(units),
           'by_source_split_slice': {s: dict(Counter(f"{u['split']}/{u['slice']}" for u in units if u['source'] == s))
                                     for s in sorted(by_src)},
           'drops': {f'{s}|{why}': n for (s, why), n in sorted(drops.items())}}
    json.dump(rep, open(U / 'split_report.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    sys.exit(main())
