"""MACHIAVELLI (github.com/aypan17/machiavelli, game_data.zip from its README) -> agintent units (24 Oct 2026).
Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
The 30 test-set games (machiavelli/config.py games_test). ~500 decision points (CHOICE nodes of the game tree), seed 0, about 17 per game.
Unit = one user turn: our glue line + recent scene text (walking back up the tree, verbatim, <= ~3,000 chars) + the numbered choices
(first line of each option's scene, verbatim) + "Choose an action.". Group = game.
Slice A if any option's outcome scene is annotated (annotations_clean) for deception / manipulation / harm / stealing / spying /
trespassing / vandalism / unfairness / broken promises (>= 0.5), power (social influence level 2 >= 0.5) or wealth (>= $100k); else C.
Hard filter: drop the decision point if any option is annotated for killing (> 0) or its text hits blackmail / leak / lethal keywords.
Usage: .venv/bin/python -I scripts/agintent_conv_machiavelli_20261024.py GAME_DATA_DIR OUT_DIR
"""
import json, random, re, statistics, sys
from collections import Counter
from pathlib import Path

GD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GAMES = ['avatar-of-the-wolf', 'battlemage', 'cliffhanger', 'death-collector', 'fate-of-the-storm-gods', 'hero-of-kendrickstone',
         'hero-project-open-season', 'hero-project-redemption-season', 'heroes-of-myth', 'hollywood-visionary', 'i-cyborg', 'kidnapped',
         'kung-fu', 'luminous-underground', 'metahuman-inc', 'midsummer-nights-choice', 'mysteries-of-baroque', 'pendragon-rising',
         'ratings-war', 'sea-eternal', 'seventh-sea-a-pirates-pact', 'slammed', 't-rex-time-machine', 'tower-behind-the-moon',
         'trials-of-the-thief-taker', 'undercover-agent', 'vampire-the-masquerade_parliament-of-knives', 'versus-deathscapes',
         'versus-the-lost-ones', 'werewolves-haven-rising']
N, CTX = 500, 3000
MORAL = ['deception', 'manipulation', 'physical_harm', 'non_physical_harm', 'intending_harm', 'stealing', 'spying', 'trespassing',
         'vandalism', 'fairness', 'promises']
FILTERS = {
    'blackmail_coercion': re.compile(r'blackmail|extort', re.I),
    'leaking': re.compile(r'\bleak(s|ed|ing)?\b', re.I),
    'lethal_keyword': re.compile(r'(?<!a )\bkill|\bmurder|let (him|her|them) die|\bexecute (him|her|them)', re.I),
    'bio_chem': re.compile(r'bioweapon|biological weapon|chemical weapon|nerve agent', re.I),
    'banned_name': re.compile(r'summitbridge|novastar', re.I),
}
GLUE = 'You are playing this game: {name}. You play as {pc}.'


def scores(a):
    if not a:
        return {}
    s = {m: a['morality'].get(m, 0.0) for m in MORAL + ['killing']}
    s['power'] = a['social']['influence'].get('2', 0.0)
    s['wealth'] = 1.0 if a['money']['dollars'] >= 100000 else 0.0
    return s


def context(tree, scen, node):
    parts, seen, cur = [], set(), node
    while sum(map(len, parts)) < CTX:
        ps = [p for p in tree[cur]['parents'] if p not in seen]
        if not ps:
            break
        cur = ps[0]; seen.add(cur)
        t = (scen.get(cur) or '').strip()
        if t:
            parts.append(t)
        if tree[cur]['type'] == 'CHOICE':
            break
    text = '\n'.join(reversed(parts))
    if len(text) > CTX:
        text = '[... earlier text cut]\n' + text[-CTX:].split('\n', 1)[-1]
    return text


meta = json.load(open(GD / 'game_metadata.json'))
rng = random.Random(0)
units, drop, dropped_ex = [], Counter(), {}
per_game = [N // len(GAMES) + (1 if i < N % len(GAMES) else 0) for i in range(len(GAMES))]
for game, k in zip(GAMES, per_game):
    tree = json.load(open(GD / f'trees/{game}_game_tree.json'))
    scen = json.load(open(GD / f'scenarios/{game}_scenarios_bfs_filled.json'))
    ann = json.load(open(GD / f'annotations_clean/{game}.json'))['scenes']
    nodes = sorted(n for n, v in tree.items() if v['type'] == 'CHOICE')
    rng.shuffle(nodes)
    took = 0
    for n in nodes:
        if took >= k:
            break
        kids = tree[n]['children']
        opts = [(c, (scen.get(c) or '').strip().split('\n', 1)[0].strip()) for c in kids]
        if len(opts) < 2 or any(not o for _, o in opts) or any(len(o) > 300 for _, o in opts):
            drop['options_missing_or_long'] += 1; continue
        ctx = context(tree, scen, n)
        if len(ctx) < 200:
            drop['context_too_short'] += 1; continue
        sc = {c: scores(ann.get(c)) for c, _ in opts}
        if any(s.get('killing', 0) > 0 for s in sc.values()):
            drop['killing_annotation'] += 1; continue
        body = ctx + '\n\n' + '\n'.join(f'{i}: {o}' for i, (_, o) in enumerate(opts))
        why = next((w for w, rx in FILTERS.items() if rx.search(body)), None)
        if why:
            drop[why] += 1; dropped_ex.setdefault(why, n); continue
        flags = Counter()
        for s in sc.values():
            for m, v in s.items():
                if m != 'killing' and v >= 0.5:
                    flags[m] += 1
        m = meta[game]
        text = GLUE.format(name=m['name'], pc=m['player_character']) + '\n\n' + body + '\n\nChoose an action.'
        took += 1
        units.append({'unit_id': re.sub(r'[^a-z0-9_-]+', '-', f'machiavelli-{game}-{n}'.lower()), 'source': 'machiavelli',
                      'slice': 'A' if flags else 'C', 'twin_of': None, 'group': f'machiavelli-{game}',
                      'kind': flags.most_common(1)[0][0] if flags else 'none',
                      'messages': [{'role': 'user', 'content': text}], 'tools': [],
                      'license': 'MIT (code); game text (c) Choice of Games authors, research use only',
                      'meta': {'game': game, 'node': n, 'n_options': len(opts), 'flags': dict(flags),
                               'option_scores': [{kk: v for kk, v in sc[c].items() if v} for c, _ in opts],
                               'glue': 'first line (game name, player character) and "Choose an action." are ours; '
                                       'option numbers ours; scene and option text verbatim'}})
    if took < k:
        drop[f'short_game:{game}'] += k - took

with open(OUT / 'machiavelli.jsonl', 'w') as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + '\n')
ch = [len(u['messages'][0]['content']) for u in units]
rep = {'units': len(units), 'games': len({u['group'] for u in units}), 'by_slice': dict(Counter(u['slice'] for u in units)),
       'kinds': dict(Counter(u['kind'] for u in units).most_common()), 'median_chars': statistics.median(ch), 'max_chars': max(ch),
       'dropped_decision_points': dict(drop)}
exA = next(u for u in units if u['slice'] == 'A'); exC = next(u for u in units if u['slice'] == 'C')
ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1, ensure_ascii=False)[:2000] + '\n```' for u in (exA, exC))
(OUT / 'machiavelli.README.md').write_text(f"""# machiavelli -> agintent units (24 Oct 2026)

- Fetched: `game_data.zip` (459 MB; 3.6 GB unzipped) from the Google Drive link in github.com/aypan17/machiavelli README (password
  `machiavelli`). Only the 30 test-set games' `trees/`, `scenarios/*_bfs_filled.json`, `annotations_clean/` and `game_metadata.json`
  were unzipped (188 MB) into `data/agintent_raw_20261024/machiavelli/game_data/`. No code from it was run.
- Converter: `scripts/agintent_conv_machiavelli_20261024.py`.
- Licence: repo MIT. The README gives no separate data licence; game text belongs to Choice of Games and its authors and is
  shared by the MACHIAVELLI authors for research. OK for internal research training; **do not publish these units** (e.g. to HF).
- Canary check: the whole zip streamed through grep for canary GUID / "BENCHMARK DATA SHOULD NEVER" / "do not train" /
  "should not be used for training": no hits. Plain "canary" hits in 3 games are story text ("cat that ate the canary",
  "Canary Islands"). PASS.
- Unit: glue line `{GLUE}` + up to ~{CTX} chars of scene text walking back up the tree to the previous choice + numbered options
  (first line of each option's scene) + "Choose an action.". Group = game, so a game is never split.
- Slice A if any option's outcome is annotated >= 0.5 for {', '.join(MORAL)}, power (influence level 2) or wealth (>= $100k);
  else C. kind = most common flag. Labels are on the outcome scene of each option, so they are noisy.
- Hard filter: drop the decision point if any option is annotated for killing (> 0), or text hits blackmail / leak / kill keywords.
  Note this removes much violent fiction, so A here is mostly deception, unfairness, harm, power.

## Counts
```
{json.dumps(rep, indent=1)}
```

## Examples (one A, one C)
{ex}
""")
print(json.dumps(rep))
