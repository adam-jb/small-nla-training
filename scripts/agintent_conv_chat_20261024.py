"""Chat sources -> agintent units (24 Oct 2026). Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
  darkbench (Apache-2.0)       : every prompt, slice A, kind = dark pattern. Bio/chem-looking prompts dropped.
  mwe_advanced_ai_risk (CC-BY-4.0): human_generated_evals, 75 per category (self-awareness + one-box skipped: no temptation), slice A.
  deceptionbench (CC-BY-SA-4.0): whole source is holdout (meta.holdout_source). One group per item: L2 pressure/reward
                                 prompts slice A; L1 (no pressure) slice B, twin of L2-self-pressure / L2-other-pressure.
  wildchat (ODC-BY)            : 1,500 English, non-toxic, non-redacted chats from shard 0, slice C; cut after the last user turn.
MASK is dropped: its README carries a do-not-train canary. MACHIAVELLI skipped (not easy to fetch).
Usage: .venv/bin/python -I scripts/agintent_conv_chat_20261024.py RAW_DIR OUT_DIR
"""
import csv, glob, json, random, re, sys
from collections import Counter
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BIOCHEM = re.compile(r'\b(pathogen|virus|bioweapon|anthrax|toxin|nerve agent|sarin|ricin|chemical weapon|explosive|synthesi[sz]e)', re.I)
MAXC = 24000


def uid(s):
    return re.sub(r'[^a-z0-9_-]+', '-', s.lower()).strip('-')


def write(name, units, notes):
    with open(OUT / f'{name}.jsonl', 'w') as f:
        for u in units:
            f.write(json.dumps(u) + '\n')
    sl = Counter(u['slice'] for u in units)
    ch = sorted(sum(len(m['content']) for m in u['messages']) for u in units)
    rep = {'source': name, 'units': len(units), 'by_slice': dict(sl), 'median_chars': ch[len(ch) // 2], 'max_chars': ch[-1],
           'kinds': dict(Counter(u['kind'] for u in units).most_common(12))}
    ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1)[:1200] + '\n```' for u in units[:2])
    (OUT / f'{name}.README.md').write_text(f'# {name} -> agintent units (24 Oct 2026)\n\nConverter: `scripts/agintent_conv_chat_20261024.py`.\n\n'
                                           + notes + '\n\n## Counts\n```\n' + json.dumps(rep, indent=1) + '\n```\n\n## Examples\n' + ex + '\n')
    print(json.dumps(rep))


def unit(src, i, sl, kind, msgs, lic, meta, group=None, twin=None):
    return {'unit_id': uid(f'{src}-{i}'), 'source': src, 'slice': sl, 'twin_of': twin, 'group': group or uid(f'{src}-{i}'),
            'kind': kind, 'messages': msgs, 'tools': [], 'license': lic, 'meta': meta}


# darkbench
units, drop = [], Counter()
with open(RAW / 'darkbench/darkbench.tsv') as f:
    for i, r in enumerate(csv.DictReader(f, delimiter='\t')):
        p = r['Example'].strip()
        if BIOCHEM.search(p):
            drop['biochem_keyword'] += 1; continue
        units.append(unit('darkbench', i, 'A', uid(r['Deceptive Pattern']), [{'role': 'user', 'content': p[:MAXC]}],
                          'Apache-2.0', {'pattern': r['Deceptive Pattern']}))
write('darkbench', units, f'- Licence: Apache-2.0 (HF card). Canary grep: no hits. PASS.\n- One user turn per prompt, verbatim. Dropped: {dict(drop)}.')

# mwe advanced-ai-risk (human-written)
units, rng = [], random.Random(0)
for fp in sorted(glob.glob(str(RAW / 'mwe_advanced_ai_risk/advanced-ai-risk/human_generated_evals/*.jsonl'))):
    cat = Path(fp).stem
    if cat.startswith('self-awareness') or cat == 'one-box-tendency':
        continue
    rows = [json.loads(l) for l in open(fp)]
    rng.shuffle(rows)
    for j, r in enumerate(rows[:75]):
        units.append(unit('mwe_advanced_ai_risk', f'{cat}-{j}', 'A', cat, [{'role': 'user', 'content': r['question'].strip()}],
                          'CC-BY-4.0', {'category': cat, 'answer_matching_behavior': r['answer_matching_behavior']}))
write('mwe_advanced_ai_risk', units, '- Licence: CC-BY-4.0 (anthropics/evals). Canary grep: one hit, a false positive ("humans do not train me" inside a question). PASS.\n'
      '- human_generated_evals only, 75 random per category (seed 0); self-awareness-* and one-box-tendency skipped (no temptation). Question + choices verbatim.')

# deceptionbench (whole holdout)
units = []
for fp in sorted(glob.glob(str(RAW / 'deceptionbench/*.json'))):
    for j, r in enumerate(json.load(open(fp))):
        g = uid(f'deceptionbench-{Path(fp).stem}-{j}')
        for key in ['L1-self', 'L2-self-pressure', 'L2-self-reward', 'L1-other', 'L2-other-pressure', 'L2-other-reward']:
            if not r.get(key):
                continue
            sl = 'B' if key.startswith('L1') else 'A'
            twin = uid(f'{g}-{key.replace("L1", "L2")}-pressure') if sl == 'B' else None
            units.append(unit('deceptionbench', f'{Path(fp).stem}-{j}-{key}', sl, 'deception', [{'role': 'user', 'content': r[key].strip()}],
                              'CC-BY-SA-4.0', {'holdout_source': True, 'topic': r['topic'], 'dimension': r['dimension'], 'variant': key},
                              group=g, twin=twin))
write('deceptionbench', units, '- Licence: CC-BY-SA-4.0. Canary grep: no hits. PASS.\n- Whole source = holdout (meta.holdout_source). '
      'L1 prompts are slice B, twin of the matching L2 pressure prompt; L2 pressure/reward prompts slice A. One group per item.')

# wildchat (slice C)
import pyarrow.parquet as pq
t = pq.read_table(RAW / 'wildchat/data/train-00000-of-00014.parquet',
                  columns=['conversation_hash', 'conversation', 'language', 'toxic', 'redacted', 'turn']).to_pylist()
rng = random.Random(0); rng.shuffle(t)
units, drop = [], Counter()
for r in t:
    if len(units) >= 1500:
        break
    if r['language'] != 'English' or r['toxic'] or r['redacted']:
        drop['lang_toxic_redacted'] += 1; continue
    msgs = [{'role': m['role'], 'content': m['content']} for m in r['conversation'] if m['role'] in ('user', 'assistant')]
    while msgs and msgs[-1]['role'] != 'user':
        msgs.pop()
    while msgs and sum(len(m['content']) for m in msgs) > MAXC:
        msgs = msgs[2:] if len(msgs) > 2 else []
    if not msgs or msgs[0]['role'] != 'user' or len(msgs[-1]['content'].strip()) < 20:
        drop['empty_or_short'] += 1; continue
    if BIOCHEM.search(' '.join(m['content'] for m in msgs)):
        drop['biochem_keyword'] += 1; continue
    units.append(unit('wildchat', r['conversation_hash'][:16], 'C', 'none', msgs, 'ODC-BY', {'turns': r['turn']}))
write('wildchat', units, f'- Licence: ODC-BY. Canary grep of README/LICENSE: no hits. PASS.\n- Shard 0, shuffled (seed 0), English, not toxic, not redacted; cut after the last user turn; '
      f'oldest turns dropped in pairs if over {MAXC} chars. Dropped while sampling: {dict(drop)}.')
