"""WildChat (HF allenai/WildChat-1M, ODC-BY) -> agintent slice-C units (24 Oct 2026).
Spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md
(data/tmplae2_chats_20261017/chats.jsonl has only 118 English WildChat rows and no moderation fields, so we use HF shard 0.)
1,000 ordinary chats over many languages, seed 0. Unit = the first user turn (half) or the first 3 turns user/assistant/user (half,
when the chat has them). Dropped: WildChat `toxic`, any OpenAI-moderation flag, sexual scores, sexual / jailbreak keywords,
hard-filter keywords, very short or code-only first turns, likely mis-tagged languages.
Usage: .venv/bin/python -I scripts/agintent_conv_wildchat_20261024.py RAW_DIR OUT_DIR
"""
import json, random, re, statistics, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N, MAXC = 1000, 24000
OTHER = ['French', 'Spanish', 'Portuguese', 'German', 'Italian', 'Turkish', 'Japanese', 'Korean', 'Arabic', 'Persian',
         'Vietnamese', 'Indonesian', 'Polish', 'Dutch', 'Ukrainian', 'Hindi']
QUOTA = {'English': 400, 'Chinese': 150, 'Russian': 100, **{l: 22 for l in OTHER}}   # English tops up any shortfall to N
# small languages are left out: on short or code-heavy text WildChat's language tag is often wrong (e.g. "Maori", "Sotho")
SCRIPT = {'Chinese': r'[\u4e00-\u9fff]', 'Japanese': r'[\u3040-\u30ff\u4e00-\u9fff]', 'Korean': r'[\uac00-\ud7af]',
          'Russian': r'[\u0400-\u04ff]', 'Ukrainian': r'[\u0400-\u04ff]', 'Arabic': r'[\u0600-\u06ff]', 'Persian': r'[\u0600-\u06ff]',
          'Hindi': r'[\u0900-\u097f]'}
SEXUAL = re.compile(r'sex|nsfw|porn|erotic|nude|naked|fetish|tickl|breast|boob|horny|lewd|seduc|kink|bdsm|onlyfans|hentai|'
                    r'色情|做爱|性爱|裸|секс|эрот|порн|голая|голый', re.I)
JAILBREAK = re.compile(r'\bDAN\b|jailbreak|ignore (all )?(the )?(previous|prior) instructions|developer mode|no restrictions|'
                       r'without (any )?(filters|censorship)|uncensored|stay in character', re.I)
FILTERS = {
    'blackmail_coercion': re.compile(r'blackmail|extort', re.I),
    'bio_chem': re.compile(r'bioweapon|biological weapon|chemical weapon|nerve agent|sarin|ricin|anthrax', re.I),
    'banned_name': re.compile(r'summitbridge|novastar', re.I),
    'canary_text': re.compile(r'canary GUID|BENCHMARK DATA SHOULD NEVER', re.I),
}


def bad_moderation(r, n):
    for m in (r['openai_moderation'] or [])[:n]:
        sc = (m or {}).get('category_scores') or {}
        if not m or m['flagged'] or (sc.get('sexual') or 0) > 0.05 or (sc.get('violence') or 0) > 0.3:
            return True
    return any(not d or (d['sexual_explicit'] or 0) > 0.05 or (d['toxicity'] or 0) > 0.3 for d in (r['detoxify_moderation'] or [])[:n])


cols = ['conversation_hash', 'model', 'conversation', 'language', 'toxic', 'redacted', 'turn', 'openai_moderation', 'detoxify_moderation']
rows = pq.read_table(RAW / 'data/train-00000-of-00014.parquet', columns=cols).to_pylist()
rng = random.Random(0); rng.shuffle(rows)
canary_hits = sum(1 for r in rows for m in r['conversation'] if FILTERS['canary_text'].search(m['content'] or ''))

by_lang, drop, seen = {}, Counter(), set()
for r in rows:
    lang = r['language']
    if lang not in QUOTA:
        drop['other_language'] += 1; continue
    if len(by_lang.get(lang, [])) >= (QUOTA[lang] if lang != 'English' else N):
        continue
    conv = [m for m in r['conversation'] if m['role'] in ('user', 'assistant')]
    if not conv or conv[0]['role'] != 'user':
        drop['bad_shape'] += 1; continue
    long = len(conv) >= 3 and conv[1]['role'] == 'assistant' and conv[2]['role'] == 'user' and rng.random() < 0.5
    msgs = [{'role': m['role'], 'content': m['content']} for m in conv[:3 if long else 1]]
    first = msgs[0]['content'].strip()
    text = '\n'.join(m['content'] for m in msgs)
    if r['toxic'] or any(m.get('toxic') for m in conv[:len(msgs)]):
        drop['toxic_flag'] += 1; continue
    if bad_moderation(r, len(msgs)):
        drop['moderation'] += 1; continue
    if SEXUAL.search(text):
        drop['sexual_keyword'] += 1; continue
    if JAILBREAK.search(text):
        drop['jailbreak_keyword'] += 1; continue
    why = next((k for k, rx in FILTERS.items() if rx.search(text)), None)
    if why:
        drop[why] += 1; continue
    if len(first) < 15 or sum(c.isalpha() for c in first) < 0.4 * len(first):
        drop['short_or_code'] += 1; continue
    if lang in SCRIPT and len(re.findall(SCRIPT[lang], first)) < 0.3 * sum(c.isalpha() for c in first):
        drop['language_tag_mismatch'] += 1; continue
    if first in seen:
        drop['duplicate'] += 1; continue
    if len(text) > MAXC:
        if long:
            msgs = msgs[:1]
        if len(msgs[0]['content']) > MAXC:
            msgs[0]['content'] = msgs[0]['content'][:MAXC - 20] + '\n... [truncated]'
            drop['truncated(kept)'] += 1
    seen.add(first)
    h = r['conversation_hash'][:16]
    by_lang.setdefault(lang, []).append(
        {'unit_id': f'wildchat-{h}', 'source': 'wildchat', 'slice': 'C', 'twin_of': None, 'group': f'wildchat-{h}', 'kind': 'none',
         'messages': msgs, 'tools': [], 'license': 'ODC-BY',
         'meta': {'conversation_hash': r['conversation_hash'], 'language': lang, 'model': r['model'], 'turns_kept': len(msgs),
                  'turns_total': r['turn'], 'redacted': r['redacted']}})

non_en = [u for k, v in by_lang.items() if k != 'English' for u in v]
units = non_en + by_lang.get('English', [])[:N - len(non_en)]
units.sort(key=lambda u: (u['meta']['language'], u['unit_id']))
with open(OUT / 'wildchat.jsonl', 'w') as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False) + '\n')
ch = [sum(len(m['content']) for m in u['messages']) for u in units]
rep = {'units': len(units), 'by_slice': dict(Counter(u['slice'] for u in units)),
       'by_language': dict(Counter(u['meta']['language'] for u in units).most_common()),
       'by_turns_kept': dict(Counter(u['meta']['turns_kept'] for u in units)),
       'median_chars': statistics.median(ch), 'max_chars': max(ch), 'dropped_while_sampling': dict(drop),
       'canary_text_hits_in_shard': canary_hits}
pick = [u for u in units if u['meta']['language'] == 'English'][:1] + [u for u in units if u['meta']['language'] == 'Chinese'][:1]
ex = '\n\n'.join('```json\n' + json.dumps(u, indent=1, ensure_ascii=False)[:1200] + '\n```' for u in pick)
(OUT / 'wildchat.README.md').write_text(f"""# wildchat -> agintent units (24 Oct 2026), slice C

- Fetched: HF `allenai/WildChat-1M` @ 7d6490e462285cf85d91eabea0f9a954fbddcd1f, shard `data/train-00000-of-00014.parquet` only (230 MB, 59,857 chats).
  The local `data/tmplae2_chats_20261017/chats.jsonl` has only 118 English WildChat rows and no moderation fields, so not used.
- Converter: `scripts/agintent_conv_wildchat_20261024.py`. Licence: ODC-BY (WildChat-1M card + LICENSE.md).
- Canary check: README/LICENSE no hits; canary strings inside the shard's chat text: {canary_hits}. PASS.
- Sample (seed 0): quotas English {QUOTA['English']}, Chinese {QUOTA['Chinese']}, Russian {QUOTA['Russian']}, and 22 each of {', '.join(OTHER)};
  English tops up the rest to {N}. Small languages left out (WildChat's tag is often wrong on short or code text);
  for non-Latin-script languages at least 30% of letters must be in that script.
- Unit = first user turn (about half), or the first 3 turns user/assistant/user (assistant turn = WildChat's GPT reply, verbatim).
- Drops: WildChat `toxic`; any OpenAI-moderation flag, sexual score > 0.05 or violence > 0.3; Detoxify sexual > 0.05 or toxicity > 0.3;
  sexual and jailbreak keyword lists (several languages); hard-filter keywords; first turn < 15 chars or mostly non-letters (code/paste); duplicates.
  `redacted` (PII-scrubbed) chats are kept (meta.redacted).

## Counts
```
{json.dumps(rep, indent=1, ensure_ascii=False)}
```

## Examples
{ex}
""")
print(json.dumps(rep, ensure_ascii=False))
