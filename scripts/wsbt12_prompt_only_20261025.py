"""Prompt-only floor readouts for WorkspaceBench tiers 1+2 (25 Oct 2026), rebuilt with the bench's recorded recipe
(evals/baselines/prompt_only.json _source): Qwen3.6-27B, chat, thinking off, T 1, top-p 0.95, top-k 64, 256 tokens,
the bench's prompt_only_summary template. Sent to a vLLM OpenAI server. Writes D/notes_prompt_only.jsonl ({u, s, text})
so scripts/wsbt12_box_20261025.py export can turn it into readouts like any reader.
Usage (bench venv): python scripts/wsbt12_prompt_only_20261025.py CELLS D BASE_URL [K=4]"""
import asyncio, json, sys
from pathlib import Path
from openai import AsyncOpenAI
from transformers import AutoTokenizer

CELLS, D, BASE = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TMPL = ('Below is a prompt that was given to a language model, cut off at a specific token. The token in question is the '
        'final token of the text below: {token!r}.\n\n<prompt>\n{prefix}\n</prompt>\n\nWhat would a language model be '
        'thinking about on this token? Please write me a summary for this token.')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B')
cells = [json.loads(l) for l in open(CELLS)]
keys, units = {}, []
for c in cells:
    k = json.dumps(c['ids'][:c['idx'] + 1])
    if k not in keys:
        keys[k] = len(keys); units.append(c)
client = AsyncOpenAI(base_url=BASE, api_key='x')
sem = asyncio.Semaphore(64)


async def one(u, c):
    prompt = TMPL.format(token=c['token'], prefix=tok.decode(c['ids'][:c['idx'] + 1]))
    async with sem:
        r = await client.chat.completions.create(model='qwen', messages=[{'role': 'user', 'content': prompt}], n=K,
                                                 temperature=1.0, top_p=0.95, max_tokens=256,
                                                 extra_body={'top_k': 64, 'chat_template_kwargs': {'enable_thinking': False}})
    return [{'u': u, 's': s, 'text': ch.message.content.strip(), 'parsed': True, 'n_tok': 0} for s, ch in enumerate(r.choices)]


async def main():
    rows = await asyncio.gather(*[one(u, c) for u, c in enumerate(units)])
    with open(D / 'notes_prompt_only.jsonl', 'w') as f:
        for rs in rows:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(len(units), 'units', sum(len(r) for r in rows), 'notes; example:', rows[0][0]['text'][:300])

asyncio.run(main())
