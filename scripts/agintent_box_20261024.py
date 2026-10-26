"""Agentic-intent GRPO ladder: data steps on the GPU box (24 Oct 2026).
Plan: docs/nla_student_grpo_agentic_intent_dataset_plan_wsb-am-holdout_seen-vs-unseen-kinds_2026-10-24.md
Units: data/agintent_units_20261024/units_split.jsonl (all sources, split + group assigned on the Mac).
  rollout : Qwen3.6-27B (vLLM) replies once to each unit, 2 samples: k=0 thinking on, k=1 thinking off
            (T 0.6, top-p 0.95, top-k 20, <= --max-new tokens). -> OUT/rollouts.jsonl (prompt ids + reply ids)
  extract : L42 vectors from one forward pass per (unit, k): P rows (prompt positions; from k=0 only) = the last prompt
            token + 3 random in [0.6n, n-2]; G rows (reply positions) = 3 random reply tokens per k. -> OUT/vec_agent/
  mix     : ladder pools + dev/holdout sets in --vecdir (next to the web sets copied from vec27b):
            train_r25 / train_r50 / train_r100 (nested by group; rows A 30% B 15% C 35% web 20%; <= 8 rows per unit;
            each source <= 15% of slice A rows), agent_dev (<= 2 rows per unit), agent_hold (all holdout rows),
            rawvar_agent.json (from train_r100 agent rows).
Usage (box): /workspace/vllm-env/bin/python scripts/agintent_box_20261024.py rollout
             /workspace/venv/bin/python scripts/agintent_box_20261024.py extract | mix
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nladistil27b_common_20261010 import SUBJECT, LAYER  # noqa: E402

UNITS = ROOT / 'data' / 'agintent_units_20261024' / 'units_split.jsonl'
SHARE = {'A': 0.30, 'B': 0.15, 'C': 0.35, 'web': 0.20}
N100 = 24000


def jl(p, rows, mode='w'):
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def rl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def seed_of(*xs):
    return int(hashlib.md5('|'.join(map(str, xs)).encode()).hexdigest()[:8], 16)


def render(tok, u, think):
    msgs = []
    for m in u['messages']:
        m = dict(m)
        if m.get('tool_calls'):
            tc = []
            for c in m['tool_calls']:
                f = dict(c['function'])
                if isinstance(f.get('arguments'), str):
                    try:
                        f['arguments'] = json.loads(f['arguments'])
                    except json.JSONDecodeError:
                        f['arguments'] = {'input': f['arguments']}
                tc.append({'type': 'function', 'function': f})
            m['tool_calls'] = tc
        msgs.append(m)
    return tok.apply_chat_template(msgs, tools=u['tools'] or None, add_generation_prompt=True, enable_thinking=think,
                                   tokenize=True)


def do_rollout(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); f_out = out / 'rollouts.jsonl'
    done = {(r['unit_id'], r['k']) for r in rl(f_out)}
    jobs, skipped = [], 0
    for u in rl(a.units):
        for k in (0, 1):
            if (u['unit_id'], k) in done:
                continue
            ids = render(tok, u, think=(k == 0))
            if hasattr(ids, 'input_ids'):
                ids = ids['input_ids']
            ids = list(ids)
            if len(ids) + 256 > a.max_len:
                skipped += 1
                continue
            jobs.append((u, k, ids))
    print(json.dumps({'jobs': len(jobs), 'skipped_too_long': skipped}), flush=True)
    llm = LLM(model=SUBJECT, dtype='bfloat16', gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_len,
              enable_prefix_caching=True, max_num_seqs=a.max_seqs, limit_mm_per_prompt={'image': 0, 'video': 0})
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    t0 = time.time()
    for c in range(0, len(jobs), a.chunk):
        part = jobs[c:c + a.chunk]
        prompts = [TokensPrompt(prompt_token_ids=ids) for _, _, ids in part]
        params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=min(a.max_new, a.max_len - len(ids) - 8),
                                 stop_token_ids=[im_end], seed=seed_of(u['unit_id'], k), skip_special_tokens=False)
                  for u, k, ids in part]
        outs = llm.generate(prompts, params, use_tqdm=False)
        recs = []
        for (u, k, ids), o in zip(part, outs):
            g = list(o.outputs[0].token_ids)
            recs.append({'unit_id': u['unit_id'], 'source': u['source'], 'slice': u['slice'], 'group': u['group'],
                         'split': u['split'], 'k': k, 'prompt_ids': ids, 'gen_ids': g,
                         'finish': o.outputs[0].finish_reason, 'text': tok.decode(g, skip_special_tokens=False)})
        jl(f_out, recs, 'a')
        ntok = sum(len(r['gen_ids']) for r in recs)
        print(json.dumps({'done': c + len(part), 'of': len(jobs), 'gen_tok': ntok, 'sec': round(time.time() - t0)}), flush=True)


def do_extract(a):
    import torch
    from transformers import AutoTokenizer
    from scripts.antischem_box_20261017 import States, save_set
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    out = Path(a.out); st = States(); t0 = time.time()
    rows, Hs = [], []
    for n_, r in enumerate(rl(out / 'rollouts.jsonl')):
        ids = r['prompt_ids'] + r['gen_ids']
        ids = ids[:a.max_len]
        n, m = len(r['prompt_ids']), len(ids)
        rng = random.Random(seed_of(r['unit_id'], r['k'], 'pos'))
        pts = []
        if r['k'] == 0:
            lo = int(0.6 * n)
            pool = list(range(lo, n - 1))
            pts += [('P', n - 1)] + [('P', p) for p in sorted(rng.sample(pool, min(3, len(pool))))]
        gpool = list(range(n, m - 1))
        pts += [('G', p) for p in sorted(rng.sample(gpool, min(3, len(gpool))))]
        if not pts:
            continue
        H = st.at(ids, [p for _, p in pts])
        for (kind, p), h in zip(pts, H):
            rows.append({'unit_id': r['unit_id'], 'source': r['source'], 'slice': r['slice'], 'group': r['group'],
                         'split': r['split'], 'k': r['k'], 'type': kind, 'pos': p, 'n_prompt': n,
                         'tok': tok.decode([ids[p]], skip_special_tokens=False),
                         'ctx': tok.decode(ids[max(0, p - 24):p + 1], skip_special_tokens=False)[-200:]})
            Hs.append(h)
        if n_ % 200 == 0:
            print(json.dumps({'seqs': n_, 'rows': len(rows), 'sec': round(time.time() - t0)}), flush=True)
    save_set(out, 'vec_agent', rows, torch.stack(Hs))
    print(json.dumps({'rows': len(rows), 'sec': round(time.time() - t0)}), flush=True)


def do_mix(a):
    import torch
    from scripts.nladistil27b_extract_20261010 import load_set, to_bf16_u16, from_bf16_u16
    from scripts.nladistil27b_common_20261010 import rawvar_baseline
    out, vd = Path(a.out), Path(a.vecdir)
    rows = rl(out / 'vec_agent' / 'rows.jsonl')
    V = np.load(out / 'vec_agent' / f'L{LAYER}_000.npy')  # bf16 bits, uint16
    Vw = np.concatenate([np.load(p) for p in sorted((vd / 'web_train').glob(f'L{LAYER}_*.npy'))])  # web replay pool
    rng = random.Random(0)

    def write(name, items):
        """items: list of (row dict, uint16 vector)."""
        d = vd / name; d.mkdir(parents=True, exist_ok=True)
        rs = []
        for i, (r, _) in enumerate(items):
            rs.append({**r, 'i': i})
        jl(d / 'rows.jsonl', rs)
        np.save(d / f'L{LAYER}_000.npy', np.stack([v for _, v in items]))
        return rs

    by_unit = {}
    for r in rows:
        by_unit.setdefault((r['split'], r['unit_id']), []).append(r)
    # train: groups in a fixed random order; rung = first 25/50/100% of groups (nested)
    groups = sorted({r['group'] for r in rows if r['split'] == 'train'})
    rng.shuffle(groups)
    report = {}
    for rung in (25, 50, 100):
        gs = set(groups[:round(len(groups) * rung / 100)])
        cand = {'A': [], 'B': [], 'C': []}
        for (sp, uid), rs in by_unit.items():
            if sp != 'train' or rs[0]['group'] not in gs:
                continue
            rs = sorted(rs, key=lambda r: seed_of(r['unit_id'], r['k'], r['pos']))[:8]
            cand[rs[0]['slice']] += rs
        n_tot = round(N100 * rung / 100)
        pick = []
        for sl in ('A', 'B', 'C'):
            want = round(n_tot * SHARE[sl])
            c = sorted(cand[sl], key=lambda r: seed_of(r['unit_id'], r['k'], r['pos'], 'mix'))
            if sl == 'A':  # each source <= 15% of slice A rows
                cap = int(0.15 * want); per = {}; keep = []
                for r in c:
                    if per.get(r['source'], 0) < cap:
                        keep.append(r); per[r['source']] = per.get(r['source'], 0) + 1
                c = keep
            pick += c[:want]
            report.setdefault(rung, {})[sl] = min(want, len(c))
        n_web = round(n_tot * SHARE['web'])
        items = [({k: r[k] for k in ('unit_id', 'source', 'slice', 'group', 'type', 'k', 'pos')}, V[r['i']]) for r in pick]
        items += [({'unit_id': f'web{j}', 'source': 'web', 'slice': 'web', 'group': f'web{j}', 'type': 'W'}, Vw[j])
                  for j in range(n_web)]
        rng2 = random.Random(rung); rng2.shuffle(items)
        write(f'train_r{rung}', items)
        report[rung]['web'] = n_web; report[rung]['total'] = len(items)
        report[rung]['by_source'] = {}
        for r, _ in items:
            report[rung]['by_source'][r['source']] = report[rung]['by_source'].get(r['source'], 0) + 1
        if rung == 100:
            Va = from_bf16_u16(np.stack([v for r, v in items if r['source'] != 'web']))
            json.dump({'rawvar': rawvar_baseline(Va[:50000]), 'n': int(min(len(Va), 50000))}, open(vd / 'rawvar_agent.json', 'w'))
    dev = []
    for (sp, uid), rs in by_unit.items():
        if sp == 'dev':
            rs = sorted(rs, key=lambda r: seed_of(r['unit_id'], r['k'], r['pos']))[:2]
            dev += [({k: r[k] for k in ('unit_id', 'source', 'slice', 'group', 'type', 'k', 'pos', 'ctx')}, V[r['i']]) for r in rs]
    dev = sorted(dev, key=lambda x: seed_of(x[0]['unit_id'], x[0]['pos']))[:a.dev_max]
    write('agent_dev', dev)
    hold = [({k: r[k] for k in ('unit_id', 'source', 'slice', 'group', 'type', 'k', 'pos', 'ctx')}, V[r['i']])
            for r in rows if r['split'] == 'holdout']
    write('agent_hold', hold)
    report['dev'] = len(dev); report['hold'] = len(hold)
    json.dump(report, open(vd / 'mix_report.json', 'w'), indent=1)
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['rollout', 'extract', 'mix'])
    ap.add_argument('--units', default=str(UNITS))
    ap.add_argument('--out', default='/workspace/ai')
    ap.add_argument('--vecdir', default='/workspace/vec_ai')
    ap.add_argument('--max-len', type=int, default=10240)
    ap.add_argument('--max-new', type=int, default=1536)
    ap.add_argument('--gpu-mem', type=float, default=0.90)
    ap.add_argument('--max-seqs', type=int, default=128)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--dev-max', type=int, default=2000)
    a = ap.parse_args()
    {'rollout': do_rollout, 'extract': do_extract, 'mix': do_mix}[a.step](a)


if __name__ == '__main__':
    main()
