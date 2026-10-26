"""WorkspaceBench preview (P): teacher vs distilled students on the same cells, cheap and local (10 Oct 2026).

Read-only use of the bench: never trained on, used only to predict parity before the official run.
Families: basic_readout (plain items only), multihop, multilingual, typo, poetry, association (items from
camilablank/workspace-bench@92d763e, vendored in data/wsb_items_92d763e/). One cell per item at layer 42:
final_prompt_token (-1), last_word_token (offset as given), line_one_newline. Readers: the teacher (HF, Karvonen hook)
and each student run; one sample per cell, pure T=1, 256 tokens (the official run uses top-p 0.95 / top-k 64).
Judge: Qwen3.6-27B (thinking off, greedy) with the repo's verbatim bank prompt; a YES needs a quote that is really in the
readout (else void), as in the repo. Not comparable to published numbers; compare readers with each other only.
Writes runs/remote_logs/wsb_preview/{reader}.jsonl and summary.json.
Usage: python scripts/nladistil27b_wsb_preview_20261010.py --vec DIR --runs RUN1 RUN2 ...
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (LAYER, MAX_NEW, SUBJECT, Adapter, av_prompt_ids, eos_ids, extract_expl,
                                                 left_pad_embeds, sample_embeds, chat_ids)
from scripts.nladistil27b_extract_20261010 import Extractor

FAMS = ['basic_readout', 'multihop', 'multilingual', 'typo', 'poetry', 'association']
SYSTEM = open(Path(__file__).resolve().parents[1] / 'data/wsb_items_92d763e/judge_system.txt').read() \
    if (Path(__file__).resolve().parents[1] / 'data/wsb_items_92d763e/judge_system.txt').exists() else None


def targets(it):
    out = []
    units = it.get('units') or []
    if isinstance(units, str):
        units = eval(units)
    chosen = [u for u in units if u.get('headline')] or units
    for u in chosen:
        for m in u.get('match', []):
            if m not in out:
                out.append(m)
    inter = it.get('intermediates') or ([it.get('target')] if not out else [])
    if isinstance(inter, str):
        inter = eval(inter)
    for m in inter:
        if m and m not in out:
            out.append(str(m))
    return out


def cell_pos(tok, ids, rd):
    kind = rd['kind']
    if kind == 'final_prompt_token':
        return len(ids) + rd['offsets'][0]
    if kind == 'last_word_token':
        word = [i for i, t in enumerate(ids) if re.search(r'\w', tok.decode([t]))]
        return word[rd['offsets'][0]]
    if kind == 'line_one_newline':
        nl = [i for i, t in enumerate(ids) if '\n' in tok.decode([t])]
        return nl[1] if len(nl) > 1 else nl[0]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--runs', nargs='+', default=[])
    ap.add_argument('--merged', default='/home/ubuntu/teacher_merged')
    ap.add_argument('--out', default='runs/remote_logs/wsb_preview')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    t0 = time.time()
    root = Path(__file__).resolve().parents[1] / 'data/wsb_items_92d763e'
    items = []
    for f in FAMS:
        for it in json.load(open(root / f'{f}.json'))['items']:
            if it['eval_render'] != 'plain':
                continue
            items.append({'family': f, 'name': it['name'], 'prompt': it['prompt'], 'targets': targets(it), 'readout': it['readout']})
    # 1. capture L42 at each cell
    vp = out / 'vectors.npy'
    if not vp.exists():
        ex = Extractor(dev)
        pref = []
        for it in items:
            ids = ex.tok(it['prompt'], add_special_tokens=True)['input_ids']
            p = cell_pos(ex.tok, ids, it['readout'])
            it['pos'] = p
            pref.append(ids[:p + 1])
        H = ex.last_states(pref, 32768)[LAYER].float().numpy()
        np.save(vp, H); json.dump(items, open(out / 'items.json', 'w'))
        del ex; torch.cuda.empty_cache()
    H = np.load(vp); items = json.load(open(out / 'items.json'))
    print(json.dumps({'cells': len(items), 'sec': round(time.time() - t0)}), flush=True)
    # 2. readouts
    readers = {}
    if not (out / 'teacher.jsonl').exists():
        from scripts.nladistil27b_teacher_20261010 import HFTeacher, merge_teacher
        T = HFTeacher(merge_teacher(a.merged), dev)
        res = T.write(H, seed=7, bs=64)
        with open(out / 'teacher.jsonl', 'w') as f:
            for it, r in zip(items, res):
                f.write(json.dumps({'name': it['name'], 'expl': extract_expl(r[0])[0]}) + '\n')
        del T; torch.cuda.empty_cache()
    readers['teacher'] = [json.loads(x)['expl'] for x in open(out / 'teacher.jsonl')]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for run in a.runs:
        name = Path(run).name
        if not (out / f'{name}.jsonl').exists():
            if not (Path(run) / 'final_lm').exists():
                continue
            tok = AutoTokenizer.from_pretrained(Path(run) / 'final_lm')
            lm = AutoModelForCausalLM.from_pretrained(Path(run) / 'final_lm', dtype=torch.bfloat16).to(dev).eval()
            emb, d = lm.get_input_embeddings(), lm.config.hidden_size
            ad = Adapter(d, 4).to(dev); ad.load_state_dict(torch.load(Path(run) / 'final_adapter.pt', map_location=dev))
            ids, p = av_prompt_ids(tok)
            pre, suf = emb(torch.tensor(ids[:p], device=dev)), emb(torch.tensor(ids[p + 1:], device=dev))
            outs = []
            torch.manual_seed(7)
            with torch.no_grad():
                for s in range(0, len(items), 128):
                    A = ad(torch.tensor(H[s:s + 128], device=dev)).to(emb.weight.dtype)
                    P = [torch.cat([pre, A[j], suf]) for j in range(A.shape[0])]
                    E, M = left_pad_embeds(P, d, P[0].dtype, dev)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        tk = sample_embeds(lm, E, M, eos_ids(tok), max_new=MAX_NEW)
                    outs += [extract_expl(tok.decode(t, skip_special_tokens=True))[0] for t in tk]
            with open(out / f'{name}.jsonl', 'w') as f:
                for it, e in zip(items, outs):
                    f.write(json.dumps({'name': it['name'], 'expl': e}) + '\n')
            del lm; torch.cuda.empty_cache()
        readers[name] = [json.loads(x)['expl'] for x in open(out / f'{name}.jsonl')]
    # 3. judge (local Qwen3.6-27B, the repo's prompt)
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    tok.padding_side = 'left'
    jm = AutoModelForCausalLM.from_pretrained(SUBJECT, dtype=torch.bfloat16).to(dev).eval()
    system = SYSTEM
    summary = {}
    for rname, expls in readers.items():
        jp = out / f'judge_{rname}.jsonl'
        if not jp.exists():
            prompts = []
            for it, e in zip(items, expls):
                user = f"targets: {json.dumps(it['targets'])}\nThe samples below are free-text readouts.\n\n[position {it['pos']}]\n  {e}"
                prompts.append(tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                                       tokenize=False, add_generation_prompt=True, enable_thinking=False))
            verdicts = []
            for s in range(0, len(prompts), 32):
                enc = tok(prompts[s:s + 32], return_tensors='pt', padding=True).to(dev)
                with torch.no_grad():
                    g = jm.generate(**enc, do_sample=False, max_new_tokens=120, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for k, seq in enumerate(g[:, enc['input_ids'].shape[1]:]):
                    txt = tok.decode(seq, skip_special_tokens=True)
                    m = re.search(r'\{.*\}', txt, re.S)
                    try:
                        v = json.loads(m.group(0)) if m else {}
                    except Exception:
                        v = {}
                    e = expls[s + k]
                    ok = bool(v.get('expressed')) and bool(v.get('quote')) and v['quote'] in e
                    verdicts.append({'name': items[s + k]['name'], 'family': items[s + k]['family'], 'pass': ok, 'raw': txt[:300]})
            with open(jp, 'w') as f:
                for v in verdicts:
                    f.write(json.dumps(v) + '\n')
        vs = [json.loads(x) for x in open(jp)]
        fam = {f: float(np.mean([v['pass'] for v in vs if v['family'] == f])) for f in FAMS}
        summary[rname] = {'macro': float(np.mean(list(fam.values()))), **fam}
    tv = [json.loads(x)['pass'] for x in open(out / 'judge_teacher.jsonl')]
    rng = np.random.default_rng(0)
    for rname in readers:
        if rname == 'teacher':
            continue
        sv = [json.loads(x)['pass'] for x in open(out / f'judge_{rname}.jsonl')]
        diff = np.array(sv, float) - np.array(tv, float)
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        summary[rname]['minus_teacher'] = float(diff.mean())
        summary[rname]['minus_teacher_ci95'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    summary['_note'] = 'local judge (Qwen3.6-27B), one cell per item at L42, pure T=1; compare readers only'
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
