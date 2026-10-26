"""WorkspaceBench tiers 1+2 on three L42 readers (25 Oct 2026): teacher (ceselder/qwen3.6-27b-nla-rl iter_000400, the
bench's own NLA arm), GRPO526 and AGI-r100 (2B students). Cells come from scripts/wsbt12_render_cells_20261025.py
(the bench's render + position rules), so the box only runs forward passes and writes notes.
Sampling for every reader = the bench's in-house setting: T 1.0, top-k 64, top-p 0.95, max 256 new tokens; K samples per
cell (each sample is judged as its own k=1 run, so scores stay comparable to the published k=1 arms).
Stages:
  capture  --cells C --d D                  Qwen3.6-27B block-42 output at each cell -> D/vec.npy, D/units.json
  teacher  --d D --k K [--bs B]             -> D/notes_teacher.jsonl
  student  --d D --k K --model DIR --name N -> D/notes_N.jsonl
  export   --cells C --d D --out R          -> R/<reader>/s<j>/<family>.jsonl (readouts contract, layer 42)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (MAX_NEW, Adapter, av_prompt_ids, eos_ids, extract_expl,
                                                 left_pad_embeds, rl)

TOP_K, TOP_P = 64, 0.95


def units_of(cells):
    """Unique read prefixes (wrong_bridge and wrong_bridge_true share prompts)."""
    keys, unit = {}, []
    for c in cells:
        k = json.dumps(c['ids'][:c['idx'] + 1])
        if k not in keys:
            keys[k] = len(keys)
        unit.append(keys[k])
    return [json.loads(k) for k in keys], unit


def capture(a):
    from scripts.nladistil27b_extract_20261010 import Extractor
    cells = rl(a.cells)
    prefixes, unit = units_of(cells)
    t0 = time.time()
    ex = Extractor('cuda', layers=(42,))
    V = ex.last_states(prefixes, 65536)[42].float().numpy()
    d = Path(a.d); d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'vec.npy', V)
    json.dump({'unit': unit, 'n_units': len(prefixes)}, open(d / 'units.json', 'w'))
    print(json.dumps({'stage': 'capture', 'cells': len(cells), 'units': len(prefixes), 'norm_mean': float(np.linalg.norm(V, axis=1).mean()),
                      'sec': round(time.time() - t0)}), flush=True)


def filt(logits):
    """Temperature 1, then top-k 64, then top-p 0.95 (HF order)."""
    v, i = logits.topk(TOP_K, -1)
    p = torch.softmax(v, -1)
    c = p.cumsum(-1)
    p = torch.where(c - p > TOP_P, torch.zeros_like(p), p)
    j = torch.multinomial(p / p.sum(-1, keepdim=True), 1)
    return i.gather(-1, j)[:, 0]


@torch.no_grad()
def sample_embeds_f(model, embeds, attn, eos):
    B, dev = embeds.shape[0], embeds.device
    eos_t = torch.tensor(eos, device=dev)
    out = model(inputs_embeds=embeds, attention_mask=attn, use_cache=True, logits_to_keep=1)
    pkv, logits = out.past_key_values, out.logits[:, -1].float()
    toks = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(MAX_NEW):
        nxt = filt(logits)
        nxt = torch.where(done, torch.full_like(nxt, eos[0]), nxt)
        nc, dn = nxt.cpu().tolist(), done.cpu().tolist()
        for b in range(B):
            if not dn[b]:
                toks[b].append(nc[b])
        done = done | torch.isin(nxt, eos_t)
        if bool(done.all()):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=dev)], 1)
        out = model(input_ids=nxt[:, None], attention_mask=attn, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv, logits = out.past_key_values, out.logits[:, -1].float()
    return toks


def jobs(d, k):
    V = np.load(Path(d) / 'vec.npy')
    return V, [(u, s) for s in range(k) for u in range(V.shape[0])]


def save(path, recs):
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def summary(name, recs, t0):
    n = np.array([r['n_tok'] for r in recs])
    print(json.dumps({'stage': name, 'notes': len(recs), 'mean_tok': round(float(n.mean()), 1), 'at_cap': float((n >= MAX_NEW).mean()),
                      'parsed': float(np.mean([r['parsed'] for r in recs])), 'sec': round(time.time() - t0),
                      'example': recs[0]['text'][:300]}), flush=True)


def teacher(a):
    from scripts.nladistil27b_teacher_20261010 import HFTeacher, merge_teacher
    t0 = time.time()
    T = HFTeacher(merge_teacher(a.merged), 'cuda', a.bs)
    V, J = jobs(a.d, a.k)
    torch.manual_seed(a.seed)
    recs = []
    for s in range(0, len(J), a.bs):
        b = J[s:s + a.bs]
        T.state['v'] = torch.tensor(V[[u for u, _ in b]]).to('cuda')
        inp = torch.tensor([T.ids] * len(b), device='cuda')
        g = T.m.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_k=TOP_K, top_p=TOP_P,
                         max_new_tokens=MAX_NEW, eos_token_id=T.eos, pad_token_id=T.eos[0])
        T.state['v'] = None
        for (u, si), t in zip(b, g[:, len(T.ids):].cpu().tolist()):
            n = next((j + 1 for j, x in enumerate(t) if x in T.eos), len(t))
            e, ok = extract_expl(T.tok.decode(t[:n], skip_special_tokens=True))
            recs.append({'u': u, 's': si, 'text': e, 'parsed': ok, 'n_tok': n})
        print(json.dumps({'done': len(recs), 'of': len(J), 'sec': round(time.time() - t0)}), flush=True)
    save(Path(a.d) / 'notes_teacher.jsonl', recs)
    summary('teacher', recs, t0)


def student(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    md, dev = Path(a.model), 'cuda'
    tok = AutoTokenizer.from_pretrained(md / 'final_lm')
    lm = AutoModelForCausalLM.from_pretrained(md / 'final_lm', dtype=torch.bfloat16).to(dev).eval()
    emb = lm.get_input_embeddings(); dS = lm.config.hidden_size; eos = eos_ids(tok)
    ad = Adapter(dS, 4).to(dev); ad.load_state_dict(torch.load(md / 'final_adapter.pt', map_location=dev)); ad.eval()
    ids, pinj = av_prompt_ids(tok)
    pre = emb(torch.tensor(ids[:pinj], device=dev)); suf = emb(torch.tensor(ids[pinj + 1:], device=dev))
    V, J = jobs(a.d, a.k)
    torch.manual_seed(a.seed)
    recs = []
    with torch.no_grad():
        for s in range(0, len(J), a.bs):
            b = J[s:s + a.bs]
            A = ad(torch.tensor(V[[u for u, _ in b]], device=dev)).to(emb.weight.dtype)
            E, M = left_pad_embeds([torch.cat([pre, A[i], suf]) for i in range(A.shape[0])], dS, emb.weight.dtype, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tk = sample_embeds_f(lm, E, M, eos)
            for (u, si), t in zip(b, tk):
                e, ok = extract_expl(tok.decode(t, skip_special_tokens=True))
                recs.append({'u': u, 's': si, 'text': e, 'parsed': ok, 'n_tok': len(t)})
    save(Path(a.d) / f'notes_{a.name}.jsonl', recs)
    summary(a.name, recs, t0)


def export(a):
    cells = rl(a.cells)
    unit = json.load(open(Path(a.d) / 'units.json'))['unit']
    for f in sorted(Path(a.d).glob('notes_*.jsonl')):
        name = f.stem.removeprefix('notes_')
        notes = {(r['u'], r['s']): r['text'] for r in rl(f)}
        ks = sorted({s for _, s in notes})
        for s in ks:
            od = Path(a.out) / name / f's{s}'; od.mkdir(parents=True, exist_ok=True)
            by = {}
            for c, u in zip(cells, unit):
                by.setdefault(c['family'], []).append({'id': c['id'], 'layer': 42, 'pos': c['pos'], 'token': c['token'],
                                                       'samples': [notes[(u, s)]]})
            for fam, rows in by.items():
                save(od / f'{fam}.jsonl', rows)
        print(json.dumps({'export': name, 'samples': ks}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['capture', 'teacher', 'student', 'export'])
    ap.add_argument('--cells'); ap.add_argument('--d'); ap.add_argument('--out')
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--bs', type=int, default=192)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model'); ap.add_argument('--name')
    ap.add_argument('--merged', default='/workspace/teacher_merged')
    a = ap.parse_args()
    {'capture': capture, 'teacher': teacher, 'student': student, 'export': export}[a.stage](a)


if __name__ == '__main__':
    main()
