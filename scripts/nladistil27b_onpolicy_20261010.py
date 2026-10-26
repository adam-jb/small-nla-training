"""One on-policy distillation round for the 27B NLA student (10 Oct 2026), as on the 7B (op30k / op200k rounds).

sample: the student writes one T=1 explanation per pool vector -> OUT/{name}/S_{k:03d}.jsonl ({'i', 'toks'}).
score:  the teacher (merged av_base + iter_000400, HF, Karvonen hook) reads the vector plus the student's tokens
        (teacher-forced) and gives its top-20 log-probs at every student token -> OUT/{name}/T_{k:03d}.jsonl +
        T_{k:03d}_top.npz in the teacher format, so the trainer can use it with --teacher-sets {name} --soft-w 1.0.
Usage: python scripts/nladistil27b_onpolicy_20261010.py sample --vec DIR --init RUN --rows 0:100000 --out DIR --name op_r1
       python scripts/nladistil27b_onpolicy_20261010.py score  --vec DIR --out DIR --name op_r1
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (LAYER, MAX_NEW, SMOKE, SUBJECT, Adapter, av_prompt_ids, eos_ids,
                                                 karvonen_hook, left_pad_embeds, rl, sample_embeds)
from scripts.nladistil27b_extract_20261010 import load_set
from scripts.nladistil27b_teacher_20261010 import merge_teacher, TOPK


def do_sample(a, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    init = Path(a.init)
    tok = AutoTokenizer.from_pretrained(init / 'final_lm')
    lm = AutoModelForCausalLM.from_pretrained(init / 'final_lm', dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
    emb, d, eos = lm.get_input_embeddings(), lm.config.hidden_size, eos_ids(tok)
    ad = Adapter(d, 4).to(dev); ad.load_state_dict(torch.load(init / 'final_adapter.pt', map_location=dev)); ad.eval()
    ids, p = av_prompt_ids(tok)
    pre, suf = emb(torch.tensor(ids[:p], device=dev)), emb(torch.tensor(ids[p + 1:], device=dev))
    V = load_set(a.vec, a.set, LAYER)
    lo, hi = (int(x) for x in a.rows.split(':'))
    od = Path(a.out) / a.name; od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in range(lo, hi, a.shard):
        k = s // a.shard
        if (od / f'S_{k:03d}.jsonl').exists():
            continue
        recs = []
        for b in range(s, min(s + a.shard, hi), a.bs):
            idx = list(range(b, min(b + a.bs, s + a.shard, hi)))
            with torch.no_grad():
                A = ad(torch.tensor(V[idx], device=dev)).to(emb.weight.dtype)
                P = [torch.cat([pre, A[j], suf]) for j in range(len(idx))]
                E, M = left_pad_embeds(P, d, P[0].dtype, dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                    toks = sample_embeds(lm, E, M, eos, max_new=MAX_NEW)
            recs += [{'i': i, 'toks': t if t else [eos[0]]} for i, t in zip(idx, toks)]
        with open(od / f'S_{k:03d}.tmp', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        (od / f'S_{k:03d}.tmp').rename(od / f'S_{k:03d}.jsonl')
        print(json.dumps({'sampled_shard': k, 'rows': len(recs), 'mean_tok': float(np.mean([len(r['toks']) for r in recs])),
                          'sec': round(time.time() - t0)}), flush=True)


def do_score(a, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = SUBJECT if SMOKE else merge_teacher(a.merged)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    ids, pos = av_prompt_ids(tok)
    state = {'pos': pos}
    karvonen_hook(m, state)
    V = load_set(a.vec, a.set, LAYER)
    od = Path(a.out) / a.name
    t0 = time.time()
    for f in sorted(od.glob('S_*.jsonl')):
        k = f.stem.split('_')[1]
        if (od / f'T_{k}.jsonl').exists():
            continue
        recs = rl(f)
        toks_all, tid_all, tlp_all, off = [], [], [], [0]
        for b in range(0, len(recs), a.bs):
            ch = recs[b:b + a.bs]
            seqs = [ids + r['toks'][:-1] if len(r['toks']) > 1 else ids for r in ch]
            L = max(map(len, seqs))
            x = torch.zeros(len(ch), L, dtype=torch.long); att = torch.zeros(len(ch), L, dtype=torch.long)
            for j, s_ in enumerate(seqs):
                x[j, :len(s_)] = torch.tensor(s_); att[j, :len(s_)] = 1
            state['v'] = torch.tensor(V[[r['i'] for r in ch]]).to(dev)
            with torch.no_grad():
                h = m.model(input_ids=x.to(dev), attention_mask=att.to(dev), use_cache=False).last_hidden_state
                for j, r in enumerate(ch):
                    n = len(r['toks'])
                    hs = h[j, len(ids) - 1:len(ids) - 1 + n]
                    lp = torch.log_softmax(m.lm_head(hs).float(), -1)
                    v_, i_ = lp.topk(TOPK, -1)
                    toks_all.append(np.array(r['toks'], np.int32)); tid_all.append(i_.cpu().numpy().astype(np.int32))
                    tlp_all.append(v_.cpu().numpy().astype(np.float16)); off.append(off[-1] + n)
            state['v'] = None
        np.savez(od / f'T_{k}_top.npz', off=np.array(off, np.int64), tok=np.concatenate(toks_all),
                 tk_ids=np.concatenate(tid_all), tk_lp=np.concatenate(tlp_all))
        with open(od / f'T_{k}.tmp', 'w') as fo:
            for r in recs:
                fo.write(json.dumps({'i': r['i'], 'n_tok': len(r['toks'])}) + '\n')
        (od / f'T_{k}.tmp').rename(od / f'T_{k}.jsonl')
        print(json.dumps({'scored_shard': k, 'rows': len(recs), 'tokens': off[-1], 'sec': round(time.time() - t0)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sample', 'score'])
    ap.add_argument('--vec', required=True)
    ap.add_argument('--set', default='train')
    ap.add_argument('--init', default='')
    ap.add_argument('--rows', default='0:100000')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='op_r1')
    ap.add_argument('--merged', default='/home/ubuntu/teacher_merged')
    ap.add_argument('--shard', type=int, default=10000)
    ap.add_argument('--bs', type=int, default=256)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    (do_sample if a.cmd == 'sample' else do_score)(a, dev)


if __name__ == '__main__':
    main()
