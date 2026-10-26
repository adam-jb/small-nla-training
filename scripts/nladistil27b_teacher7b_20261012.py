"""The released Qwen2.5-7B L20 NLA (kitft) writes one explanation per vector, saved in the 27B teacher's file format (12 Oct).

Teacher-size comparison: the 7B and 27B NLAs both teach the same Qwen3.5-2B student. Qwen2.5 and Qwen3.5 tokenizers share
almost no ids (281 of 151,665), so the 7B's top-20 log-probs are useless to the student: we keep the text only and
re-spell it in Qwen3.5 tokens (+ <|im_end|> if the teacher stopped). The _top.npz files hold a one-hot "top-20" (the
token itself, log-prob 0, the rest -1e4), so the trainer's soft term equals plain CE: these are hard labels.
Generation as nladistil_teacher_20261003.py: vector scaled to L2 150 replaces the marker embedding, pure T=1 sampling.
Vectors from an NLA_PROFILE=7b extraction (vec7b/{set}/L20_*.npy). Output OUT/{set}/T_r{start:07d}.jsonl + _top.npz.
Resumable: only rows missing from saved files are labelled.
Usage: NLA_PROFILE=7b python scripts/nladistil27b_teacher7b_20261012.py --vec DIR --set train --out DIR [--rows 0:263000]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get('NLA_PROFILE') == '7b', 'run with NLA_PROFILE=7b'
from scripts.nladistil27b_common_20261010 import LAYER, MAX_NEW, extract_expl, rl
from scripts.nladistil27b_extract_20261010 import load_set
import scripts.nladistil_common_20261003 as c7

TOPK, STUDENT = 20, 'Qwen/Qwen3.5-2B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--set', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='')
    ap.add_argument('--shard', type=int, default=4096)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    t0 = time.time()
    dev = 'cuda'
    path = snapshot_download(c7.AV_REPO)
    c7.check_meta(Path(path) / 'nla_meta.yaml')
    tok7 = AutoTokenizer.from_pretrained(path)
    tok35 = AutoTokenizer.from_pretrained(STUDENT)
    end35 = tok35.convert_tokens_to_ids('<|im_end|>')
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    ids, p = c7.av_prompt_ids(tok7)
    base = model.get_input_embeddings()(torch.tensor(ids, device=dev))
    V = load_set(a.vec, a.set, LAYER)
    nrows = len(rl(Path(a.vec) / a.set / 'rows.jsonl'))
    assert V.shape == (nrows, c7.D), (V.shape, nrows)
    lo, hi = (int(x) for x in a.rows.split(':')) if a.rows else (0, nrows)
    od = Path(a.out) / (a.set if a.seed == 0 else f'{a.set}_s{a.seed}'); od.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in od.glob('T_*.jsonl'):
        done.update(r['i'] for r in rl(f))
    missing = [i for i in range(lo, hi) if i not in done]
    print(json.dumps({'ready': True, 'sec': round(time.time() - t0), 'n': len(missing)}), flush=True)
    for c in range(0, len(missing), a.shard):
        idx = missing[c:c + a.shard]
        t1, recs, toks, nt7 = time.time(), [], [], 0
        torch.manual_seed(a.seed * 1_000_000 + idx[0])
        for s in range(0, len(idx), a.bs):
            sub = idx[s:s + a.bs]
            v = c7.normalise(torch.tensor(V[sub], device=dev).float(), c7.INJ_SCALE).to(torch.bfloat16)
            E = base.unsqueeze(0).repeat(len(sub), 1, 1).clone()
            E[:, p] = v
            M = torch.ones(E.shape[:2], dtype=torch.long, device=dev)
            with torch.no_grad():
                out, _ = c7.sample(model, E, M, max_new=MAX_NEW, temperature=1.0)
            for i, t in zip(sub, out):
                stop = bool(t) and t[-1] == c7.EOS_ID
                raw = tok7.decode(t, skip_special_tokens=True)
                t35 = tok35(raw, add_special_tokens=False)['input_ids'] + ([end35] if stop else [])
                expl, ok = extract_expl(raw)
                recs.append({'i': i, 'raw': raw, 'expl': expl, 'parsed': ok, 'n_tok': len(t35), 'n_tok_teacher': len(t),
                             'finish': 'stop' if stop else 'length', 'backend': 'hf7b'})
                toks.append(np.array(t35, np.int32)); nt7 += len(t)
        off = np.cumsum([0] + [len(t) for t in toks]).astype(np.int64)
        tok = np.concatenate(toks)
        tk_ids = np.repeat(tok[:, None], TOPK, 1).astype(np.int32)
        tk_lp = np.full((len(tok), TOPK), -1e4, np.float16); tk_lp[:, 0] = 0
        k = f'r{idx[0]:07d}'
        np.savez(od / f'T_{k}_top.npz', off=off, tok=tok, tk_ids=tk_ids, tk_lp=tk_lp)
        with open(od / f'T_{k}.tmp', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        (od / f'T_{k}.tmp').rename(od / f'T_{k}.jsonl')
        print(json.dumps({'shard': k, 'rows': idx[-1] + 1, 'of': hi, 'tok_s_teacher': round(nt7 / (time.time() - t1)),
                          'mean_tok_teacher': round(nt7 / len(recs), 1), 'mean_tok_qwen35': round(off[-1] / len(recs), 1),
                          'parsed': float(np.mean([r['parsed'] for r in recs])),
                          'at_cap': float(np.mean([r['finish'] == 'length' for r in recs])),
                          'sec': round(time.time() - t0), 'example': recs[0]['expl'][:300]}), flush=True)


if __name__ == '__main__':
    main()
