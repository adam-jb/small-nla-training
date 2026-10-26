"""The 27B NLA teacher writes one explanation per vector, keeping its top-20 log-probs at every token (10 Oct 2026).

Teacher = av_base + av_rl_adapters/iter_000400 merged once into --merged (bf16), pure T=1 sampling, max 256 new tokens.
Backends:
  vllm: EasyNLA's path (vllm 0.19 + vllm-lens 1.1 + EasyNLA's norm-match patch): a per-request SteeringVector at
        layer 1, norm_match=True, at the marker position = h + |h| v/|v|; prefix caching OFF (prompts are identical).
  hf:   transformers generate with the same Karvonen hook (slow fallback, and the cross-check of the vLLM path).
Output per shard: OUT/{set}/T_{k:03d}.jsonl ({'i','raw','expl','parsed','n_tok','finish'}) and T_{k:03d}_top.npz
(off, tok int32, tk_ids int32 [n,20], tk_lp fp16 [n,20]). Resumable by shard. --samples 2 writes a second sample
with a different seed into OUT/{set}_s{seed}.
Usage: python scripts/nladistil27b_teacher_20261010.py --vec DIR --set train --out DIR --backend vllm [--rows 0:263000]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (AV_ADAPTER, LAYER, MAX_NEW, SMOKE, SUBJECT, av_prompt_ids, eos_ids,
                                                 extract_expl, karvonen_hook, rl, teacher_dir, check_meta)
from scripts.nladistil27b_extract_20261010 import load_set

TOPK = 20


def merge_teacher(dst):
    """av_base + iter_000400 LoRA -> one bf16 HF folder (done once per box)."""
    dst = Path(dst)
    if (dst / 'merged_ok.json').exists():
        return dst
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    td = teacher_dir()
    check_meta(td / 'av_base' / 'nla_meta.yaml')
    base = AutoModelForCausalLM.from_pretrained(td / 'av_base', dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(base, td / AV_ADAPTER).merge_and_unload()
    m.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(td / 'av_base').save_pretrained(dst)
    for f in ('nla_meta.yaml', 'chat_template.jinja'):
        if (td / 'av_base' / f).exists():
            (dst / f).write_bytes((td / 'av_base' / f).read_bytes())
    json.dump({'adapter': AV_ADAPTER}, open(dst / 'merged_ok.json', 'w'))
    return dst


class VLLMTeacher:
    def __init__(self, path, mem, max_len=1024, ssm_dtype='auto', max_seqs=256, row_seeds=True):
        from vllm import LLM
        from transformers import AutoTokenizer
        # tokenizer from the base model: the merged folder was saved by transformers 5.x ("TokenizersBackend"), which
        # the vLLM venv's transformers 4.57 cannot load; the vocab is identical (checked 10 Oct)
        self.tok = AutoTokenizer.from_pretrained(SUBJECT)
        self.ids, self.pos = av_prompt_ids(self.tok)
        self.row_seeds = row_seeds  # per-request seeds: vLLM samples those requests one by one with their own generators
        self.llm = LLM(model=str(path), tokenizer=SUBJECT, dtype='bfloat16', gpu_memory_utilization=mem,
                       max_model_len=max_len, enable_prefix_caching=False, max_logprobs=TOPK,
                       mamba_ssm_cache_dtype=ssm_dtype, max_num_seqs=max_seqs)

    def write(self, V, seed):
        from vllm import SamplingParams, TokensPrompt
        from vllm_lens import SteeringVector
        prompts, params = [], []
        for k in range(V.shape[0]):
            sv = SteeringVector(activations=torch.tensor(V[k]).view(1, 1, -1).float(), layer_indices=[1], scale=1.0,
                                norm_match=True, position_indices=[self.pos])
            prompts.append(TokensPrompt(prompt_token_ids=self.ids))
            params.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=MAX_NEW, logprobs=TOPK,
                                         seed=seed * 1_000_003 + k if self.row_seeds else None, extra_args={'apply_steering_vectors': [sv]}))
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = np.zeros((len(c.token_ids), TOPK), np.int32); lp = np.full((len(c.token_ids), TOPK), -1e4, np.float16)
            for t, d in enumerate(c.logprobs or []):
                top = sorted(d.items(), key=lambda kv: kv[1].rank)[:TOPK]
                for j, (tid, L) in enumerate(top):
                    ids[t, j] = tid; lp[t, j] = L.logprob
            res.append((c.text, list(c.token_ids), c.finish_reason, ids, lp))
        return res


class HFTeacher:
    def __init__(self, path, dev, bs=64):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.bs = bs
        p = SUBJECT if SMOKE else path
        self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to(dev).eval()
        self.ids, self.pos = av_prompt_ids(self.tok)
        self.state = {'pos': self.pos}
        karvonen_hook(self.m, self.state)
        self.dev, self.eos = dev, eos_ids(self.tok)

    @torch.no_grad()
    def write(self, V, seed, bs=None):
        """Batched HF sampling; top-20 log-probs are kept per step by a logits processor (no full-vocab score store)."""
        from transformers import LogitsProcessor, LogitsProcessorList
        bs = bs or self.bs
        res = []
        torch.manual_seed(seed)

        class Top(LogitsProcessor):
            def __init__(self):
                self.ids, self.lp = [], []

            def __call__(self, input_ids, scores):
                v, i = torch.log_softmax(scores.float(), -1).topk(TOPK, -1)
                self.ids.append(i.to(torch.int32)); self.lp.append(v.half())
                return scores

        for s in range(0, V.shape[0], bs):
            v = torch.tensor(V[s:s + bs]).to(self.dev)
            self.state['v'] = v
            inp = torch.tensor([self.ids] * v.shape[0], device=self.dev)
            top = Top()
            g = self.m.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_k=0,
                                top_p=1.0, max_new_tokens=MAX_NEW, logits_processor=LogitsProcessorList([top]),
                                eos_token_id=self.eos, pad_token_id=self.eos[0])
            self.state['v'] = None
            seqs = g[:, len(self.ids):].cpu()
            ti = torch.stack(top.ids, 1).cpu(); tv = torch.stack(top.lp, 1).cpu()  # [B, T, 20]
            for b in range(seqs.shape[0]):
                t = seqs[b].tolist()
                n = next((j + 1 for j, x in enumerate(t) if x in self.eos), len(t))
                t = t[:n]
                res.append((self.tok.decode(t, skip_special_tokens=True), t, 'stop' if n < MAX_NEW else 'length',
                            ti[b, :n].numpy(), tv[b, :n].numpy()))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True, help='extraction dir')
    ap.add_argument('--set', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', choices=['vllm', 'hf'], default='vllm')
    ap.add_argument('--merged', default='/home/ubuntu/teacher_merged')
    ap.add_argument('--rows', default='', help='a:b slice of the set')
    ap.add_argument('--shard', type=int, default=1000, help='rows per saved file (small: spot VMs get pre-empted)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu-mem', type=float, default=0.85)
    ap.add_argument('--hf-bs', type=int, default=256)
    ap.add_argument('--vllm-ssm-dtype', default='auto', help="GDN recurrent-state dtype in vLLM ('bfloat16' halves its memory)")
    ap.add_argument('--vllm-max-seqs', type=int, default=256)
    ap.add_argument('--vllm-max-len', type=int, default=1024)
    ap.add_argument('--vllm-no-row-seeds', action='store_true', help='one seed per call, not per row (faster sampling)')
    a = ap.parse_args()
    t0 = time.time()
    V = load_set(a.vec, a.set, LAYER)
    rows = rl(Path(a.vec) / a.set / 'rows.jsonl')
    lo, hi = (int(x) for x in a.rows.split(':')) if a.rows else (0, len(rows))
    od = Path(a.out) / (a.set if a.seed == 0 else f'{a.set}_s{a.seed}'); od.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    path = None if SMOKE else merge_teacher(a.merged)
    T = VLLMTeacher(path, a.gpu_mem, a.vllm_max_len, a.vllm_ssm_dtype, a.vllm_max_seqs,
                                not a.vllm_no_row_seeds) if a.backend == 'vllm' else HFTeacher(path, dev, a.hf_bs)
    print(json.dumps({'ready': True, 'sec': round(time.time() - t0), 'n': hi - lo}), flush=True)
    done = set()  # rows already in any saved file (old 10k files T_000.. or small files T_r0000000..)
    for f in od.glob('T_*.jsonl'):
        done.update(r['i'] for r in rl(f))
    missing = [i for i in range(lo, hi) if i not in done]  # label only these: shards of any size never overlap
    for c in range(0, len(missing), a.shard):
        idx = missing[c:c + a.shard]
        s, e = idx[0], idx[-1] + 1
        k = f'r{s:07d}'
        t1 = time.time()
        res = T.write(V[idx], a.seed * 1_000_000 + s)
        recs, toks, tids, tlps, off = [], [], [], [], [0]
        for i, (raw, t, fin, ids, lp) in zip(idx, res):
            expl, ok = extract_expl(raw)
            recs.append({'i': i, 'raw': raw, 'expl': expl, 'parsed': ok, 'n_tok': len(t), 'finish': fin, 'backend': a.backend})
            toks.append(np.array(t, np.int32)); tids.append(ids); tlps.append(lp); off.append(off[-1] + len(t))
        np.savez(od / f'T_{k}_top.npz', off=np.array(off, np.int64), tok=np.concatenate(toks),
                 tk_ids=np.concatenate(tids), tk_lp=np.concatenate(tlps))
        with open(od / f'T_{k}.tmp', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        (od / f'T_{k}.tmp').rename(od / f'T_{k}.jsonl')
        nt = off[-1]
        print(json.dumps({'shard': k, 'rows': e, 'of': hi, 'tok_s': round(nt / (time.time() - t1)), 'mean_tok': round(nt / len(recs), 1),
                          'parsed': float(np.mean([r['parsed'] for r in recs])),
                          'at_cap': float(np.mean([r['n_tok'] >= MAX_NEW for r in recs])), 'sec': round(time.time() - t0),
                          'example': recs[0]['expl'][:300]}), flush=True)


if __name__ == '__main__':
    main()
