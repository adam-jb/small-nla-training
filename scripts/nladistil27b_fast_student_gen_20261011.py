"""Fast sampling from the soft-token student with vLLM (prompt embeddings + CUDA graphs) (11 Oct 2026).

Why: GRPO, the on-policy round and every eval sample from the 2B student with `common.sample_embeds`, a Python loop
that syncs GPU->CPU every token. On a 2B model that is launch-bound. vLLM takes the same prompt embeddings
(pre + adapter(v) + suf) directly, so no steering hook is needed and CUDA graphs stay on.

FastGen(student_repo): loads the student's original (multimodal) checkpoint in vLLM, in-process (so weights can be
swapped from a trainer with load_lm_state), vllm-lens plugin kept out (it forces eager mode).
  .generate(embeds [N, T, d] or list of [T, d], max_new, temperature, seed) -> token lists (EOS kept, as sample_embeds)
  .load_lm_state(hf_state_dict) -> copy a trained Qwen3_5ForCausalLM's weights in (names model.* -> language_model.*)

Bench (two envs: the HF baseline needs transformers 5.x, vLLM 0.19 lives in its own venv):
  /data/venv/bin/python scripts/nladistil27b_fast_student_gen_20261011.py hf --vec /data/vec27b --out /data/fastgen
  /data/envs/vllm-lens/bin/python scripts/nladistil27b_fast_student_gen_20261011.py vllm --out /data/fastgen
Both time 1,020 held-out prompts x 182 forced tokens (no EOS stop) and greedy-decode 32 prompts for a match check.
"""
import argparse, json, os, sys, time
from pathlib import Path
import torch

os.environ['VLLM_PLUGINS'] = ''  # keep vllm-lens out: it forces enforce_eager on every engine
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')  # engine in this process: weights can be swapped
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
STUDENT = 'Qwen/Qwen3.5-2B'
N_TOK, N_GREEDY = 182, 32


def _drop_mrope(cfg):
    """hf_overrides hook: remove the M-RoPE keys (a dict override would merge, not delete)."""
    for c in {id(cfg): cfg, id(cfg.get_text_config()): cfg.get_text_config()}.values():
        rp = getattr(c, 'rope_parameters', None)
        if rp and 'mrope_section' in rp:
            c.rope_parameters = {k: v for k, v in rp.items() if not k.startswith('mrope')}
    return cfg


class FastGen:
    def __init__(self, repo=STUDENT, mem=0.30, max_seqs=512, max_len=512):
        from vllm import LLM
        # M-RoPE needs token ids, so vLLM refuses prompt embeddings with it. Text-only, M-RoPE puts the same position
        # on all three axes, which is exactly 1-D RoPE with the same frequencies: drop mrope_section (greedy check below).
        self.llm = LLM(model=repo, dtype='bfloat16', gpu_memory_utilization=mem, max_model_len=max_len,
                       max_num_seqs=max_seqs, enable_prompt_embeds=True, enable_prefix_caching=False,
                       mamba_ssm_cache_dtype='float16', limit_mm_per_prompt={'image': 0, 'video': 0},
                       hf_overrides=_drop_mrope)

    def generate(self, embeds, max_new=256, temperature=1.0, seed=0, ignore_eos=False):
        from vllm import SamplingParams
        sp = [SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_new, seed=seed * 1_000_003 + k,
                             ignore_eos=ignore_eos, min_tokens=max_new if ignore_eos else 0)
              for k in range(len(embeds))]
        outs = self.llm.generate([{'prompt_embeds': e.detach().to('cpu', torch.bfloat16)} for e in embeds], sp,
                                 use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]

    def load_lm_state(self, sd):
        """sd: state_dict of a transformers Qwen3_5ForCausalLM (model.*, lm_head tied)."""
        ws = [(('model.language_model.' + k[len('model.'):]) if k.startswith('model.') else k, v.detach())
              for k, v in sd.items() if not k.startswith('lm_head')]
        self.llm.apply_model(lambda m: m.load_weights(iter(ws)))


def build_embeds(vec, n):
    """Prompt embeddings for n held-out vectors through a freshly initialised adapter (speed test only)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.nladistil27b_common_20261010 import Adapter, av_prompt_ids, LAYER
    from scripts.nladistil27b_extract_20261010 import load_set
    tok = AutoTokenizer.from_pretrained(STUDENT)
    lm = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).cuda().eval()
    emb = lm.get_input_embeddings()
    ids, p = av_prompt_ids(tok)
    torch.manual_seed(0)
    ad = Adapter(lm.config.hidden_size, 4).cuda()
    ad.init_to(emb.weight.float().std().item())
    V = torch.tensor(load_set(vec, 'heldout', LAYER)[:n]).cuda()
    with torch.no_grad():
        A = ad(V).to(torch.bfloat16)
        pre, suf = emb(torch.tensor(ids[:p]).cuda()), emb(torch.tensor(ids[p + 1:]).cuda())
        E = torch.stack([torch.cat([pre, A[b], suf]) for b in range(n)])
    return lm, tok, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=['hf', 'vllm'])
    ap.add_argument('--vec', default='/data/vec27b')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1020)
    ap.add_argument('--bs', type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.which == 'hf':
        from scripts.nladistil27b_common_20261010 import sample_embeds
        lm, tok, E = build_embeds(a.vec, a.n)
        torch.save(E.cpu(), out / 'embeds.pt')
        M = torch.ones(E.shape[:2], dtype=torch.long, device='cuda')
        with torch.no_grad():
            sample_embeds(lm, E[:8], M[:8], [-1], max_new=8)  # warm-up
            torch.cuda.synchronize(); t = time.time(); n = 0
            for s in range(0, a.n, a.bs):
                n += sum(len(x) for x in sample_embeds(lm, E[s:s + a.bs], M[s:s + a.bs], [-1], max_new=N_TOK))
            torch.cuda.synchronize(); dt = time.time() - t
            g = sample_embeds(lm, E[:N_GREEDY], M[:N_GREEDY], [-1], max_new=64, temperature=1e-6)
        json.dump(g, open(out / 'greedy_hf.json', 'w'))
        print(json.dumps({'which': 'hf', 'tok_s': round(n / dt), 'sec': round(dt), 'tokens': n, 'bs': a.bs,
                          'mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)
    else:
        E = torch.load(out / 'embeds.pt')
        t0 = time.time(); fg = FastGen(); load = time.time() - t0
        fg.generate(list(E[:16]), max_new=8)  # warm-up
        t = time.time()
        res = fg.generate(list(E[:a.n]), max_new=N_TOK, ignore_eos=True)
        dt = time.time() - t
        n = sum(len(x) for x in res)
        g = fg.generate(list(E[:N_GREEDY]), max_new=64, temperature=0.0, ignore_eos=True)
        json.dump(g, open(out / 'greedy_vllm.json', 'w'))
        rec = {'which': 'vllm', 'tok_s': round(n / dt), 'sec': round(dt), 'tokens': n, 'load_sec': round(load)}
        hp = out / 'greedy_hf.json'
        if hp.exists():
            h = json.load(open(hp))
            pre = [next((j for j, (x, y) in enumerate(zip(a_, b_)) if x != y), min(len(a_), len(b_)))
                   for a_, b_ in zip(h, g)]
            rec.update({'greedy_first5_same': sum(p >= 5 for p in pre), 'greedy_prefix_median': sorted(pre)[len(pre) // 2],
                        'of': len(pre)})
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
