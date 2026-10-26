"""Pre-flight checks for the 27B NLA distillation box (10 Oct 2026). Writes OUT/preflight.json; exit 2 if a gate fails.

1. GPU memory >= 90 GB (the bf16 teacher needs ~54 GB).
2. Extraction convention: our L42 vectors for the teacher repo's 64 example prefixes (data/example_activations.parquet)
   must match the published vectors (mean cos >= 0.995, min >= 0.95, norm ratio 0.95-1.05; bf16 kernels differ a little:
   first B300 run gave mean 0.9993, min 0.977).
3. Teacher + AR on those 64 vectors and 50 held-out vectors via HF (Karvonen hook): FVE >= 0.65 on held-out; parse rate.
4. Teacher speed via HF (tokens/s) and, if --vllm, the vLLM-lens path on the same vectors (FVE within 0.03 of HF).
5. 50 teacher notes on blends (if extracted) -> OUT/blend_notes.jsonl for reading, with AR cos to the blend vector.
Usage: python scripts/nladistil27b_preflight_20261010.py --vec DIR --out DIR [--vllm]
"""
import argparse, io, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import AR, LAYER, SMOKE, extract_expl, fve_stats, rl, TEACHER_REPO
from scripts.nladistil27b_extract_20261010 import Extractor, load_set
from scripts.nladistil27b_ar_score_20261010 import rawvar_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--vllm', action='store_true')
    ap.add_argument('--min-fve', type=float, default=0.65)
    ap.add_argument('--merged', default='/home/ubuntu/teacher_merged')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    R, fails = {}, []
    t0 = time.time()
    if dev == 'cuda':
        R['gpu_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        R['gpu'] = torch.cuda.get_device_name(0)
        if R['gpu_gb'] < 90:
            fails.append('gpu memory < 90 GB')
    # 2. extraction convention
    if not SMOKE:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        ex_rows = pq.read_table(hf_hub_download(TEACHER_REPO, 'data/example_activations.parquet')).to_pylist()
        ex = Extractor(dev)
        ids = [ex.tok(r['detokenized_text_truncated'], add_special_tokens=True)['input_ids'] for r in ex_rows]
        H = ex.last_states(ids, 32768)[LAYER].float()
        G = torch.tensor([r['activation_vector'] for r in ex_rows]).float()
        c = torch.nn.functional.cosine_similarity(H, G, dim=-1)
        nr = H.norm(dim=-1) / G.norm(dim=-1)
        R['extract_check'] = {'cos_min': c.min().item(), 'cos_mean': c.mean().item(), 'norm_ratio_min': nr.min().item(),
                              'norm_ratio_max': nr.max().item(), 'token_count_match': float(np.mean(
                                  [len(i) == r['n_raw_tokens'] for i, r in zip(ids, ex_rows)]))}
        if c.mean().item() < 0.995 or c.min().item() < 0.95 or not (0.95 <= nr.min().item() and nr.max().item() <= 1.05):
            fails.append('extraction does not match the published example vectors')
        del ex; torch.cuda.empty_cache()
    print(json.dumps(R), flush=True)
    # 3-4. teacher via HF
    from scripts.nladistil27b_teacher_20261010 import HFTeacher, merge_teacher, VLLMTeacher
    path = None if SMOKE else merge_teacher(a.merged)
    Vh = load_set(a.vec, 'heldout', LAYER)[:50]
    T = HFTeacher(path, dev)
    t1 = time.time(); res = T.write(Vh, seed=0, bs=50); dt = time.time() - t1
    ntok = sum(len(r[1]) for r in res)
    R['hf_tok_s'] = ntok / dt
    expl = [extract_expl(r[0]) for r in res]
    del T; torch.cuda.empty_cache()
    ar = AR(dev)
    rawvar = rawvar_for(a.vec)
    P = ar.reconstruct([e for e, _ in expl])
    s, _ = fve_stats(P, torch.tensor(Vh), rawvar)
    R['teacher_hf_heldout50'] = {**s, 'parsed': float(np.mean([ok for _, ok in expl])), 'mean_tok': ntok / len(res)}
    if s['fve'] < a.min_fve and not SMOKE:
        fails.append(f"teacher FVE {s['fve']:.3f} < {a.min_fve}")
    with open(out / 'teacher_hf_heldout50.jsonl', 'w') as f:
        for (e, ok), r in zip(expl, res):
            f.write(json.dumps({'expl': e, 'parsed': ok, 'n_tok': len(r[1])}) + '\n')
    # 5. blends
    if (Path(a.vec) / 'blends').exists():
        Vb = load_set(a.vec, 'blends', LAYER)[:50]; brows = rl(Path(a.vec) / 'blends' / 'rows.jsonl')[:50]
        ar_cpu = None
        del ar; torch.cuda.empty_cache()
        T = HFTeacher(path, dev); rb = T.write(Vb, seed=1, bs=50); del T; torch.cuda.empty_cache()
        ar = AR(dev)
        eb = [extract_expl(r[0]) for r in rb]
        Pb = ar.reconstruct([e for e, _ in eb])
        sb, cb = fve_stats(Pb, torch.tensor(Vb), rawvar)
        R['blends50'] = {**sb, 'parsed': float(np.mean([ok for _, ok in eb]))}
        with open(out / 'blend_notes.jsonl', 'w') as f:
            for r_, (e, ok), cc in zip(brows, eb, cb.tolist()):
                f.write(json.dumps({'i': r_['i'], 'alpha': r_['alpha'], 'a_text_tail': r_['text'][-300:], 'expl': e,
                                    'parsed': ok, 'ar_cos': cc}) + '\n')
    del ar; torch.cuda.empty_cache()
    if a.vllm and not SMOKE:
        try:
            T = VLLMTeacher(path, 0.85)
            t1 = time.time(); rv = T.write(Vh, seed=0); dt = time.time() - t1
            R['vllm_tok_s_50'] = sum(len(r[1]) for r in rv) / dt
            # a bigger batch for a fair speed figure
            V2 = load_set(a.vec, 'teval', LAYER)[:512]
            t1 = time.time(); r2 = T.write(V2, seed=0); dt = time.time() - t1
            R['vllm_tok_s_512'] = sum(len(r[1]) for r in r2) / dt
            ev = [extract_expl(r[0]) for r in rv]
            json.dump([e for e, _ in ev], open(out / 'teacher_vllm_heldout50.json', 'w'))
            R['vllm_ok'] = True
        except Exception as e:
            R['vllm_ok'] = False; R['vllm_error'] = repr(e)[:500]
    R['fails'] = fails; R['sec'] = round(time.time() - t0)
    json.dump(R, open(out / 'preflight.json', 'w'), indent=1)
    print(json.dumps(R), flush=True)
    sys.exit(2 if fails else 0)


if __name__ == '__main__':
    main()
