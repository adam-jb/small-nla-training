"""GRPO against the 27B teacher's AR (R1), from a distilled Qwen3.5 student (10 Oct 2026). Adapted from the 7B run.

policy = start student + LoRA on every linear in the LM (attention, DeltaNet projections, MLP; base frozen, bf16) + the
vector adapter (full). Each step: P vectors x G samples at T 1; reward = -2(1 - cos(AR(expl), v)) (= the teacher run's
-MSE on sqrt(d)-normalised vectors); advantage = (r - group mean) / group std; token-level PPO-clip + beta * k2 KL to
the start policy (LoRA off + start adapter).
Every --eval-every steps: held-out FVE (256 fixed vectors; full 1,020 at the end), injection (1,200), alpha-0 false
alarms. Stop rules (plan s.15): plateau (< --plateau gain over the last 2 evals) -> done; KL > --kl-max, length drift
> 20%, or injection alpha 0.5 down > 5 pts / alpha-0 false alarms up > 5 pts vs step 0 -> STOPPED.json.
Usage: python scripts/nladistil27b_rl_grpo_20261010.py --vec DIR --init RUN_DIR --out runs/NAME
"""
import argparse, copy, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (AR, LAYER, MAX_NEW, Adapter, av_prompt_ids, eos_ids, extract_expl,
                                                 fve_stats, left_pad_embeds, rl, sample_embeds)
from scripts.nladistil27b_extract_20261010 import load_set
from scripts.nladistil27b_ar_score_20261010 import rawvar_for

MODS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj')


class LoRA(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.s, self.on = base, alpha / r, True
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        y = self.base(x)
        if self.on:
            y = y + ((x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)) * self.s
        return y


def add_lora(lm, r, alpha):
    loras = []
    for _, mod in list(lm.model.named_modules()):
        for c in MODS:
            if isinstance(getattr(mod, c, None), nn.Linear):
                l = LoRA(getattr(mod, c), r, alpha); setattr(mod, c, l); loras.append(l)
    return loras


def set_lora(loras, on):
    for l in loras:
        l.on = on


def seq_logps(lm, emb, P, toks, dev):
    seqs = []
    for p, t in zip(P, toks):
        te = emb(torch.tensor(t, device=dev)).to(p.dtype)
        seqs.append(torch.cat([p, te[:-1]]) if len(t) > 1 else p)
    T = max(s.shape[0] for s in seqs); d = seqs[0].shape[-1]
    E = torch.zeros(len(seqs), T, d, device=dev, dtype=seqs[0].dtype)
    M = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        E[i, :s.shape[0]] = s; M[i, :s.shape[0]] = 1
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
        h = lm.model(inputs_embeds=E, attention_mask=M, use_cache=False).last_hidden_state
    out = []
    for i, (p, t) in enumerate(zip(P, toks)):
        n = len(t); s0 = p.shape[0] - 1
        hs = h[i, s0:s0 + n]; y = torch.tensor(t, device=dev)
        lps = []
        for c in range(0, n, 1024):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                lg = lm.lm_head(hs[c:c + 1024])
            lps.append(torch.log_softmax(lg.float(), -1).gather(1, y[c:c + 1024, None])[:, 0])
        out.append(torch.cat(lps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool', default='train', help='vector set for prompts')
    ap.add_argument('--prompts', type=int, default=64)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--minibatches', type=int, default=2)
    ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--alpha', type=float, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lr-adapter', type=float, default=5e-6)
    ap.add_argument('--beta', type=float, default=0.01)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--hours', type=float, default=14.0)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--eval-n', type=int, default=256)
    ap.add_argument('--gen-bs', type=int, default=512)
    ap.add_argument('--plateau', type=float, default=0.003)
    ap.add_argument('--min-steps', type=int, default=200)
    ap.add_argument('--kl-max', type=float, default=0.1)
    ap.add_argument('--grad-ckpt', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / 'config.json', 'w'), indent=1)
    init = Path(a.init)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lm_dir = init / 'final_lm'
    tok = AutoTokenizer.from_pretrained(lm_dir)
    lm = AutoModelForCausalLM.from_pretrained(lm_dir, dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev)
    for p_ in lm.parameters():
        p_.requires_grad_(False)
    loras = add_lora(lm, a.rank, a.alpha)
    if a.grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    emb = lm.get_input_embeddings()
    dS = lm.config.hidden_size
    eos = eos_ids(tok)
    adapter = Adapter(dS, 4).to(dev)
    adapter.load_state_dict(torch.load(init / 'final_adapter.pt', map_location=dev))
    ref_adapter = copy.deepcopy(adapter).eval()
    for p_ in ref_adapter.parameters():
        p_.requires_grad_(False)
    ids, pinj = av_prompt_ids(tok)
    pre = torch.tensor(ids[:pinj], device=dev); suf = torch.tensor(ids[pinj + 1:], device=dev)

    def prompts(ad, vecs):
        A = ad(vecs).to(emb.weight.dtype)
        pe, se = emb(pre), emb(suf)
        return [torch.cat([pe, A[b], se]) for b in range(A.shape[0])]

    ar = AR(dev)
    rawvar = rawvar_for(a.vec)
    V = load_set(a.vec, a.pool, LAYER)
    pool_ids = list(range(len(V)))
    Vh = load_set(a.vec, 'heldout', LAYER); held = rl(Path(a.vec) / 'heldout' / 'rows.jsonl')
    Vi = load_set(a.vec, 'inject', LAYER); inj = rl(Path(a.vec) / 'inject' / 'rows.jsonl')
    params = [p_ for l in loras for p_ in (l.A, l.B)]
    opt = torch.optim.AdamW([{'params': params, 'lr': a.lr}, {'params': list(adapter.parameters()), 'lr': a.lr_adapter}],
                            weight_decay=0.0, betas=(0.9, 0.99))
    base_lr = [a.lr, a.lr_adapter]
    step, order = 0, []
    ck = out / 'ckpt.pt'
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        for l, (A_, B_) in zip(loras, s['lora']):
            l.A.data.copy_(A_); l.B.data.copy_(B_)
        adapter.load_state_dict(s['adapter']); opt.load_state_dict(s['opt'])
        step, order = s['step'], s['order']
        random.setstate(s['py_rng']); torch.set_rng_state(s['rng'].cpu())
        print('resumed at step', step, flush=True)
    log = open(out / 'metrics.jsonl', 'a'); elog = open(out / 'evals.jsonl', 'a')
    evals = [json.loads(x) for x in open(out / 'evals.jsonl')] if (out / 'evals.jsonl').stat().st_size else []
    t0 = time.time()

    @torch.no_grad()
    def generate(Vs, idx, bs):
        lm.eval(); res = []
        for s in range(0, len(idx), bs):
            vec = torch.tensor(Vs[idx[s:s + bs]], device=dev)
            P = prompts(adapter, vec)
            E, M = left_pad_embeds(P, dS, P[0].dtype, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                res += sample_embeds(lm, E, M, eos, max_new=MAX_NEW)
        lm.train()
        return res

    def evaluate(tag, n_held):
        hi = list(range(min(n_held, len(held))))
        tk = generate(Vh, hi, a.gen_bs)
        texts = [tok.decode(t, skip_special_tokens=True) for t in tk]
        ex = [extract_expl(x) for x in texts]
        P = ar.reconstruct([e for e, _ in ex])
        s, c = fve_stats(P, torch.tensor(Vh[hi]), rawvar, groups=[held[i]['doc_id'] for i in hi])
        with open(out / f'gen_heldout_step{tag}.jsonl', 'w') as f:
            for i, x, (e, ok), t, cc in zip(hi, texts, ex, tk, c.tolist()):
                f.write(json.dumps({'i': i, 'raw': x, 'expl': e, 'parsed': ok, 'n_tok': len(t), 'ar_cos': cc}) + '\n')
        tki = generate(Vi, [r['i'] for r in inj], a.gen_bs)
        hits = {}
        for r, t in zip(inj, tki):
            e, _ = extract_expl(tok.decode(t, skip_special_tokens=True))
            hits.setdefault(str(r['alpha']), []).append(any(k in e.lower() for k in r['keywords']))
        rec = {'step': tag, 'n': len(hi), **s, 'parsed': float(np.mean([ok for _, ok in ex])),
               'len': float(np.mean([len(t) for t in tk])), 'inject': {k: float(np.mean(v)) for k, v in sorted(hits.items())},
               'example': ex[0][0][:240], 'sec': round(time.time() - t0)}
        elog.write(json.dumps(rec) + '\n'); elog.flush(); print('EVAL', json.dumps(rec), flush=True)
        return rec

    def save_ckpt():
        torch.save({'lora': [(l.A.data, l.B.data) for l in loras], 'adapter': adapter.state_dict(), 'opt': opt.state_dict(),
                    'step': step, 'order': order, 'py_rng': random.getstate(), 'rng': torch.get_rng_state()}, str(ck) + '.tmp')
        Path(str(ck) + '.tmp').rename(ck)

    def stop(why, **kw):
        json.dump({'why': why, 'step': step, **kw}, open(out / 'STOPPED.json', 'w'))
        print('STOP', why, kw, flush=True)

    import signal
    term = {'hit': False}
    signal.signal(signal.SIGTERM, lambda *_: term.__setitem__('hit', True))  # pre-emption notice: save and exit
    if not evals:
        evals.append(evaluate(0, a.eval_n))
    e0 = evals[0]
    lm.train()
    reason = 'steps'
    while step < a.steps:
        if term['hit']:
            save_ckpt(); print('SIGTERM: checkpoint saved at step', step, flush=True); sys.exit(143)
        if time.time() - t0 > a.hours * 3600:
            reason = 'hours'; break
        ts = time.time()
        if len(order) < a.prompts:
            o = pool_ids[:]; random.shuffle(o); order = order + o
        pi, order = order[:a.prompts], order[a.prompts:]
        idx = [i for i in pi for _ in range(a.group)]
        toks = generate(V, idx, a.gen_bs)
        toks = [t if t else [eos[0]] for t in toks]
        tg = time.time() - ts
        texts = [tok.decode(t, skip_special_tokens=True) for t in toks]
        ex = [extract_expl(x) for x in texts]
        Pr = ar.reconstruct([e for e, _ in ex])
        cos = torch.nn.functional.cosine_similarity(Pr.float(), torch.tensor(V[idx]), dim=-1)
        r = -2 * (1 - cos)
        rg = r.view(a.prompts, a.group)
        adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
        vec_all = torch.tensor(V[idx], device=dev)
        old, ref = [], []
        with torch.no_grad():
            lm.eval()
            for s in range(0, len(idx), a.micro * 2):
                sl = slice(s, s + a.micro * 2)
                old += seq_logps(lm, emb, prompts(adapter, vec_all[sl]), toks[sl], dev)
                set_lora(loras, False)
                ref += seq_logps(lm, emb, prompts(ref_adapter, vec_all[sl]), toks[sl], dev)
                set_lora(loras, True)
            lm.train()
        for g, b in zip(opt.param_groups, base_lr):
            g['lr'] = b * min(1.0, (step + 1) / 10)
        perm = torch.randperm(len(idx)).tolist()
        mbs = len(idx) // a.minibatches
        stt = {'kl': 0.0, 'clipfrac': 0.0, 'ntok': 0}
        for m in range(a.minibatches):
            mb = perm[m * mbs:(m + 1) * mbs]
            ntok = sum(len(toks[j]) for j in mb)
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(mb), a.micro):
                js = mb[s:s + a.micro]
                lps = seq_logps(lm, emb, prompts(adapter, vec_all[js]), [toks[j] for j in js], dev)
                loss = 0.0
                for j, lp in zip(js, lps):
                    ratio = torch.exp(lp - old[j]); A_ = adv[j].item()
                    pg = -torch.minimum(ratio * A_, ratio.clamp(1 - a.clip, 1 + a.clip) * A_)
                    loss = loss + (pg + a.beta * 0.5 * (lp - ref[j]) ** 2).sum()
                    stt['kl'] += (lp - ref[j]).detach().sum().item()
                    stt['clipfrac'] += ((ratio - 1).abs() > a.clip).float().sum().item(); stt['ntok'] += len(lp)
                (loss / ntok).backward()
            torch.nn.utils.clip_grad_norm_(params + list(adapter.parameters()), 1.0)
            opt.step()
        step += 1
        ntok = max(1, stt['ntok'])
        rec = {'step': step, 'reward': r.mean().item(), 'fve_train': 1 - (-r).mean().item() / rawvar,
               'kl': stt['kl'] / ntok, 'clipfrac': stt['clipfrac'] / ntok, 'len': float(np.mean([len(t) for t in toks])),
               'parsed': float(np.mean([ok for _, ok in ex])), 'gen_s': round(tg, 1), 'step_s': round(time.time() - ts, 1),
               'rollouts': step * len(idx), 'sec': round(time.time() - t0)}
        if dev == 'cuda':
            rec['mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)
        if abs(rec['kl']) > a.kl_max:
            stop('KL too large', kl=rec['kl']); reason = 'kl'; break
        if step % a.eval_every == 0:
            save_ckpt()
            ev = evaluate(step, a.eval_n); evals.append(ev)
            if abs(ev['len'] - e0['len']) > 0.2 * e0['len']:
                stop('length drift', len=ev['len'], len0=e0['len']); reason = 'length'; break
            if (ev['inject'].get('0.5', 0) < e0['inject'].get('0.5', 0) - 0.05 or
                    ev['inject'].get('0.0', 0) > e0['inject'].get('0.0', 0) + 0.05):
                stop('injection worse or false alarms up', inject=ev['inject'], inject0=e0['inject']); reason = 'inject'; break
            if step >= a.min_steps and len(evals) >= 3 and evals[-1]['fve'] - evals[-3]['fve'] < a.plateau:
                reason = 'plateau'; break
    save_ckpt()
    final = evaluate(f'{step}_full', len(held))
    with torch.no_grad():
        for l in loras:
            l.base.weight.data = (l.base.weight.data.float() + l.s * (l.B.data.float() @ l.A.data.float())).to(l.base.weight.dtype)
            l.on = False
    for _, mod in list(lm.model.named_modules()):
        for c in MODS:
            if isinstance(getattr(mod, c, None), LoRA):
                setattr(mod, c, getattr(mod, c).base)
    lm.save_pretrained(out / 'final_lm', safe_serialization=True); tok.save_pretrained(out / 'final_lm')
    torch.save(adapter.state_dict(), out / 'final_adapter.pt')
    json.dump({'done': True, 'step': step, 'reason': reason, 'final': final, 'sec': time.time() - t0},
              open(out / 'done.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
