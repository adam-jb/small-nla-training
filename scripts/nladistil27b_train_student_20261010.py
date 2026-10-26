"""Distil the 27B NLA (Qwen3.6-27B L42) into a small Qwen3.5 student (10 Oct 2026). Adapted from the 7B trainer.

mode vector: L42 vector -> Adapter -> k slots in place of the marker token, same prompt as the teacher.
mode text:   control that reads the source text (last --text-tokens tokens of the prefix), no vector.
Loss per output token: (1-w) * CE(teacher's sampled token) + w * soft CE to the teacher's renormalised top-20.
Stage A (vector mode): adapter only. Stage B: all weights, cosine LR with 50-step warm-up, --epochs passes.
Training items: rows of --set whose rung flag is set (--rung 66k) or the first --n-train rows, from the teacher dirs in
--teacher-sets (e.g. 'train' or 'train,train_s1' for two notes per vector; 'train,blends' with --mix-blends).
Monitoring (plan s.15): every --log-every steps CE / soft / grad norm / LR / tok/s / memory; every --quick-every examples
FVE on a fixed 256-vector held-out subset; at --eval-at and the end full held-out FVE (+ inject / notes / teval at the end);
stop rules: NaN, or share of teacher below --floor30 at 30k / --floor60 at 60k (needs --teacher-fve) -> STOPPED.json.
Usage: python scripts/nladistil27b_train_student_20261010.py --vec DIR --teacher DIR --student Qwen/Qwen3.5-2B --rung 66k --out runs/NAME
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (AR, LAYER, MAX_NEW, Adapter, av_prompt_ids, eos_ids, extract_expl,
                                                 jl, left_pad_embeds, rl, sample_embeds, text_prompt_ids)
from scripts.nladistil27b_extract_20261010 import load_set
from scripts.nladistil27b_ar_score_20261010 import rawvar_for
from scripts.nladistil27b_common_20261010 import fve_stats


def load_teacher(tdir, set_name, want):
    """Teacher tokens and top-20 for row ids in `want` -> dict i -> (tok, ids, lp)."""
    out = {}
    d = Path(tdir) / set_name
    for f in sorted(d.glob('T_*.jsonl')):
        recs = rl(f)
        z = np.load(str(f).replace('.jsonl', '_top.npz'))
        off, tok, tid, tlp = z['off'], z['tok'], z['tk_ids'], z['tk_lp']
        for j, r in enumerate(recs):
            if r['i'] in want and off[j + 1] > off[j]:
                out[r['i']] = (tok[off[j]:off[j + 1]].astype(np.int64), tid[off[j]:off[j + 1]].astype(np.int64),
                               tlp[off[j]:off[j + 1]].astype(np.float32))
    return out


class Student:
    def __init__(self, a, dev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.student)
        dt = torch.bfloat16 if a.bf16_weights else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(a.init_lm or a.student, dtype=dt).to(dev)
        if a.grad_ckpt:
            self.lm.gradient_checkpointing_enable()
        self.mode, self.k, self.dev, self.a = a.mode, a.slots, dev, a
        self.emb = self.lm.get_input_embeddings()
        self.d = self.lm.config.hidden_size
        self.eos = eos_ids(self.tok)
        if a.mode == 'vector':
            ids, p = av_prompt_ids(self.tok)
            self.pre, self.suf = ids[:p], ids[p + 1:]
            self.adapter = Adapter(self.d, a.slots).to(dev)
            self.adapter.init_to(self.emb.weight.detach().float().std().item())
            if a.init_adapter:
                self.adapter.load_state_dict(torch.load(a.init_adapter, map_location=dev))
        else:
            self.adapter = None

    def prompt_embeds(self, vecs=None, texts=None):
        if self.mode == 'vector':
            A = self.adapter(vecs).to(self.emb.weight.dtype)
            pre = self.emb(torch.tensor(self.pre, device=self.dev))
            suf = self.emb(torch.tensor(self.suf, device=self.dev))
            return [torch.cat([pre, A[b], suf]) for b in range(A.shape[0])]
        return [self.emb(torch.tensor(text_prompt_ids(self.tok, t), device=self.dev)) for t in texts]

    def text_for(self, text):
        ids = self.tok(text, add_special_tokens=False)['input_ids'][-self.a.text_tokens:]
        return self.tok.decode(ids)


def batch_loss(st, items, w):
    """items: list of (vec [D] np, text, (tok, ids, lp))."""
    dev = st.dev
    if st.mode == 'vector':
        P = st.prompt_embeds(vecs=torch.tensor(np.stack([x[0] for x in items]), device=dev))
    else:
        P = st.prompt_embeds(texts=[x[1] for x in items])
    seqs = []
    for p, x in zip(P, items):
        t = x[2][0]
        te = st.emb(torch.tensor(t, device=dev))
        seqs.append(torch.cat([p, te[:-1]]) if len(t) > 1 else p)
    T = max(s.shape[0] for s in seqs)
    E = torch.zeros(len(seqs), T, st.d, device=dev, dtype=seqs[0].dtype)
    M = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        E[i, :s.shape[0]] = s; M[i, :s.shape[0]] = 1
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
        h = st.lm.model(inputs_embeds=E, attention_mask=M, use_cache=False).last_hidden_state
    hs, ys, kid, klp = [], [], [], []
    for i, (p, x) in enumerate(zip(P, items)):
        n = len(x[2][0]); s0 = p.shape[0] - 1
        hs.append(h[i, s0:s0 + n])
        ys.append(torch.tensor(x[2][0], device=dev)); kid.append(torch.tensor(x[2][1], device=dev))
        klp.append(torch.tensor(x[2][2], device=dev))
    H = torch.cat(hs); Y = torch.cat(ys); KI = torch.cat(kid); KL = torch.cat(klp)
    ce_sum = soft_sum = 0.0
    for s in range(0, H.shape[0], 2048):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            logits = st.lm.lm_head(H[s:s + 2048])
        lf = logits.float()
        lse = torch.logsumexp(lf, -1, keepdim=True)  # no full log-softmax tensor: gather, then subtract
        ce = -(lf.gather(1, Y[s:s + 2048, None]) - lse)[:, 0]
        pt = torch.softmax(KL[s:s + 2048], -1)
        soft = -(pt * (lf.gather(1, KI[s:s + 2048]) - lse)).sum(-1)
        ce_sum = ce_sum + ce.sum(); soft_sum = soft_sum + soft.sum()
    n = H.shape[0]
    return ((1 - w) * ce_sum + w * soft_sum) / n, (ce_sum / n).item(), (soft_sum / n).item(), n


@torch.no_grad()
def generate(st, V, rows, bs, temperature=1.0):
    st.lm.eval()
    outs = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        if st.mode == 'vector':
            P = st.prompt_embeds(vecs=torch.tensor(V[[r['i'] for r in chunk]], device=st.dev))
        else:
            P = st.prompt_embeds(texts=[st.text_for(r['text']) for r in chunk])
        E, M = left_pad_embeds(P, st.d, P[0].dtype, st.dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=st.dev == 'cuda'):
            toks = sample_embeds(st.lm, E, M, st.eos, max_new=MAX_NEW, temperature=temperature)
        for r, t in zip(chunk, toks):
            text = st.tok.decode(t, skip_special_tokens=True)
            expl, ok = extract_expl(text)
            outs.append({'i': r['i'], 'raw': text, 'expl': expl, 'parsed': ok, 'n_tok': len(t), 'toks': t})
    st.lm.train()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--teacher-sets', default='train')
    ap.add_argument('--set', default='train')
    ap.add_argument('--rung', default='', help='e.g. 66k: rows with in_66k')
    ap.add_argument('--n-train', type=int, default=0)
    ap.add_argument('--vec-rows', type=int, default=0, help='use only the first N vectors of the rung (S4: 65.5k x 2 notes)')
    ap.add_argument('--mix-blends', type=float, default=0.0, help='share of items from the blends set (S3)')
    ap.add_argument('--student', default='Qwen/Qwen3.5-2B')
    ap.add_argument('--mode', choices=['vector', 'text'], default='vector')
    ap.add_argument('--out', required=True)
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--text-tokens', type=int, default=192)
    ap.add_argument('--stage-a', type=int, default=10000)
    ap.add_argument('--lr-a', type=float, default=1e-3)
    ap.add_argument('--lr-lm', type=float, default=1e-5)
    ap.add_argument('--lr-adapter', type=float, default=2e-4)
    ap.add_argument('--epochs', type=float, default=1.5)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--soft-w', type=float, default=0.5)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--quick-every', type=int, default=10000)
    ap.add_argument('--quick-n', type=int, default=256)
    ap.add_argument('--eval-at', default='30000,60000,120000,240000')
    ap.add_argument('--eval-bs', type=int, default=128)
    ap.add_argument('--final-sets', default='inject,notes,teval')
    ap.add_argument('--teacher-fve', type=float, default=0.0)
    ap.add_argument('--floor30', type=float, default=0.25)
    ap.add_argument('--floor60', type=float, default=0.45)
    ap.add_argument('--ar-offload', action='store_true', help='keep the AR on CPU between evals (4B student)')
    ap.add_argument('--bf16-weights', action='store_true')
    ap.add_argument('--grad-ckpt', action='store_true')
    ap.add_argument('--opt-offload', action='store_true', help='move AdamW state to CPU during evals (4B + AR on 96 GB)')
    ap.add_argument('--init-lm', default=None)
    ap.add_argument('--init-adapter', default=None)
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / 'config.json', 'w'), indent=1)
    t0 = time.time()
    log = open(out / 'metrics.jsonl', 'a')

    def note(rec):
        rec['sec'] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)

    # ---- data
    V = load_set(a.vec, a.set, LAYER)
    rows = rl(Path(a.vec) / a.set / 'rows.jsonl')
    pick = [r['i'] for r in rows if (r.get(f'in_{a.rung}', False) if a.rung else True)]
    if a.n_train:
        pick = pick[:a.n_train]
    if a.vec_rows:
        pick = pick[:a.vec_rows]
    want = set(pick)
    items = []
    for ts in a.teacher_sets.split(','):
        td = load_teacher(a.teacher, ts, want)
        items += [(i, ts) for i in pick if i in td]
        if ts == a.teacher_sets.split(',')[0]:
            tdata = {ts: td}
        else:
            tdata[ts] = td
    if a.mix_blends > 0:
        Vb = load_set(a.vec, 'blends', LAYER)
        brows = rl(Path(a.vec) / 'blends' / 'rows.jsonl')
        nb = int(round(len(items) * a.mix_blends / (1 - a.mix_blends)))
        tb = load_teacher(a.teacher, 'blends', set(range(nb)))
        tdata['blends'] = tb
        keep_n = len(items) - nb  # keep the total fixed: drop real items from the end
        items = items[:keep_n] + [(i, 'blends') for i in range(nb) if i in tb]
    assert items, f'no training items: check --teacher-sets ({a.teacher_sets}) against {a.teacher}'
    note({'data': True, 'items': len(items), 'vectors': len(want), 'sets': a.teacher_sets, 'blends': a.mix_blends})

    st = Student(a, dev)
    texts = {r['i']: r['text'] for r in rows}

    def item(x):
        i, ts = x
        if ts == 'blends':
            return (Vb[i], brows[i]['text'], tdata['blends'][i])
        return (V[i], st.text_for(texts[i]) if a.mode == 'text' else '', tdata[ts][i])

    # ---- held-out + AR
    Vh = load_set(a.vec, 'heldout', LAYER)
    held = rl(Path(a.vec) / 'heldout' / 'rows.jsonl')
    quick = held[:a.quick_n]
    rawvar = rawvar_for(a.vec)
    ar = AR('cpu' if a.ar_offload else dev)

    def ar_eval(outs, Vs, srows, tag):
        if a.ar_offload:
            ar.body.to(dev); ar.head.to(dev); ar.dev = dev
        P = ar.reconstruct([o['expl'] for o in outs])
        if a.ar_offload:
            ar.body.to('cpu'); ar.head.to('cpu'); ar.dev = 'cpu'; torch.cuda.empty_cache()
        G = torch.tensor(Vs[[o['i'] for o in outs]])
        s, c = fve_stats(P, G, rawvar, groups=[srows[o['i']].get('doc_id', o['i']) for o in outs])
        for o, x in zip(outs, c.tolist()):
            o['ar_cos'] = x
        s.update({'tag': tag, 'parsed': float(np.mean([o['parsed'] for o in outs])),
                  'mean_tok': float(np.mean([o['n_tok'] for o in outs]))})
        if a.teacher_fve:
            s['share'] = s['fve'] / a.teacher_fve
        return s

    def full_eval(seen, sets=('heldout',)):
        res = {}
        for sname in sets:
            Vs = Vh if sname == 'heldout' else load_set(a.vec, sname, LAYER)
            srows = held if sname == 'heldout' else rl(Path(a.vec) / sname / 'rows.jsonl')
            outs = generate(st, Vs, srows, a.eval_bs)
            if sname in ('heldout', 'teval'):
                res[sname] = ar_eval(outs, Vs, srows, f'{sname}_seen{seen}')
            jl(out / f'gen_{sname}_seen{seen}.jsonl', outs)
            if sname == 'inject':
                hits = {}
                for o in outs:
                    r = srows[o['i']]
                    hits.setdefault(r['alpha'], []).append(any(k in o['expl'].lower() for k in r['keywords']))
                res['inject'] = {str(k): float(np.mean(v)) for k, v in sorted(hits.items())}
            if sname == 'notes':
                res['notes'] = {f: float(np.mean([srows[o['i']][f].lower() in o['expl'].lower() for o in outs]))
                                for f in ('city', 'job')}
        note({'eval': seen, **res, 'examples': [o['expl'][:240] for o in outs[:3]]})
        return res

    # ---- training
    evals_full = sorted(int(x) for x in a.eval_at.split(',') if x)

    def opt_to(opt, d):  # 12 Oct: 4B AdamW state (~34 GB) + the AR (35 GB) do not fit together on 96 GB
        if not a.opt_offload:
            return
        for s_ in opt.state.values():
            for k, v in s_.items():
                if torch.is_tensor(v) and v.dim() > 0:
                    s_[k] = v.to(d)
        if d == 'cpu' and dev == 'cuda':
            torch.cuda.empty_cache()

    def run_stage(groups, n_examples, name, do_evals):
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        steps = max(1, n_examples // a.bs)
        base = [g['lr'] for g in opt.param_groups]
        seen, step, order, ntok, t1 = 0, 0, [], 0, time.time()
        next_quick = a.quick_every
        ev = [e for e in evals_full if e < n_examples] if do_evals else []
        params = [p for g in groups for p in g['params']]
        while step < steps:
            if not order:
                order = items[:]; random.shuffle(order)
            batch = [order.pop() for _ in range(min(a.bs, len(order)))]
            frac = step / steps
            for g, b in zip(opt.param_groups, base):
                g['lr'] = b * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * frac))
            opt.zero_grad(set_to_none=True)
            ce_t = soft_t = 0.0
            for s in range(0, len(batch), a.micro):
                mb = [item(x) for x in batch[s:s + a.micro]]
                loss, ce, soft, n = batch_loss(st, mb, a.soft_w)
                (loss * len(mb) / len(batch)).backward()
                ce_t += ce * len(mb) / len(batch); soft_t += soft * len(mb) / len(batch); ntok += n
            if not math.isfinite(ce_t) or not math.isfinite(soft_t):
                json.dump({'why': 'NaN loss', 'stage': name, 'step': step}, open(out / 'STOPPED.json', 'w'))
                note({'stopped': 'NaN loss', 'step': step}); sys.exit(3)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)  # free grads before any eval (the AR needs the room)
            step += 1; seen += len(batch)
            if step % a.log_every == 0 or step == steps:
                rec = {'stage': name, 'step': step, 'seen': seen, 'ce': ce_t, 'soft': soft_t, 'gnorm': gn,
                       'lr': opt.param_groups[0]['lr'], 'tok_s': round(ntok / (time.time() - t1))}
                if dev == 'cuda':
                    rec['mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
                if st.adapter is not None:
                    rec['gain'] = st.adapter.gain.item()
                note(rec); ntok, t1 = 0, time.time()
            do_q, do_f = do_evals and seen >= next_quick, bool(ev and seen >= ev[0])
            if do_q or do_f:
                opt_to(opt, 'cpu')
            if do_q:
                next_quick += a.quick_every
                outs = generate(st, Vh, quick, a.eval_bs)
                s = ar_eval(outs, Vh, held, f'quick_seen{seen}')
                note({'quick': seen, **s, 'example': outs[0]['expl'][:240]})
            while ev and seen >= ev[0]:
                k = ev.pop(0)
                res = full_eval(k)
                share = res.get('heldout', {}).get('share')
                floor = a.floor30 if k == 30000 else a.floor60 if k == 60000 else None
                if share is not None and floor is not None and share < floor and a.stage_a > 0:
                    json.dump({'why': 'share below floor', 'seen': k, 'share': share, 'floor': floor},
                              open(out / 'STOPPED.json', 'w'))
                    note({'stopped': 'share below floor', 'seen': k, 'share': share}); sys.exit(4)
            if do_q or do_f:
                opt_to(opt, dev)

    if a.mode == 'vector' and a.stage_a > 0:
        for p_ in st.lm.parameters():
            p_.requires_grad_(False)
        run_stage([{'params': list(st.adapter.parameters()), 'lr': a.lr_a}], a.stage_a, 'A', False)
        for p_ in st.lm.parameters():
            p_.requires_grad_(True)
    groups = [{'params': list(st.lm.parameters()), 'lr': a.lr_lm}]
    if st.adapter is not None:
        groups.append({'params': list(st.adapter.parameters()), 'lr': a.lr_adapter})
    nB = int(len(items) * a.epochs)
    run_stage(groups, nB, 'B', True)
    res = full_eval(nB, ['heldout'] + [x for x in a.final_sets.split(',') if x])
    if a.save:
        sd = out / 'final_lm'
        st.lm.to(torch.bfloat16).save_pretrained(sd, safe_serialization=True); st.tok.save_pretrained(sd)
        if st.adapter is not None:
            torch.save(st.adapter.state_dict(), out / 'final_adapter.pt')
    json.dump({'done': True, 'sec': time.time() - t0, 'final_seen': nB, 'items': len(items), 'final': res},
              open(out / 'done.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
