"""Agentic-intent GRPO ladder: final paired eval, start model (GRPO526) vs one arm's best checkpoint (24 Oct 2026).
Question (Adam, 24 Oct): did the arm learn to describe agent text without forgetting web text?
Both models score the SAME rows with the same seed, so the gain is paired (CI by bootstrap over groups/docs):
  web    : all 1,020 web held-out rows (web rawvar)          -> forgetting test
  dev    : all agent_dev rows (agent rawvar)                 -> what the arm picked its best step on (biased up)
  hold   : agent_hold, <= --per-source rows per source (agent rawvar), never used for training or model choice
           broken down by source, slice (A tempting / B twin / C ordinary), row type:
           P_last = last prompt token (Qwen about to act: the decision point), P_mid = other prompt rows, G = reply rows;
           deceptionbench is a source never seen in training (whole-source holdout) -> transfer test.
  twin   : decision-point (P_last) rows of holdout twin pairs (tempting A vs its no-pressure twin B; 385 pairs, mostly
           deceptionbench + propensitybench). Score = cos(AR(note_A) - AR(note_B), v_A - v_B): does the note pair capture what
           the pressure changed in Qwen's state? A label-free proxy for "reads the intent shift", not just the setting.
Pre-set verdict: LEARNED if hold gain CI > 0; NO FORGETTING if web gain CI lower end > -0.01.
Usage (box): python scripts/agintent_final_eval_20261024.py --vec /workspace/vec_ai --arm runs/NAME --start runs/START --out runs/NAME/final_eval
"""
import argparse, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import (AR, LAYER, MAX_NEW, Adapter, av_prompt_ids, eos_ids, extract_expl,
                                                 left_pad_embeds, rl, sample_embeds)
from scripts.nladistil27b_extract_20261010 import load_set
from scripts.nladistil27b_ar_score_20261010 import rawvar_for
from scripts.agintent_junk_20261024 import is_junk


def pick_rows(vec, per_source):
    rows = rl(Path(vec) / 'agent_hold' / 'rows.jsonl')
    last = {}
    for r in rows:
        if r['type'] == 'P':
            last[r['unit_id']] = max(last.get(r['unit_id'], -1), r['pos'])
    for r in rows:
        r['rtype'] = 'G' if r['type'] == 'G' else ('P_last' if r['pos'] == last[r['unit_id']] else 'P_mid')
    by = defaultdict(list)
    for r in rows:
        by[r['source']].append(r)
    rng = random.Random(0); keep = []
    for s in sorted(by):
        units = sorted({r['unit_id'] for r in by[s]}); rng.shuffle(units)
        want, got = set(), 0
        for u in units:  # whole units, so P_last / P_mid / G come from the same scenarios
            n = sum(r['unit_id'] == u for r in by[s])
            if got + n > per_source and got:
                break
            want.add(u); got += n
        keep += [r for r in by[s] if r['unit_id'] in want]
    return keep


def twin_rows(vec):
    """Holdout P_last rows (k=0) of twin pairs whose A and B are both present: returns (rows_A, rows_B)."""
    tw = {}
    for l in open(Path(__file__).resolve().parents[1] / 'data/agintent_units_20261024/units_split.jsonl'):
        u = json.loads(l)
        if u.get('twin_of') and u['split'] == 'holdout':
            tw[u['unit_id']] = u['twin_of']
    rows = rl(Path(vec) / 'agent_hold' / 'rows.jsonl'); last = {}
    for r in rows:
        if r['type'] == 'P' and (r['unit_id'] not in last or r['pos'] > last[r['unit_id']]['pos']):
            last[r['unit_id']] = r
    pairs = [(last[a], last[b]) for b, a in sorted(tw.items()) if a in last and b in last]
    return [a for a, _ in pairs], [b for _, b in pairs]


def score_model(model_dir, sets, ar, dev, seed, gen_bs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    md = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(md / 'final_lm')
    lm = AutoModelForCausalLM.from_pretrained(md / 'final_lm', dtype=torch.bfloat16).to(dev).eval()
    emb = lm.get_input_embeddings(); dS = lm.config.hidden_size; eos = eos_ids(tok)
    ad = Adapter(dS, 4).to(dev); ad.load_state_dict(torch.load(md / 'final_adapter.pt', map_location=dev)); ad.eval()
    ids, pinj = av_prompt_ids(tok)
    pre = emb(torch.tensor(ids[:pinj], device=dev)); suf = emb(torch.tensor(ids[pinj + 1:], device=dev))
    out = {}
    for name, (V, idx) in sets.items():
        torch.manual_seed(seed); texts = []
        with torch.no_grad():
            for s in range(0, len(idx), gen_bs):
                A = ad(torch.tensor(V[idx[s:s + gen_bs]], device=dev)).to(emb.weight.dtype)
                P = [torch.cat([pre, A[b], suf]) for b in range(A.shape[0])]
                E, M = left_pad_embeds(P, dS, P[0].dtype, dev)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tk = sample_embeds(lm, E, M, eos, max_new=MAX_NEW)
                texts += [(tok.decode(t, skip_special_tokens=True), len(t)) for t in tk]
        ex = [extract_expl(x) for x, _ in texts]
        R = ar.reconstruct([e for e, _ in ex])
        c = torch.nn.functional.cosine_similarity(R.float(), torch.tensor(V[idx]).float(), dim=-1).numpy()
        out[name] = [{'expl': e, 'parsed': ok, 'n_tok': n, 'cos': float(cc), 'junk': is_junk(e)}
                     for (e, ok), (_, n), cc in zip(ex, texts, c)]
        if name == 'twin':
            out['twin_R'] = R.float().cpu()
        print(json.dumps({'model': md.name, 'set': name, 'n': len(idx), 'cos': float(c.mean())}), flush=True)
    del lm; torch.cuda.empty_cache()
    return out


def paired(c0, c1, groups, rawvar, n_boot=1000, seed=0):
    """FVE = 1 - mean(2(1-cos))/rawvar for start (0) and arm (1), paired gain with CI by bootstrap over groups."""
    m0, m1 = 2 * (1 - np.asarray(c0)), 2 * (1 - np.asarray(c1))
    ug = sorted(set(groups)); gi = defaultdict(list)
    for k, u in enumerate(groups):
        gi[u].append(k)
    g = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        ix = np.concatenate([gi[ug[j]] for j in g.choice(len(ug), len(ug))])
        boot.append((m0[ix].mean() - m1[ix].mean()) / rawvar)
    f0, f1 = 1 - m0.mean() / rawvar, 1 - m1.mean() / rawvar
    return {'n': len(m0), 'groups': len(ug), 'fve_start': round(float(f0), 4), 'fve_arm': round(float(f1), 4),
            'gain': round(float(f1 - f0), 4), 'gain_ci95': [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True); ap.add_argument('--arm', required=True); ap.add_argument('--start', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--per-source', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gen-bs', type=int, default=512)
    a = ap.parse_args(); dev = 'cuda'; t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    held = rl(Path(a.vec) / 'heldout' / 'rows.jsonl'); Vh = load_set(a.vec, 'heldout', LAYER)
    drows = rl(Path(a.vec) / 'agent_dev' / 'rows.jsonl'); Vd = load_set(a.vec, 'agent_dev', LAYER)
    hrows = pick_rows(a.vec, a.per_source); Va = load_set(a.vec, 'agent_hold', LAYER)
    tA, tB = twin_rows(a.vec)
    sets = {'web': (Vh, list(range(len(held)))), 'dev': (Vd, list(range(len(drows)))), 'hold': (Va, [r['i'] for r in hrows]),
            'twin': (Va, [r['i'] for r in tA + tB])}
    meta = {'web': [{'group': r['doc_id']} for r in held], 'dev': drows, 'hold': hrows, 'twin': tA + tB}
    rv_web = rawvar_for(a.vec); rv_ag = json.load(open(Path(a.vec) / 'rawvar_agent.json'))['rawvar']
    ar = AR(dev)
    res = {k: score_model(p, sets, ar, dev, a.seed, a.gen_bs) for k, p in (('start', a.start), ('arm', a.arm))}
    for name in sets:
        with open(out / f'notes_{name}.jsonl', 'w') as f:
            for m, s0, s1 in zip(meta[name], res['start'][name], res['arm'][name]):  # twin: first half A rows, second half B
                f.write(json.dumps({**{k: m.get(k) for k in ('unit_id', 'source', 'slice', 'group', 'rtype', 'k', 'ctx')},
                                    'start': s0, 'arm': s1}) + '\n')
    rep = {}
    for name, rv in (('web', rv_web), ('dev', rv_ag), ('hold', rv_ag), ('twin', rv_ag)):
        s0, s1 = res['start'][name], res['arm'][name]; M = meta[name]
        def sub(keep):
            ix = [i for i in range(len(M)) if keep(M[i])]
            return paired([s0[i]['cos'] for i in ix], [s1[i]['cos'] for i in ix], [M[i]['group'] for i in ix], rv) if ix else None
        r = {'all': sub(lambda m: True)}
        for key, k2 in (('mean_len', 'n_tok'), ('junk', 'junk'), ('parsed', 'parsed')):
            r[key] = {'start': round(float(np.mean([x[k2] for x in s0])), 4), 'arm': round(float(np.mean([x[k2] for x in s1])), 4)}
        if name == 'hold':
            for f in ('source', 'slice', 'rtype'):
                r['by_' + f] = {v: sub(lambda m, v=v, f=f: m.get(f) == v) for v in sorted({m.get(f) for m in M})}
            r['seen_sources'] = sub(lambda m: m['source'] != 'deceptionbench')
            r['unseen_source_deceptionbench'] = sub(lambda m: m['source'] == 'deceptionbench')
            r['decision_point_slice_A'] = sub(lambda m: m['rtype'] == 'P_last' and m['slice'] == 'A')
        rep[name] = r
    h, w = rep['hold']['all'], rep['web']['all']
    nP = len(tA); G = torch.tensor(Va[[r['i'] for r in tA + tB]]).float(); dv = G[:nP] - G[nP:]
    tc = {k: torch.nn.functional.cosine_similarity(res[k]['twin_R'][:nP] - res[k]['twin_R'][nP:], dv, dim=-1).numpy() for k in res}
    g = np.random.default_rng(0); bt = []
    for _ in range(1000):
        ix = g.choice(nP, nP); bt.append(float(tc['arm'][ix].mean() - tc['start'][ix].mean()))
    rep['twin_shift'] = {'pairs': nP, 'by_source': dict(Counter(r['source'] for r in tA)),
                         'cos_start': round(float(tc['start'].mean()), 4), 'cos_arm': round(float(tc['arm'].mean()), 4),
                         'gain': round(float(tc['arm'].mean() - tc['start'].mean()), 4),
                         'gain_ci95': [round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)],
                         'by_source_arm_minus_start': {s_: round(float(np.mean([tc['arm'][j] - tc['start'][j] for j in range(nP) if tA[j]['source'] == s_])), 4)
                                                       for s_ in sorted({r['source'] for r in tA})},
                         'meaning': 'cos between (AR(note_A)-AR(note_B)) and (v_A-v_B) at the decision point; higher = the notes capture what the pressure changed'}
    rep['verdict'] = {'learned_agent_text': h['gain_ci95'][0] > 0, 'no_web_forgetting': w['gain_ci95'][0] > -0.01,
                      'rule': 'learned if hold gain CI95 > 0; no forgetting if web gain CI95 lower end > -0.01 (FVE)'}
    rep['sec'] = round(time.time() - t0)
    json.dump(rep, open(out / 'final_eval.json', 'w'), indent=1)
    print(json.dumps(rep['verdict']), json.dumps({'hold': h, 'web': w}), flush=True)


if __name__ == '__main__':
    main()
