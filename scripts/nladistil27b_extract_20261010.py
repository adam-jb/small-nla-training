"""Extract Qwen3.6-27B residual vectors for the 27B NLA distillation (10 Oct 2026).

Each prefix is fed alone (as nanoNLA's regenerate_activations does) and the outputs of blocks 28, 42 and 52 are taken at
its last token. Vectors are stored as bf16 bits in uint16 .npy shards of --shard rows: OUT/{set}/L{layer}_{k:03d}.npy,
plus OUT/{set}/rows.jsonl. Resumable: finished shards are skipped.
Sets: train (sample train.parquet), heldout (heldout_fresh), teval (heldout_teachereval), then probe sets built from
heldout: inject (30 concepts x 10 vectors x alpha 0/0.25/0.5/1), notes (120 synthetic notes), and blends (S3 arm).
Usage: python scripts/nladistil27b_extract_20261010.py --sample data/nladistil27b_sample_20261010 --out DIR [--sets ...]
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import SUBJECT, LAYER, LAYERS_SAVED, SMOKE, D, jl, rl, decoder_layers

MAX_TOK = 96 if SMOKE else 4096


def to_bf16_u16(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def from_bf16_u16(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).float()


def load_set(out, name, layer=LAYER):
    """All shards of one set and layer -> float32 numpy [N, D]."""
    d = Path(out) / name
    parts = sorted(d.glob(f'L{layer}_*.npy'))
    return np.concatenate([from_bf16_u16(np.load(p)).numpy() for p in parts]) if parts else None


class Extractor:
    def __init__(self, dev, layers=LAYERS_SAVED):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(SUBJECT)
        m = AutoModelForCausalLM.from_pretrained(SUBJECT, dtype=torch.bfloat16)
        self.layers_saved = tuple(layers)
        layers = decoder_layers(m)
        keep = max(self.layers_saved) + 1
        body = m.model
        body.layers = layers[:keep]  # later blocks never affect these outputs
        self.body = body.to(dev).eval()
        self.dev, self.cap = dev, {}
        for L in self.layers_saved:
            self.body.layers[L].register_forward_hook(
                lambda mod, i, o, L=L: self.cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def last_states(self, id_lists, tok_budget):
        """{layer: [N, D] bf16 cpu} at each list's last token. Right padded, batched by token budget."""
        order = sorted(range(len(id_lists)), key=lambda i: -len(id_lists[i]))
        res = {L: [None] * len(id_lists) for L in self.layers_saved}
        s = 0
        while s < len(order):
            L0 = len(id_lists[order[s]])
            n = max(1, min(256, tok_budget // max(L0, 1)))
            idx = order[s:s + n]; s += n
            ids = torch.zeros((len(idx), L0), dtype=torch.long)
            att = torch.zeros((len(idx), L0), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(id_lists[i])] = torch.tensor(id_lists[i]); att[j, :len(id_lists[i])] = 1
            self.body(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev), use_cache=False)
            last = (att.sum(1) - 1).to(self.dev)
            ar = torch.arange(len(idx), device=self.dev)
            for L in self.layers_saved:
                h = self.cap[L][ar, last].to(torch.bfloat16).cpu()
                for j, i in enumerate(idx):
                    res[L][i] = h[j]
        return {L: torch.stack(v) for L, v in res.items()}


def run_set(ex, name, rows, out, shard, tok_budget, t0):
    d = Path(out) / name; d.mkdir(parents=True, exist_ok=True)
    jl(d / 'rows.jsonl', rows)
    for k, s in enumerate(range(0, len(rows), shard)):
        if all((d / f'L{L}_{k:03d}.npy').exists() for L in ex.layers_saved):
            continue
        chunk = rows[s:s + shard]
        ids = [ex.tok(r['text'], add_special_tokens=True, truncation=True, max_length=MAX_TOK)['input_ids'] for r in chunk]
        H = ex.last_states(ids, tok_budget)
        for L in ex.layers_saved:
            np.save(d / f'L{L}_{k:03d}.tmp.npy', to_bf16_u16(H[L]))
            (d / f'L{L}_{k:03d}.tmp.npy').rename(d / f'L{L}_{k:03d}.npy')
        ntok = sum(map(len, ids))
        print(json.dumps({'set': name, 'shard': k, 'rows': s + len(chunk), 'of': len(rows), 'tokens': ntok,
                          'norm42_mean': float(H[LAYER].float().norm(dim=-1).mean()), 'sec': round(time.time() - t0)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', default='heldout,teval,train,probes,blends')
    ap.add_argument('--shard', type=int, default=10000)
    ap.add_argument('--tok-budget', type=int, default=65536)
    ap.add_argument('--n-train', type=int, default=0, help='limit train rows (smoke / pilot)')
    ap.add_argument('--blends', type=int, default=14400)
    ap.add_argument('--layers', default='', help="e.g. '42': save only these blocks (12 Oct: train2 needs L42 only)")
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sets = a.sets.split(',')
    smp = Path(a.sample)
    lays = tuple(int(x) for x in a.layers.split(',')) if a.layers else LAYERS_SAVED
    ex = Extractor(dev, lays) if set(sets) & {'heldout', 'teval', 'train', 'train2', 'probes'} else None
    for name, fn in (('heldout', 'heldout_fresh'), ('teval', 'heldout_teachereval'), ('train', 'train'), ('train2', 'train2')):
        if name not in sets:
            continue
        rows = pq.read_table(smp / f'{fn}.parquet').to_pylist()
        if name in ('train', 'train2') and a.n_train:
            rows = rows[:a.n_train]
        if SMOKE:
            rows = rows[:a.n_train or 20] if name == 'train' else rows[:20]
        for i, r in enumerate(rows):
            r['i'] = i
        run_set(ex, name, rows, out, a.shard, a.tok_budget, t0)

    if 'probes' in sets:
        C = json.load(open(Path(__file__).resolve().parents[1] / 'data/nladistil_concepts_20261003.json'))
        if SMOKE:
            C['concepts'] = [dict(c, sentences=c['sentences'][:2]) for c in C['concepts'][:2]]
        Vtr = load_set(out, 'train')
        mu = torch.tensor(Vtr[:50000]).mean(0)
        dirs = []
        for c in C['concepts']:
            ids = [ex.tok(s, add_special_tokens=True)['input_ids'] for s in c['sentences']]
            # mean over sentence tokens >= 2 needs all positions: run each sentence's prefixes ending at each token
            pref = [x[:j + 1] for x in ids for j in range(2, len(x))]
            H = ex.last_states(pref, a.tok_budget)[LAYER].float()
            m = H.mean(0) - mu
            dirs.append((m / m.norm()).numpy())
        Dc = np.stack(dirs).astype(np.float32); np.save(out / 'concepts.npy', Dc)
        held = rl(out / 'heldout' / 'rows.jsonl')
        Vh = load_set(out, 'heldout')
        rng = random.Random(7)
        inj_rows, inj = [], []
        for ci, c in enumerate(C['concepts']):
            for b in rng.sample(held, 2 if SMOKE else 10):
                v = torch.tensor(Vh[b['i']])
                for alpha in (0.0, 0.25, 0.5, 1.0):
                    inj.append(v + alpha * v.norm() * torch.tensor(Dc[ci]))
                    inj_rows.append({'i': len(inj_rows), 'concept': c['name'], 'keywords': c['keywords'], 'alpha': alpha,
                                     'base_i': b['i'], 'doc_id': b['doc_id'], 'text': b['text'][-2000:]})
        (out / 'inject').mkdir(exist_ok=True)
        np.save(out / 'inject' / f'L{LAYER}_000.npy', to_bf16_u16(torch.stack(inj))); jl(out / 'inject' / 'rows.jsonl', inj_rows)
        N = C['notes']; rng = random.Random(11)
        note_rows = []
        for k in range(8 if SMOKE else 120):
            name = rng.choice(N['first']) + ' ' + rng.choice(N['last'])
            city, job = rng.choice(N['cities']), rng.choice(N['jobs'])
            t = rng.choice(N['templates']).format(name=name, city=city, job=job)
            note_rows.append({'i': k, 'doc_id': f'note{k}', 'name': name, 'city': city, 'job': job, 'text': t})
        run_set(ex, 'notes', note_rows, out, a.shard, a.tok_budget, t0)
        print('probes', len(inj_rows), len(note_rows), flush=True)

    if 'blends' in sets:
        rows = rl(out / 'train' / 'rows.jsonl')
        V = load_set(out, 'train')
        pool = [r['i'] for r in rows if r.get('in_131k', True)]
        mu = torch.tensor(V[pool[:50000]]).mean(0)
        rng = random.Random(13)
        b_rows, B = [], []
        for k in range(a.blends):
            ia, ib = rng.sample(pool, 2)
            while rows[ia]['doc_id'] == rows[ib]['doc_id']:
                ia, ib = rng.sample(pool, 2)
            alpha = (0.25, 0.5, 1.0)[k % 3]
            va, vb = torch.tensor(V[ia]), torch.tensor(V[ib])
            u = vb - mu
            B.append(va + alpha * va.norm() * u / u.norm())
            b_rows.append({'i': k, 'a': ia, 'b': ib, 'alpha': alpha, 'doc_id': rows[ia]['doc_id'],
                           'text': rows[ia]['text'][-2000:]})
        (out / 'blends').mkdir(exist_ok=True)
        np.save(out / 'blends' / f'L{LAYER}_000.npy', to_bf16_u16(torch.stack(B))); jl(out / 'blends' / 'rows.jsonl', b_rows)
        print('blends', len(b_rows), flush=True)
    json.dump({'sec': time.time() - t0, 'sets': sets, 'subject': SUBJECT, 'layers': lays},
              open(out / f'extract_done_{"_".join(sets)}.json', 'w'))


if __name__ == '__main__':
    main()
