"""Build the input sample for the 27B NLA distillation (10 Oct 2026). Mac only, no GPU, no model weights.

Train rows are the teacher's own training prefixes (ceselder/qwen3-8b-nla-L24-finefineweb-100k, AV-SFT + AR-SFT + RL
splits; the RL split's last 50k rows, which the teacher held out for eval, are excluded). Docs are taken in a fixed
random order, up to --per-doc rows each, so every ladder rung is a prefix of the same list (33k < 66k < 131k < 263k).
Held-out 'fresh': new FineFineWeb docs from the same 67 domain files, past the docs the teacher's corpus could have
used, positions sampled exactly as nanoNLA stage 0 does (Qwen3-8B tokens, index >= 50, per-doc RNG), de-duplicated
against every teacher doc. Held-out 'teacher_eval': rows from the teacher's own RL eval slice (a second check).

Writes OUT/train.parquet, OUT/heldout_fresh.parquet, OUT/heldout_teachereval.parquet, OUT/manifest.json.
Usage: .venv/bin/python scripts/nladistil27b_build_sample_20261010.py --src data/nladistil27b_teacher_public_20261010 \
         --out data/nladistil27b_sample_20261010 [--smoke]
"""
import argparse, hashlib, json, random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

RUNGS = (33_000, 66_000, 131_000, 263_000)
SPLITS = ('av_sft_shuf', 'ar_sft_shuf', 'rl_shuf')
RL_EVAL_START = 449_846          # run_config.yaml: eval_skip_rows (the last 50k RL rows are the teacher's eval set)
MIN_POSITION = 50                # nanoNLA stage0_extract._MIN_POSITION
CORPUS_PER_DOMAIN = 100_000 // 67 + 200   # build_finefineweb_corpus.py: per_domain_target; later docs are unseen
FFW = 'm-a-p/FineFineWeb'


def head_key(text):
    return hashlib.sha1(text[:300].encode('utf-8', 'ignore')).hexdigest()


def load_split(src, name):
    t = pq.read_table(f'{src}/{name}.parquet', columns=['doc_id', 'detokenized_text_truncated', 'n_raw_tokens'])
    d = t.to_pydict()
    return [{'src': name, 'src_row': i, 'doc_id': d['doc_id'][i], 'text': d['detokenized_text_truncated'][i],
             'n_raw_tokens_qwen3_8b': d['n_raw_tokens'][i]} for i in range(t.num_rows)]


def build_train(src, per_doc, seed, n_max):
    rows, teval = [], []
    for name in SPLITS:
        for r in load_split(src, name):
            (teval if name == 'rl_shuf' and r['src_row'] >= RL_EVAL_START else rows).append(r)
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r['doc_id'], []).append(r)
    docs = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(docs)
    out = []
    for d in docs:
        rs = sorted(by_doc[d], key=lambda r: (r['src'], r['src_row']))
        rng.shuffle(rs)
        out.extend(rs[:per_doc])
        if len(out) >= n_max:
            break
    out = out[:n_max]
    for i, r in enumerate(out):
        r['row'] = i
        for n in RUNGS:
            r[f'in_{n // 1000}k'] = i < n
    teacher_keys = {head_key(r['text']) for r in rows + teval}
    return out, teval, teacher_keys, len(by_doc)


def build_fresh(n_docs, per_doc, seed, teacher_keys, smoke):
    """New FineFineWeb docs from each domain's first file, after the first CORPUS_PER_DOMAIN kept docs."""
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    special = set(tok.all_special_ids)
    info = HfApi().dataset_info(FFW)
    first = {}
    for s in sorted(info.siblings, key=lambda x: x.rfilename):
        n = s.rfilename
        if n.endswith('.jsonl') and '/' in n:
            first.setdefault(n.split('/')[0], n)
    domains = sorted(first)[:3] if smoke else sorted(first)
    per_dom = -(-n_docs // len(domains))
    fs, rng, out = HfFileSystem(), random.Random(seed), []
    for dom in domains:
        kept = picked = 0
        with fs.open(f'datasets/{FFW}/{first[dom]}', 'r') as f:
            for line in f:
                text = (json.loads(line).get('text') or '')
                if len(text) < 200:
                    continue
                kept += 1
                if kept <= CORPUS_PER_DOMAIN + 50:      # could be in the teacher's corpus (+50 margin)
                    continue
                if head_key(text) in teacher_keys:
                    continue
                ids = tok(text, truncation=True, max_length=2048)['input_ids']
                doc_id = f'ffw_fresh:{dom}:{kept}'
                cand = [i for i, t in enumerate(ids) if i >= MIN_POSITION and t not in special]
                if len(cand) < per_doc:
                    continue
                r = random.Random(hashlib.sha256(f'{seed}|{doc_id}'.encode()).digest())
                for p in sorted(r.sample(cand, per_doc)):
                    out.append({'doc_id': doc_id, 'domain': dom, 'pos_qwen3_8b': p, 'n_raw_tokens_qwen3_8b': p + 1,
                                'text': tok.decode(ids[:p + 1])})
                picked += 1
                if picked >= per_dom:
                    break
        print(f'  fresh {dom}: {picked} docs', flush=True)
    by_doc = {}
    for r in out:
        by_doc.setdefault(r['doc_id'], []).append(r)
    keep = sorted(by_doc)
    rng.shuffle(keep)
    keep = set(keep[:n_docs])
    return [r for r in out if r['doc_id'] in keep]


def write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-doc', type=int, default=3)
    ap.add_argument('--n-train', type=int, default=RUNGS[-1])
    ap.add_argument('--fresh-docs', type=int, default=340)
    ap.add_argument('--fresh-per-doc', type=int, default=3)
    ap.add_argument('--teacher-eval', type=int, default=1020)
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train, teval, keys, n_docs_pool = build_train(a.src, a.per_doc, a.seed, a.n_train)
    write(train, out / 'train.parquet')
    rng = random.Random(a.seed + 1)
    teval_pick = rng.sample(teval, a.teacher_eval)
    write(teval_pick, out / 'heldout_teachereval.parquet')
    fresh = build_fresh(30 if a.smoke else a.fresh_docs, a.fresh_per_doc, a.seed, keys, a.smoke)
    write(fresh, out / 'heldout_fresh.parquet')
    m = {'train_rows': len(train), 'train_docs': len({r['doc_id'] for r in train}), 'pool_docs': n_docs_pool,
         'rungs': {f'{n // 1000}k': sum(r[f'in_{n // 1000}k'] for r in train) for n in RUNGS},
         'rung_docs': {f'{n // 1000}k': len({r['doc_id'] for r in train if r[f'in_{n // 1000}k']}) for n in RUNGS},
         'train_by_split': {s: sum(r['src'] == s for r in train) for s in SPLITS},
         'teacher_eval_rows': len(teval_pick), 'fresh_rows': len(fresh), 'fresh_docs': len({r['doc_id'] for r in fresh}),
         'fresh_domains': len({r['domain'] for r in fresh}),
         'fresh_in_teacher': sum(head_key(r['text']) in keys for r in fresh), 'args': vars(a)}
    (out / 'manifest.json').write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
