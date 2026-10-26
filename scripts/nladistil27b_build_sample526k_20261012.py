"""Second batch of 263k train rows for the 27B NLA distillation (12 Oct 2026). Mac only, no GPU.

Same source and rule as nladistil27b_build_sample_20261010.py: the teacher's own training prefixes, docs in the same
fixed random order, each doc's rows in the same per-doc random order. The first batch took the first <= 3 rows of the
first 87,674 docs. This batch takes, in order: the first <= 3 rows of the docs the first batch never reached, then rows
4-6 of every doc in the same doc order, until 263k more rows. Checks that the rebuilt first batch matches train.parquet
row for row, and that no row repeats.
Writes OUT/train2.parquet (rows 263,000-525,999 in 'row'; in_526k flag) and OUT/manifest.json.
Usage: .venv/bin/python scripts/nladistil27b_build_sample526k_20261012.py --src data/nladistil27b_teacher_public_20261010 \
         --first data/nladistil27b_sample_20261010 --out data/nladistil27b_sample526k_20261012
"""
import argparse, json, random, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_build_sample_20261010 import RL_EVAL_START, SPLITS, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--first', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-doc', type=int, default=3)
    ap.add_argument('--n1', type=int, default=263_000)
    ap.add_argument('--n2', type=int, default=263_000)
    ap.add_argument('--seed', type=int, default=20261010)
    a = ap.parse_args()
    rows = []
    for name in SPLITS:
        rows += [r for r in load_split(a.src, name) if not (name == 'rl_shuf' and r['src_row'] >= RL_EVAL_START)]
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r['doc_id'], []).append(r)
    docs = sorted(by_doc)
    rng = random.Random(a.seed)
    rng.shuffle(docs)
    per = []                       # same RNG calls in the same order as build_train (it shuffles each doc as it goes)
    for d in docs:
        rs = sorted(by_doc[d], key=lambda r: (r['src'], r['src_row']))
        rng.shuffle(rs)
        per.append(rs)
    # first batch, rebuilt: first <= per_doc rows of docs in order, cut at n1
    first, n_docs1 = [], 0
    for rs in per:
        first.extend(rs[:a.per_doc]); n_docs1 += 1
        if len(first) >= a.n1:
            break
    first = first[:a.n1]
    old = pq.read_table(Path(a.first) / 'train.parquet', columns=['src', 'src_row']).to_pylist()
    assert len(old) == len(first) and all((o['src'], o['src_row']) == (f['src'], f['src_row']) for o, f in zip(old, first)), \
        'rebuilt first batch differs from train.parquet'
    used = {(r['src'], r['src_row']) for r in first}
    # second batch: unreached docs' first rows, then rows per_doc+1 .. 2*per_doc of every doc, skipping anything used
    cand = [r for rs in per[n_docs1:] for r in rs[:a.per_doc]]
    cand += [r for rs in per for r in rs[a.per_doc:2 * a.per_doc]]
    second = []
    for r in cand:
        k = (r['src'], r['src_row'])
        if k in used:
            continue
        used.add(k); second.append(r)
        if len(second) >= a.n2:
            break
    assert len(second) == a.n2, f'only {len(second)} new rows'
    for j, r in enumerate(second):
        r['row'] = a.n1 + j
        r['in_526k'] = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(second), out / 'train2.parquet', compression='zstd')
    first_docs = {r['doc_id'] for r in first}
    m = {'train2_rows': len(second), 'train2_docs': len({r['doc_id'] for r in second}),
         'train2_rows_in_new_docs': sum(r['doc_id'] not in first_docs for r in second),
         'pool_rows': len(rows), 'pool_docs': len(docs), 'first_docs': n_docs1,
         'train2_by_split': {s: sum(r['src'] == s for r in second) for s in SPLITS},
         'first_batch_rebuilt_ok': True, 'args': vars(a)}
    (out / 'manifest.json').write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
