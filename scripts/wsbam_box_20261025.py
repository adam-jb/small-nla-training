"""WorkspaceBench agentic_misalignment on three L42 readers (25 Oct 2026): one note per prompt position of the 32
scenarios (scripts/wsbam_render_items_20261025.py) for the teacher, GRPO526 and AGI-r100. Reader sampling is the
bench's (T 1.0, top-k 64, top-p 0.95, 256 new tokens), k=1 as in the source. teacher/student come from
scripts/wsbt12_box_20261025.py; only capture and export differ (all positions of a scenario from one forward pass).
Stages:
  capture --items I --d D    Qwen3.6-27B block-42 output at every read position -> D/vec.npy, D/index.json
  teacher / student          as in wsbt12_box_20261025.py (k=1)
  export  --d D --out O      -> O/<reader>.jsonl, one row per scenario: {id, notes: [[pos, text], ...]}"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import wsbt12_box_20261025 as wb
from scripts.nladistil27b_common_20261010 import rl


@torch.no_grad()
def capture(a):
    from scripts.nladistil27b_extract_20261010 import Extractor
    items = rl(a.items)
    t0 = time.time()
    ex = Extractor('cuda', layers=(42,))
    V, index = [], []
    for it in items:
        ex.body(input_ids=torch.tensor([it['ids']], device='cuda'), use_cache=False)
        V.append(ex.cap[42][0, it['pos']].float().cpu().numpy())
        index += [[it['id'], p] for p in it['pos']]
    V = np.concatenate(V)
    d = Path(a.d); d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'vec.npy', V)
    json.dump(index, open(d / 'index.json', 'w'))
    print(json.dumps({'stage': 'capture', 'items': len(items), 'positions': len(index),
                      'norm_mean': float(np.linalg.norm(V, axis=1).mean()), 'sec': round(time.time() - t0)}), flush=True)


def export(a):
    index = json.load(open(Path(a.d) / 'index.json'))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.d).glob('notes_*.jsonl')):
        by = {}
        for r in rl(f):
            i, p = index[r['u']]
            by.setdefault(i, []).append([p, r['text']])
        with open(Path(a.out) / f"{f.stem.removeprefix('notes_')}.jsonl", 'w') as g:
            for i, notes in by.items():
                g.write(json.dumps({'id': i, 'notes': sorted(notes)}, ensure_ascii=False) + '\n')
        print(json.dumps({'export': f.stem, 'items': len(by), 'notes': sum(len(v) for v in by.values())}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['capture', 'teacher', 'student', 'export'])
    ap.add_argument('--items'); ap.add_argument('--d'); ap.add_argument('--out')
    ap.add_argument('--k', type=int, default=1); ap.add_argument('--bs', type=int, default=192)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model'); ap.add_argument('--name')
    ap.add_argument('--merged', default='/workspace/teacher_merged')
    a = ap.parse_args()
    {'capture': capture, 'teacher': wb.teacher, 'student': wb.student, 'export': export}[a.stage](a)


if __name__ == '__main__':
    main()
