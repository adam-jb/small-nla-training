"""Intent probe (11 Oct 2026): what does the 27B NLA teacher write about intent-bearing states?

~62 prompts (web controls, WorkspaceBench multihop / association, harmless-payload jailbreaks + benign twins, agentic
temptations, user modelling, deception instructions, moral, role-bound). Layer-42 vector at: raw prompts -> last
token; WorkspaceBench multihop -> the item's own readout cell; chat prompts -> last token of the chat template with the
generation prompt (thinking off), i.e. the state just before the model answers. The teacher writes 2 samples per vector.
Usage: python scripts/nladistil27b_intent_probe_20261011.py --prompts runs/intent_probe_20261011/prompts.json --out DIR
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nladistil27b_common_20261010 import LAYER, extract_expl
from scripts.nladistil27b_extract_20261010 import Extractor
from scripts.nladistil27b_wsb_preview_20261010 import cell_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--merged', default='/data/teacher_merged'); ap.add_argument('--samples', type=int, default=2)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = json.load(open(a.prompts))
    vp = out / 'vectors_L42.npy'
    if not vp.exists():
        ex = Extractor('cuda'); pref = []
        for p in P:
            if p['kind'] == 'chat':
                msgs = ([{'role': 'system', 'content': p['system']}] if p['system'] else []) + [{'role': 'user', 'content': p['content']}]
                s = ex.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = ex.tok(s, add_special_tokens=False)['input_ids']
            else:
                ids = ex.tok(p['content'], add_special_tokens=True)['input_ids']
                if p['kind'] == 'wsb':
                    ids = ids[:cell_pos(ex.tok, ids, p['readout']) + 1]
            p['n_tok'] = len(ids); p['last_tok'] = ex.tok.decode(ids[-1:]); pref.append(ids)
        H = ex.last_states(pref, 32768)[LAYER].float().numpy()
        np.save(vp, H); json.dump(P, open(out / 'prompts_used.json', 'w'), indent=1)
        del ex; torch.cuda.empty_cache()
    H = np.load(vp); P = json.load(open(out / 'prompts_used.json'))
    from scripts.nladistil27b_teacher_20261010 import HFTeacher, merge_teacher
    T = HFTeacher(merge_teacher(a.merged), 'cuda', bs=64)
    with open(out / 'readouts.jsonl', 'w') as f:
        for s in range(a.samples):
            for p, r in zip(P, T.write(H, seed=100 + s)):
                f.write(json.dumps({'id': p['id'], 'family': p['family'], 'hidden': p['hidden'], 'last_tok': p['last_tok'],
                                    'sample': s, 'expl': extract_expl(r[0])[0], 'n_tok': len(r[1])}) + '\n')
    print('DONE', len(P), flush=True)


if __name__ == '__main__':
    main()
