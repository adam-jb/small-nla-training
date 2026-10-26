"""Push the 27B distillation inputs to the public HF dataset adamjb111/nla-qwen36-27b-L42-distill-data (10 Oct 2026).
Uploads: the sample parquets, vector shards (L28 / L42 / L52, bf16 bits in uint16 .npy) with rows.jsonl, and the
teacher's notes + top-20 shards. Token from HF_TOKEN. Resumable (upload_large_folder).
Usage: python scripts/nladistil27b_hf_upload_20261010.py --vec DIR --teacher DIR --sample DIR
"""
import argparse, os, shutil, tempfile
from pathlib import Path

REPO = 'adamjb111/nla-qwen36-27b-L42-distill-data'
CARD = """---
license: apache-2.0
tags: [interpretability, natural-language-autoencoder, nla, activations, qwen3.6, distillation]
pretty_name: Qwen3.6-27B layer-42 activations + NLA teacher explanations (distillation data)
---
# Qwen3.6-27B layer 28 / 42 / 52 activations with NLA explanations

Inputs for distilling the WorkspaceBench NLA [ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl)
(`av_base` + `av_rl_adapters/iter_000400`, layer 42) into small Qwen3.5 students.

- **Text**: the teacher's own training prefixes (FineFineWeb, from
  [ceselder/qwen3-8b-nla-L24-finefineweb-100k](https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k)),
  263k rows from 87.7k docs, up to 3 per doc, in a fixed doc order so `in_33k / in_66k / in_131k / in_263k` subsets nest.
  `heldout`: 1,020 prefixes from 340 fresh FineFineWeb docs (67 domains) never in the teacher's data, positions sampled as
  nanoNLA stage 0. `teval`: 1,020 rows of the teacher's own RL eval slice. `inject`, `notes`, `blends`: probe sets.
- **Vectors** (`vectors/{set}/L{layer}_{k}.npy`): output of decoder block L of Qwen/Qwen3.6-27B (HF `hidden_states[L+1]`)
  at the last token of each prefix fed alone; bf16 bits stored as uint16:
  `torch.from_numpy(np.load(f).view(np.int16)).view(torch.bfloat16)`. Row order = `vectors/{set}/rows.jsonl`.
- **Teacher** (`teacher/{set}/T_{k}.jsonl` + `T_{k}_top.npz`): one T=1 explanation per vector (max 256 tokens) and the
  teacher's top-20 log-probs at every token (`off`, `tok`, `tk_ids`, `tk_lp`). `train_s1` = a second sample for rows 0-65,499.

Made for a distillation study; see the project notes for the method. Text licence follows FineFineWeb (Apache-2.0).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vec', required=True)
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--sample', required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(REPO, repo_type='dataset', private=False, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='hfup_', dir=str(Path.home())))
    (stage / 'README.md').write_text(CARD)
    (stage / 'sample').mkdir()
    for f in Path(a.sample).glob('*'):
        shutil.copy(f, stage / 'sample' / f.name)
    shutil.copytree(a.vec, stage / 'vectors', ignore=shutil.ignore_patterns('*.tmp*', 'rawvar.json'))
    tdir = stage / 'teacher'; tdir.mkdir()
    for d in Path(a.teacher).iterdir():
        if d.is_dir() and not d.name.startswith('op_') and not d.name.endswith('_check'):
            shutil.copytree(d, tdir / d.name, ignore=shutil.ignore_patterns('*.tmp', '*_scored.jsonl'))
    api.upload_large_folder(repo_id=REPO, repo_type='dataset', folder_path=str(stage),
                            ignore_patterns=['*.tmp', '*.tmp.npy', '*_scored.jsonl', 'rawvar.json'])
    print('uploaded', REPO, flush=True)


if __name__ == '__main__':
    main()
