"""Upload a finished 27B->2B student run (final_lm + final_adapter.pt + config/metrics/done) to a public HF model repo.
Usage (box): python scripts/nladistil27b_upload_student_20261012.py RUN_DIR REPO_ID"""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

run, repo = Path(sys.argv[1]), sys.argv[2]
how = sys.argv[3] if len(sys.argv) > 3 else ''  # optional extra line on how this student was trained (on-policy, GRPO, 4B)
api = HfApi(token=os.environ.get('HF_TOKEN') or open(os.path.expanduser('~/.hf_token')).read().strip())
api.create_repo(repo, repo_type='model', private=False, exist_ok=True)
readme = f"""# {repo.split('/')[1]}

{'Qwen3.5-4B' if '4b' in repo else 'Qwen3.5-2B'} student distilled from the 27B NLA `ceselder/qwen3.6-27b-nla-rl` (iter 400, Qwen3.6-27B layer 42).
Input: a layer-42 residual vector (5120-d, output of block 42, last token). `final_adapter.pt` maps it to 4 input
embedding slots placed where the teacher's marker token sits in the teacher's prompt; the LM then writes the explanation.
Training: SFT on 263k teacher explanations (sampled tokens + top-20 soft targets, weight 0.5), 1.5 passes, lr 1e-5
cosine. {how}
Code: `scripts/nladistil27b_train_student_20261010.py` (class Student, Adapter in
`scripts/nladistil27b_common_20261010.py`) in the lenses repo. Data: `adamjb111/nla-qwen36-27b-L42-distill-data`.
Eval: see `done.json` (held-out FVE by the teacher's AR on 1,020 fresh FineFineWeb vectors; teacher 0.754).
"""
(run / 'README.md').write_text(readme)
api.upload_folder(folder_path=str(run / 'final_lm'), path_in_repo='final_lm', repo_id=repo)
for f in ['final_adapter.pt', 'config.json', 'metrics.jsonl', 'done.json', 'README.md']:
    if (run / f).exists():
        api.upload_file(path_or_fileobj=str(run / f), path_in_repo=f, repo_id=repo)
print('uploaded', repo)
