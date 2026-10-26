# Reading list: NLA distillation work (7B done, 27B in progress), for other agents (10 Oct 2026)

Start here, in this order. Paths are relative to the repo root (`<repo root>`).

## 1. Project docs (this repo)

| File | Why read it |
|---|---|
| `docs/docs.md` | Index of every reference file |
| `docs/90_background_7b_distillation_plan_2026-10-03.md` | 7B plan: teacher facts (layer hook, injection, prompt), measures, controls |
| `docs/91_background_7b_distillation_findings_2026-10-03.md` | **Main results**: 7B -> 1.5B ties the teacher (FVE 0.735 vs 0.731) via 263k SFT + on-policy + GRPO; data ladder; text-only control; injection; aggregation; deception probes; bugs hit |
| `docs/92_background_what_is_known_about_training_nlas_2026-10-08.md` | What is known about training NLAs well (Anthropic recipe, limits, our findings) |
| `docs/04_distillation_cost_and_scaling_plan_2026-10-10.md` | 27B cost model, student choice, storage, from-scratch comparison (s.11), teacher card and its likely cost (s.12) |
| `docs/03_distillation_plan_2026-10-10.md` | **Current 27B plan.** Read sections 12-16 for the live version: data, loss, arms, stop rules, monitoring (s.15), decisions |
| `docs/workspacebench_format_and_requirements_notes_2026-10-08.md` | WorkspaceBench formats, families, baselines |
| `docs/workspacebench_full_cost_and_judge_strictness_notes_2026-10-10.md` | Full bench cost; what the judges demand (name the target; quote is from the readout) |
| `docs/template_reader_on_workspacebench_and_flow_nla_assessment_2026-10-10.md` | Template readers vs the bench; Flow-NLA assessment |
| `docs/nla_targeted_training_for_deception_notes_2026-10-07.md`, `nla_deception_stance_sft_findings_2026-10-08.md` | Why NLAs miss intent and how a stance fine-tune fixes it |
| `docs/experiment_log.txt` | Short dated log of every run |
| `.claude` memory (`~/.claude/projects/-Users-adambricknail-Desktop-lenses/memory/`) | Ops lessons: Mac sleep + box failsafe, Lambda gotchas, budget, monitoring rule |

## 2. Code (this repo)

| Path | What |
|---|---|
| `scripts/nladistil_*_20261003.py` ... `_20261007.py` | 7B pipeline: extract, teacher sample + top-20, student SFT/KD (`train_student`, loss in `batch_loss`), AR scoring, on-policy, aggregation, GRPO (`rl_grpo_20261006`), judges, scaling page |
| `scripts/nladistil27b_build_sample_20261010.py` | 27B input sample: teacher's own prefixes, nested ladder rungs, fresh held-out docs |
| `runs/nladistil_*` | 7B run outputs (metrics, generations, AR scores; no weights). Weights on Lambda disk `claude-lens-results-20260929` (us-east-1) under `nladistil_20261003/` |

## 3. External: teachers, data, code

- 27B teacher (WorkspaceBench's NLA): https://huggingface.co/ceselder/qwen3.6-27b-nla-rl (`av_base` + `av_rl_adapters/iter_000400`; AR `ar_reconstructor`; `run_config.yaml`, `nla_meta.yaml`)
- Its training text + Sonnet labels: https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k (source corpus m-a-p/FineFineWeb)
- 7B teacher: https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av and `-ar`; same-recipe NLAs: kitft Gemma-3 12B L32, Gemma-3 27B L41, Llama-3.3-70B L53
- EasyNLA (built the 27B): https://github.com/asherps/EasyNLA (injection `nla/injection.py`, AR `nla/utils/critic.py`, FVE `nla/schema.py`)
- nanoNLA (data pipeline): https://github.com/ceselder/nanoNLA (`nla/datagen/stage0_extract.py`, `scripts/build_finefineweb_corpus.py`)
- Anthropic NLA paper: https://transformer-circuits.pub/2026/nla/index.html ; training repo https://github.com/kitft/natural_language_autoencoders
- WorkspaceBench: post https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the ; repo https://github.com/camilablank/workspace-bench (NLA reader in `src/wsbench/produce/methods.py`)
- Workspace paper (J-lens): https://transformer-circuits.pub/2026/workspace/index.html
- Flow-NLA: https://arxiv.org/html/2609.33702v1

## 4. Key facts in one place

- Conventions (27B teacher): layer L = output of decoder block L (HF `hidden_states[L+1]`); vector added at the output
  of block 1 at the marker token (id 158983, neighbours 29 / 510) as h + |h| v/|v|; AR reads the explanation in
  `Summary of the following text: <text>{explanation}</text> <summary>`, value head on the last token; FVE =
  1 - MSE / predict-the-mean variance, pred and gold both scaled to norm sqrt(d).
- 7B lessons: data is the first limit (ladder 66k/131k/263k = 0.540/0.655/0.698); every rung follows the same early curve
  (~0.30 FVE at 30k seen, ~0.49 at 60k); on-policy rounds lift injection most; GRPO closes the last FVE gap in ~150k
  rollouts; a text-only reader gets 0.61 of 0.73 FVE, so FVE mostly rewards describing the text; NLAs miss intent
  because their labels never needed it.
- Ops: Lambda's $1.99 A100 is 40 GB (the 27B needs > 54 GB); use a box-side watchdog (the Mac sleeps); check boxes every
  5 min; destroy boxes when done; never touch boxes you did not start.
