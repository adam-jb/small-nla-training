# Plan and cost estimate: distil the WorkspaceBench 27B NLA (Qwen3.6-27B, layer 42) into a ~1.5B student (10 Oct 2026)

Status: estimate and plan only. Nothing launched. Written for Adam's question of 10 Oct (08:xx UTC).
Base for every number: the 7B run, `91_background_7b_distillation_findings_2026-10-03.md`.

## 1. Short answer

- **Cost: about $120 of GPU (range $85-200), about 60 box-hours on one A100 80 GB, so 2.5-3 days of wall clock.**
  The same schedule on the 7B cost about $56 of GPU. So the 27B is about 2x the 7B, not 3.6x: the teacher-side work
  scales with teacher size, but student training does not.
- Plus $5-15 of judge API for WorkspaceBench (one-position families) and $0 for blind truth judges (Sonnet subagents, as before).
- Optional: a second batch of 263k teacher pairs, if the data ladder says it is worth it: +$50.
- Later, the deception fine-tune of the finished student: about $10-25.

## 2. Facts about the teacher (checked 10 Oct)

- Repo `ceselder/qwen3.6-27b-nla-rl`. WorkspaceBench loads `av_base` + LoRA `av_rl_adapters/iter_000400` (methods.py in
  workspace-bench commit 92d763e). That is "the exact one". Layer 42 of 64, d_model 5120, no normalisation of the vector,
  norm-matched add at the output of block 1 at the marker token (id 158983).
- AV = full 27B (warm-start LoRA merged) + r64 LoRA. AR = 43-block cut of the 27B + value head (~17.5B), **latest only**
  (saved at step 600, per `ar_reconstructor/saved_at_step.txt`), so teacher and student must both be scored with this one AR.
- Built with EasyNLA. Warm-start labels: Claude Sonnet 4.6 summaries of FineFineWeb docs (texts from
  `ceselder/qwen3-8b-nla-L24-finefineweb-100k`, public). RL: 6x B200, batch ~504 x 8, held-out FVE 53% -> 76.6% at step 400.
- Qwen3.6-27B is a hybrid: 48 Gated-DeltaNet (linear attention) layers + 16 full-attention layers. Fast generation needs the
  `flash-linear-attention` / `causal-conv1d` kernels. This is the main speed risk.
- Size in bf16: AV 54 GB, AR ~35 GB. **Needs an 80 GB card.** A 40 GB A100 cannot hold the teacher in bf16.
- **No layer-42 activations are published** (only 64 examples in `data/example_activations.parquet`). We must make them.

## 3. Which student

The 7B student was Qwen2.5-1.5B. The 27B teacher has a 248k-token vocabulary; Qwen2.5 has 152k. The top-20 soft labels
and the on-policy teacher scoring both need the same tokenizer. So:

- **Recommended: Qwen3.5-2B** (same 248k tokenizer, hidden 2048, 24 layers, tied embeddings). About 0.5B of its 2B is the
  embedding table, so its body is about the size of Qwen2.5-1.5B's. Closest like-for-like.
- Cheap second size: Qwen3.5-0.8B (same tokenizer), one rung only, to see student size.
- Not recommended: Qwen2.5-1.5B again. Keeps the student fixed, but forces text-only SFT (no soft labels, no on-policy
  round), so the recipe would differ instead.

## 4. Cost, step by step (A100 80 GB at $1.99/h, central guess and range)

How the 27B numbers were made: teacher-side work x3.6 (27B vs 7.6B params; AR 17.5B vs ~5B); extraction only runs to
layer 42, so x2.4; student-side work x1.3 (larger vocab and body of Qwen3.5-2B).

| Step | 7B run (actual h) | 27B (h) | 27B ($) |
|---|---|---|---|
| Set-up: boot, download 90 GB, kernels, hook port, smoke test | ~1 | 3 (2-6) | 6 |
| Extract L42 vectors: 263k train + 1,020 held-out (30k two-context mixes are made from these, no forward pass) | ~1.5 | 3.5 (2.5-5) | 7 |
| Teacher writes 263k explanations, top-20 kept | 3.9 (230k) | 16 (11-25) | 32 |
| Student SFT, 263k pairs, 1.5 passes | 3.1 | 4 (3.5-5) | 8 |
| On-policy: op30k, then one 200k round (round 2 added nothing on the 7B, so skip it unless round 1 gives >0.005) | ~6 | 12 (9-16) | 24 |
| GRPO, ~400 steps x 512 rollouts, frozen 27B AR as reward | 5.8 (600 steps) | 7 (5-11) | 14 |
| Evals: teacher + each student on held-out FVE, injection, AR scoring | ~2 | 5 (3-7) | 10 |
| **Core path** | **~23 ($56 with side tests)** | **~50 (36-75)** | **~$100 ($72-150)** |
| Scaling extras: ladder students at 33k / 66k / 131k (nested subsets of the same labels) | 5.2 | 3.5 | 7 |
| Text-only 2B control (sees the text, not the vector), 131k | in first pass | 2 | 4 |
| 0.8B student, 131k | - | 1.5 | 3 |
| Fairness check: Qwen3.5-2B vs Qwen2.5-1.5B, both CE-only on the 7B teacher's 131k text labels | - | 3 | 6 |
| **Core + extras** | | **~60** | **~$120** |
| Optional: second 263k teacher batch + one more ladder rung | | +24 | +$50 |

Other boxes: GH200 (96 GB, H100-class, $2.29/h) would cut the teacher-heavy steps ~2x, so ~$80-90 in total, but it is
rarely free and has an ARM CPU (the DeltaNet kernels may need building; test first). H100 SXM at $4.29/h is no cheaper
than the A100. Our CLAUDE.md allows one A100 at a time, so the plan assumes one box, steps in series.

Biggest risk to cost: teacher speed. If HF `generate` with the hybrid model runs below ~400 tokens/s on the A100, the
teacher step could double (+$30). The pre-flight measures this in the first hour and stops if so (fixes: vLLM with a
small patch for the block-1 add, or a GH200).

## 5. One shot: how to make it safe

1. **Pre-flight (~$5, 2 h).** Check `nvidia-smi` shows 80 GB (Lambda gave a 40 GB card under the same name on 6 Oct;
   destroy it if so). Load AV + iter_000400 (merge the LoRA once) and the AR. Run 50 held-out vectors: outputs must look
   like the README's, AR FVE ~0.7+. Measure tokens/s. Stop rule: under 400 tok/s -> stop and rethink before the big step.
2. **Label once, at the largest size, in shards.** Every 10k pairs are written as a shard and pushed off the box (see 7).
   A crash then loses minutes, not the run. Keep the top-20 arrays this time (the 7B run threw them away and paid to redo them).
3. **Every ladder rung uses nested subsets of the same labels**, so the ladder costs only student time.
4. **Decision point after the ladder:** label the second 263k only if 131k -> 263k gains more than ~0.03 in "share of
   teacher" (below). Otherwise go straight to on-policy and GRPO.
5. **Same inputs as the 7B run.** Same Ultra-FineWeb parts (1500-1503) and WildChat shards (10-13), same held-out 340 docs,
   positions chosen by character offset (the two tokenizers split text differently), WildChat rendered with the Qwen3.6
   chat template, thinking off. Then every 7B-vs-27B comparison is paired on the same text.

## 6. Tracking scaling: what to measure so 7B and 27B can be compared

Raw FVE cannot be compared across teachers: each is scaled by its own model's "predict the mean" baseline (the README
says so too). So compare shares of the teacher, each scored with its own AR:

- **Share of teacher FVE**: student FVE / teacher FVE.
- **Vector share** (the key number): (student - text-only) / (teacher - text-only). On the 7B the text-only floor was 0.61
  of 0.73, so most FVE paid for describing the text. This share removes that. It needs the text-only control at 27B.
- **Injection kept**: student hits / teacher hits at α 0.25 / 0.5 / 1.0 (α relative to the vector's norm, same 30 concepts).
- **Truth kept**: blind Sonnet claim truth, student / teacher, same 80 vectors.
- **WorkspaceBench kept** (27B only, new and strong): student vs teacher pass rate on the one-position families at L42,
  both re-run on the current repo (the post's chart used an older scorer).

Then for each teacher fit share(n) = s∞ - b·n^(-c) on the rungs (33k / 66k / 131k / 263k [/ 525k]) and report two
numbers: **n95** (pairs needed to reach 95% of the teacher) and **s∞** (where the curve levels off), plus the lift from
on-policy and from GRPO as separate rows. The 7B already has these rungs; its fit gave s∞ ≈ 0.99 of the teacher.
My guess for the 27B with a 2B student: a later n95 and a lower s∞ (a bigger gap between teacher and student), with GRPO
doing more of the closing. That is what the run tests.

Honest limit: two teachers are two points, not a law, and they differ in more than size: model family, layer depth
(20/28 vs 42/64), label writer (Opus 4.5 vs Sonnet 4.6), recipe (kitft vs EasyNLA), student family. The fairness check in
the table measures the student-family part. **For a cleaner teacher-size series later:** kitft released same-recipe NLAs
for Gemma-3-12B (L32), Gemma-3-27B (L41) and Llama-3.3-70B (L53). Gemma 12B -> 27B is the same family, tokenizer and
recipe; SFT ladders only (no RL) would be ~$30-50 each.

## 7. Layer

- The teacher exists only at **layer 42**. ceselder's other 27B repos (`qwen3.6-27b-nla-L42`, `nla-qwen36-27b-matryoshka`)
  are also L42. WorkspaceBench scored the NLA only at L42 (J-lens and the other token lenses were scored at 11 layers).
- Another layer means training a new 27B NLA first: the RL alone was ~100 h on 6x B200, roughly $4,000 at Lambda prices.
  Out of scope.
- So: **L42 (recommended, the only layer with a teacher).** Cheap add-on: during extraction also save L28 and L52 (+20%
  extraction time, +5 GB), so a later study can feed a neighbour layer to the student's adapter or train probes. Not
  recommended: a student reading layer X trained on L42 labels; it learns to guess L42's description, not to read X.

## 8. Storage options

Sizes: one L42 vector = 10 KB (bf16). 263k vectors = 2.7 GB (5.4 GB at 525k; x3 with L28 + L52). Teacher text ~0.3 GB gz.
Top-20 arrays ~5 GB. Student checkpoints 4 GB each (keep 3-4). Teacher weights 90 GB: do not store, re-download from
HF each boot (minutes).

| Option | Where | Cost | Notes |
|---|---|---|---|
| A (recommended) | Public HF dataset for vectors + teacher text + top-20 + metadata; final students on HF; Lambda disk only while boxes run | $0/month | A useful public set: no one has published Qwen3.6-27B L42 activations at scale. Fast download to Lambda. Check the WildChat licence before publishing its text (or publish doc ids + vectors only). |
| B | Lambda persistent disk for everything except teacher weights | ~30-50 GB ≈ $6-10/month | Region-locked: the disk must be in the box's region. Delete after. |
| C | Cloudflare R2 (private) | $0.015/GB/month, no egress fee, ~$0.75/month | Only if you want it private. AWS S3 is worse: $0.09/GB to download out. |

## 9. After the general student: deception

- The 7B stance fine-tune took under an hour on the 1.5B, so the GPU cost is small ($2-5). The data is the cost.
- The subject model changes, so the twin set (`data/nla_deception_curated_v1`) must be re-answered by Qwen3.6-27B and its
  L42 vectors taken at the four sites (~1 h).
- Labels should check both that deception was asked for or motivated in the input **and** that the model actually
  deceived in its output (Adam, 9 Oct). Treat the "pasted twin" (honest system prompt, deceptive reply pasted in) as its
  own class ("saw it, did not choose it"), not as honest.
- Add cases where the model chooses to deceive with no instruction (incentive only, or a secret it was told to guard).
  WorkspaceBench's `agentic_misalignment` family (32 items, gated on Qwen3.6-27B: the model plans blackmail, leaks or
  self-exfiltration before writing it) is a ready test of self-chosen misbehaviour on this exact model.

## 10. Order of work

1. Pre-flight (stop rules above). 2. Extract + label 263k + held-out, shards to HF. 3. Ladder students (2B at
33k/66k/131k/263k, text-only 2B, 0.8B), all evals, fairness check. 4. Decide on the second 263k. 5. On-policy round(s).
6. GRPO. 7. Final evals incl. WorkspaceBench. 8. Destroy the box; log; then a separate go-ahead for the deception phase.

Naming, following the 7B run: `nladistil_sftkd_qwen35-2b_vec_from-nla-qwen36-27b-L42-rl400_pairs263k_2026MMDD`,
`..._onpolicy_..._op200k-r1_...`, `..._grpo_..._ar-reward_...`, with `_stepNNN` for checkpoints.

## 11. Compared with training a small NLA for the 27B without the teacher (added 10 Oct, Adam's follow-up)

"From scratch" here = the NLA recipe itself on a 2B reader: warm-start SFT on LLM-written summaries, then GRPO with
reconstruction as the reward. Same A100 80 GB, $1.99/h.

Inputs that make it cheaper than it sounds: the warm-start texts and Sonnet 4.6 summaries are public
(`ceselder/qwen3-8b-nla-L24-finefineweb-100k`, 247k AV rows), so labels cost $0; we only regenerate L42 vectors (~$7).
The reward needs a reader that rebuilds the vector (AR). Only two choices: borrow ceselder's frozen 27B AR, or train our own.

GRPO cost per rollout: the 7B run did 512 rollouts in 33 s. With the 17.5B AR as reward and a 2B student, I expect
~60-70 s per 512, so ~$0.07 per 1,000 rollouts and ~2 h per 100k.

How many rollouts are needed: the evidence is the three runs we know of.
- Anthropic 7B: ~4.3M rollouts from SFT ~0.35 to 0.75.
- ceselder 27B: ~1.6M rollouts (400 steps x ~4k) from 0.53 to 0.766.
- Our 1.5B from the distilled 0.714: plateau after ~150k rollouts.
A 2B reader warm-started on summaries will likely start lower than the 27B did (guess 0.35-0.45 on this AR), so 1.5-4M
rollouts, with no evidence it reaches the teacher at all.

| Route | What it does | GPU $ | Box hours | Chance of reaching the teacher | Notes |
|---|---|---|---|---|---|
| 1. Distil, full schedule (the plan above) | 263k teacher pairs -> SFT -> on-policy -> GRPO | ~$100-120 | ~50-60 | High (the 7B tied) | Gives the scaling comparison with the 7B |
| 2. Distil, lean | 263k teacher pairs -> SFT -> GRPO (skip on-policy; on the 7B it gave +0.016, which GRPO also gives) | ~$75-90 | ~40 | High | **Best value.** Loses one like-for-like step with the 7B |
| 3. From scratch, borrowed AR | Summaries SFT -> GRPO 1.5-4M rollouts vs ceselder's frozen AR | ~$130-300 | 70-150 (3-6 days) | Unknown; maybe below | Weak start means more risk of gaming the AR. The AR was trained beside the teacher, so it still pulls the reader toward the teacher's style |
| 4. Fully from scratch | Own labels, own AR (LoRA on a 17.5B AR, or a weaker small AR), then GRPO | ~$400-1,500+ | Days, wants 2-8 GPUs | Unknown | A full fine-tune of a 17.5B AR does not fit on one A100. New Sonnet labels would add ~$300-700. Not worth it |
| 5. Mix | 66k teacher pairs + 247k public summaries -> SFT -> GRPO ~300-500k rollouts | ~$60-80 | ~35 | Medium-high | Cheapest guess that may still tie; untested |

Why distilling wins: one teacher pair carries a whole explanation plus the top-20 at every token. One rollout carries
a single reward number. Teacher pairs cost ~0.22 s of 27B time each, and a rollout ~0.12 s. So teacher pairs buy far more
learning per dollar at the start. RL is the cheap way to close the last few points, once the student is close.

When from scratch is worth it: only if we want a reader that is not tied to the teacher's habits. For example, warm-start
labels written to name intent and stance, for the deception aim. Even then, do it as a fine-tune on top of the distilled
student (cheap, under an hour, as on 8 Oct), not as a fresh RL run.

## 12. Teacher card: what it is, how it was trained, what it likely cost (added 10 Oct)

**Exact teacher.** https://huggingface.co/ceselder/qwen3.6-27b-nla-rl (by Celeste, MATS; built with EasyNLA,
https://github.com/asherps/EasyNLA, on nanoNLA). We use `av_base/` + `av_rl_adapters/iter_000400` (what WorkspaceBench
loads) and `ar_reconstructor/` to score. Subject model Qwen/Qwen3.6-27B, layer 42.

**Data.** 100k FineFineWeb docs x 10 positions. Labels: Claude Sonnet 4.6 descriptions (~80-100 words, max 300 tokens)
of what the model must be "thinking about" at the end of the prefix, first written for a Qwen3-8B L24 NLA and reused
here, with the vectors regenerated at L42. Public: https://huggingface.co/datasets/ceselder/qwen3-8b-nla-L24-finefineweb-100k
- AV warm-start: 247,261 rows. AR warm-start: 247,358 rows. RL prompts: 499,846 rows (no labels needed), 50k held out.

**Training.**
1. AV warm-start: LoRA r128 (rsLoRA, every attention / MLP / DeltaNet projection) on the 27B, 3,864 steps, then merged
   into `av_base`. Batch size not stated; 3,864 steps over 247k rows fits batch 64, one pass (my inference).
2. AR warm-start: 43-block cut of the 27B + value head, 3,865 steps.
3. GRPO: 504 prompts x 8 samples = 4,032 rollouts a step; lr 1e-4; r64 rsLoRA actor; k3 KL 0.01 (top-64); length penalty
   0.01/token over 192; max 256 new tokens; AR co-trained (r64 LoRA, lr 8e-5); generation through vLLM with a
   `qwen3_5_wrapper` (so the block-1 add already works in vLLM: useful for our teacher-speed risk); 6x B200, ~440 s a step.
   Step 400 = ~1.6M rollouts over ~200k prompts (0.4 of the RL set). FVE (on only 128 eval prompts, so +-1 point):
   53.0% at step 0, 72.6% at 100, 76.6% at 400, best 77.7% at 540. Adapters go to step 800; the logged curve and the AR stop at 600.

**Likely cost (my estimate; nothing stated by the author).**
| Part | Estimate | How |
|---|---|---|
| Sonnet 4.6 labels, ~495k rows | ~$1,300-2,600 (half with batch API) | ~1k input + ~150 output tokens a row at $3 / $15 per M. Paid once for the 8B NLA, then reused, so ~$0 extra for the 27B |
| Vectors at L42 (~1M prefixes) | ~$10-40 | a few GPU-hours |
| AV + AR warm-start | ~$100-300 | ~15-40 B200-hours for LoRA on 27B over ~250k rows each |
| GRPO to step 400 | ~$1,000-2,000 | 400 x 440 s x 6 GPUs = ~290 B200-hours at $3.5-7/h (Lambda list ~$6.7-7) |
| GRPO to step 800 | ~$2,000-4,000 | double |
| **Teacher at step 400, all in** | **~$2,500-5,000** (~$1,100-2,300 without the shared labels) | |

So our distillation (~$75-120) would be about 2-5% of what the teacher cost, and about a tenth of its RL alone.
