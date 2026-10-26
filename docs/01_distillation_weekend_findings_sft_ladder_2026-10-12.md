# Weekend findings, 10-12 Oct 2026: distilling the 27B NLA into Qwen3.5-2B

Written 12 Oct (early UTC) at the end of the weekend run. Covers everything done Sat 10 Oct to Mon 12 Oct 03:32 UTC.
Nothing ran on Fri 9 Oct (only Adam's notes). Live log: `02_distillation_live_log_526k_onpolicy_grpo_2026-10-10_to_14.md`. Plan:
`03_distillation_plan_2026-10-10.md` (s.12-16). Results table: `runs/nladistil27b_results_20261011/summary.md`.

## 0. In one paragraph

A 2B student now reads the 27B NLA's layer-42 vectors at **90% of the teacher**: held-out FVE 0.680 [0.665, 0.692] vs
0.754, after plain copying (SFT + top-20 soft targets) on 263k teacher notes, 3.7 h on one RTX PRO 6000. It is public:
`adamjb111/nld27-sftkd-qwen35-2b-from-nla-qwen36-27b-L42-pairs263k-ep1.5-20261012`. The data ladder (33k / 66k /
131k / 263k = 0.468 / 0.566 / 0.635 / 0.680) gains less with each doubling (+0.10, +0.07, +0.05), so more of the same
data alone would likely stop near ~0.71-0.73. The student copies the teacher's style and outline but is weaker where it
matters for interpretability: it names injected concepts far less often at low strength (α 0.25: 22% vs the teacher's
80%; α 0.5: 83% vs 100%) and names planted facts less (job 42% vs 61%, city 14% vs 23%). Swapping 11% of pairs for two-document blends gave a small lift (0.584 vs 0.566; injection α 0.5
0.73 vs 0.62), borderline with one seed. A student that reads the
last 192 tokens of the text instead of the vector gets 0.539 at 66k vs 0.566 for the vector student, so most of what
the AR scores can be got from the text; the vector adds a little on natural text and is the only route for injected
concepts. A 0.8B student is clearly worse (0.463 vs 0.566 at 66k); lr 2e-5 is slightly better than 1e-5 (0.578 vs
0.566, within noise). Not done in the 8 h window: the on-policy round and GRPO (the stages most likely to close the
injection gap, as on the 7B), the 4B student, the two-notes arm and the WorkspaceBench preview. Weekend spend ~$50 of
the $150 cap.

## 1. The aim

Teach a small model (Qwen3.5-2B, with a small adapter that turns the vector into 4 input slots) to write the same
explanation of a layer-42 vector that the public 27B NLA (`ceselder/qwen3.6-27b-nla-rl`, iter 400) writes, by copying
its outputs (sampled tokens + top-20 soft targets). Score = FVE: the 27B's own reconstructor (AR) rebuilds the vector
from the explanation; FVE = share of the vector's variance it gets back. Held-out = 1,020 vectors from 340 fresh
FineFineWeb docs (all 67 domains, none in the teacher's data). The teacher scores 0.754 on it.

## 2. What happened, in order

| When (UTC) | What | Result |
|---|---|---|
| Sat 10 Oct, morning | Plans and costs (no GPU): full plan, cost plan, WorkspaceBench cost, template-reader view, reading list | Plan ~$120 GPU; Adam's selection 0, 1, S1-S5, OP, R1, E, P; cap $150 |
| Sat 10 Oct ~12:30 | Input sample built on the Mac | 263k train rows from 87,674 teacher docs (≤ 3 per doc); nested rungs 33k/66k/131k/263k; held-out 1,020 |
| Sat 10 Oct 13:00-17:50 | Nebius B300 spot box | 141 GB downloads in 9 min; vectors for all sets at L28/L42/L52 (1.6 h); pre-flight passed (our vectors match the 64 published ones, cos 0.9993); teacher held-out FVE 0.754 [0.744, 0.764], shuffled -0.90, 99.9% parsed |
| Sat 10 Oct 17:50 - Sun 11 Oct 04:20 | Nebius stall | Spot price sat above our $2.75 cap: 45 stops, ~2.5 h of real work. Only 10k of 263k labels. ~$10 spent. All Nebius resources deleted |
| Sun 11 Oct 04:45 | Move to Vast.ai RTX PRO 6000 (96 GB, Spain, $1.44/h) | State restored from the HF dataset; HF teacher ~1,010 tok/s |
| Sun 11 Oct 08:00-11:00 | vLLM teacher fixed and switched on | ~1,400 tok/s, same quality (held-out FVE 0.756 vs 0.754); saved ~3.5 h and ~$5 |
| Sun 11 Oct (side work) | GPU and speed study; vllm-lens fast-path patch; fast student sampling; intent probe; template-NLA reuse note | See s.4 |
| Sun 11 Oct 16:57 / 17:29 / 19:50 | Teacher labels done: train 263,000 (99.92% parsed), blends 14,400, second notes 65,500 | All public on HF |
| Sun 11 Oct 19:55 | First student run (pilot) crashed: out of GPU memory | Box sat idle ~1 h; failsafe tried to stop it but its stop call was broken |
| Sun 11 Oct 21:00 onward (this session) | Fix + student ladder on two boxes | See s.3 |

## 3. Student results

All runs: Qwen3.5-2B unless named, 4-slot adapter, stage A 10k (adapter only, lr 1e-3) then stage B all weights,
lr 1e-5 cosine, batch 32, 1.5 passes, loss 0.5 CE on the teacher's token + 0.5 soft CE on its top-20. FVE on the 1,020
fresh held-out vectors (teacher 0.754); teacher-eval = 1,020 rows from the teacher's own RL eval slice (teacher 0.769).
Inject = share of notes naming a concept added to the vector at strength α (keyword match; teacher 0.06 / 0.80 / 1.00
/ 1.00). Notes = vectors from short notes about a person: city / job named (teacher 0.23 / 0.61). The teacher refs
for inject, notes and teacher-eval were computed this session from the same files and rule ($0 / ~2 min GPU).

| Run | Pairs | Held-out FVE [95% CI] | Share of teacher | Teacher-eval | Inject α 0 / .25 / .5 / 1 | Notes city / job | GPU h |
|---|---|---|---|---|---|---|---|
| Teacher (27B) | - | 0.754 [0.744, 0.764] | 1.00 | 0.769 | 0.06 / 0.80 / 1.00 / 1.00 | 0.23 / 0.61 | - |
| **2B, 263k (main, saved)** | 263k | **0.680 [0.665, 0.692]** | **0.90** | 0.695 | 0.07 / 0.22 / 0.83 / 1.00 | 0.14 / 0.42 | 3.71 |
| 2B, 131k | 131k | 0.635 [0.620, 0.650] | 0.84 | 0.646 | 0.07 / 0.16 / 0.70 / 0.98 | 0.13 / 0.27 | 2.03 |
| 2B, 66k | 66k | 0.566 [0.547, 0.584] | 0.75 | 0.593 | 0.07 / 0.15 / 0.62 / 0.95 | 0.12 / 0.18 | 1.03 |
| 2B, 33k | 33k | 0.468 [0.445, 0.491] | 0.62 | - | - | - | 0.55 |
| 2B, 10k pilot (1 pass) | 10k | 0.141 [0.113, 0.165] | 0.19 | - | - | - | 0.20 |
| 2B, 66k, lr 2e-5 | 66k | 0.578 [0.559, 0.597] | 0.77 | - | - | - | 1.04 |
| 0.8B, 66k | 66k | 0.463 [0.441, 0.484] | 0.61 | 0.501 | 0.07 / 0.13 / 0.50 / 0.81 | 0.10 / 0.14 | 0.68 |
| 2B text-only (last 192 text tokens, no vector), 66k | 66k | 0.539 [0.517, 0.559] | 0.71 | 0.552 | n/a | n/a | 1.40 |
| 2B, 66k, 11% two-document blends (S3) | 66k (59k real + 7k blends) | 0.584 [0.565, 0.602] | 0.77 | 0.599 | 0.08 / 0.18 / 0.73 / 0.98 | 0.12 / 0.25 | 1.08 |

All parse 99.9-100%; shuffled-vector FVE -0.78 to -0.87 for every run (so the scores come from the vector, not from
a generic note). Notes are ~123 words, the same length and three-part outline as the teacher's.

**Held-out FVE by examples seen, against the 7B distillation** (7B: Qwen2.5-7B NLA -> 1.5B student, teacher 0.731):

| Seen | 30k | 60k | 120k | 240k | end (394k) |
|---|---|---|---|---|---|
| 27B -> 2B, 263k run | 0.423 (0.56) | 0.521 (0.69) | 0.602 (0.80) | 0.653 (0.87) | 0.680 (0.90) |
| 7B -> 1.5B, 263k run | 0.296 (0.40) | 0.488 (0.66) | 0.634 (0.86) | 0.685 (0.94) | 0.698 (0.95) |

**Ladder (each a fresh run, 1.5 passes) against the 7B:** 27B 66k 0.75 / 131k 0.84 / 263k 0.90 of teacher; 7B 0.74 /
0.90 / 0.95. Same start, the 27B student then falls behind: its teacher is harder to copy (its notes carry more
specific detail per token; end CE 1.81 vs 1.44 on the 7B, though the tokenizers differ).

What it means:
1. **Copying works and is cheap.** 90% of a 27B NLA in a 2B model for ~$5 of training on top of ~$25 of teacher labels.
   The 2B is 13x smaller than the teacher, and vLLM sampling of the student (built, not yet wired in) is 2.6x faster
   than the HF loop used here.
2. **More of the same data is running out of steam.** Gains per doubling fall by about a third each time (+0.098,
   +0.069, +0.045). A 526k batch would likely add ~0.03 (to ~0.71) for another ~$30-40 of teacher labels. Not worth it
   before the on-policy and RL stages: on the 7B, one on-policy round took injection at α 0.5 from 60% to 76% and GRPO
   added ~7 points more (plan s.13).
3. **The gap that matters is injection at low strength.** At α 0.25 the teacher names the concept 80% of the time and
   the student 22%; at α 0.5, 100% vs 83%. False alarms at α 0 match the teacher (7% vs 6%). Data scale helps slowly
   (0.15 -> 0.16 -> 0.22 at α 0.25 across 66k -> 263k). The 7B showed the same pattern and on-policy + GRPO closed much of
   it. Caveat: the keyword rule is strict; e.g. "dogs" injected at α 0.5 gave "pets ... Beko Cat", scored as a miss.
4. **The vector beats the text, but not by much on natural text.** Text-only reaches 0.539 vs 0.566 for the vector at
   66k (teacher-eval 0.552 vs 0.593). On the 7B, text-only (0.613 at 32.6k) beat the first-pass vector student (0.440).
   Here the vector wins from ~40k examples on. Most of what the AR rewards (genre, topic, the last phrase) is in the text.
   So FVE alone cannot show that a reader uses the vector; the injection test can, and there the text-only model is at
   chance by design.
5. **Size: 2B is the right size, 0.8B is not.** 0.8B loses 0.10 FVE at 66k and 12 points of injection at α 0.5. 4B not run.
6. **Blends (S3) help a little, and most on injection.** Swapping 11% of the 66k pairs for teacher notes on
   two-document blends: natural held-out 0.584 vs 0.566 (+0.018, CIs just overlap), teacher-eval 0.599 vs 0.593,
   injection α 0.5 0.73 vs 0.62, job 0.25 vs 0.18; false alarms 8% vs 7%. This passes the plan's rule (S3 counts only if
   natural FVE rises), but only just, and with one seed. The injection lift was the expected risk (blends look like
   injection), yet natural FVE rose too. Worth a check at 131k or a second seed before adding to the main recipe.
7. **Learning rate:** 2e-5 edges 1e-5 (+0.012, inside the noise); the plan's rule would pick 2e-5 for any re-run.
8. **Example (held-out row 400):** student "Medical/cosmetic treatment article ... a study's findings on The Ordinary
   Vitamin C serum ..."; teacher "Medical/educational article comparing two skincare ingredients ... split-face study
   ... retinol". Both get the genre, the study set-up and the mid-sentence ending; the student swaps in a wrong product.

## 4. Side findings from the weekend (details in their own docs)

**Teacher speed on the RTX PRO 6000** (`06_gpu_choice_and_inference_speed_2026-10-11.md`):
- The card does bf16 369 TFLOPS, fp8 719, plain fp32 only 65; memory 1,467 GB/s.
- 27B teacher: HF ~1,010 tok/s; vLLM + vllm-lens ~1,400. With every output forced to the same length vLLM does
  ~1,800, so most of the gap is mixed output lengths, not the lens. fp8 weights would give ~3,000 but Adam wants
  the bf16 teacher. CUDA graphs and dropping log-probs gave nothing on the 27B.
- Our vllm-lens patch (`patches/vllm_lens_fast_steering_hook_and_batched_rpc_20261011/`): identical outputs
  (1,020/1,020); on a 2B model it cuts the lens's cost from 7.1% to 0.3%; on the 27B no change. It also found a bug:
  a steering vector passed as a tensor view pickles its whole base tensor (32.8 MB each) and can run the host out of RAM.
- Student sampling through vLLM prompt-embeds is 2.6x faster than our HF loop (19,234 vs 7,519 tok/s), greedy
  outputs match. Not yet wired into the trainer.

**What the 27B teacher writes** (`05_teacher_27b_outline_and_intent_probe_2026-10-11.md`):
- On web text it always uses the same three-part outline (38/40 read by hand): genre and topic, what the passage is
  doing, the last token and what comes next. ~123 words. It names specifics in 36/40, but often makes them up.
  It never speaks of intent on web text (0/40).
- On 62 chat prompts it reads stance well and the same way across samples: jailbreaks get "refusal expected" and
  the trick named (1.00), deception plans 0.83. It misses who did what (role bindings 0/6), which agentic dilemma
  (0.12), and makes up specifics (0.56). One rater only.

**Two-document blends (S3 gate):** the teacher's notes on blended vectors are fluent and parse, but nearly all invent
one hybrid document that links both topics ("sewing designers' race recap"). FVE 0.18 vs 0.74 on natural vectors.
The gate passed, so S3 runs as a test only.

**Template NLA** (`template_nla_from_27b_distillation_data_what_to_reuse_2026-10-11.md`, plan only): the data from this
run (vectors, texts, two notes per vector, the AR) is enough to train a fixed-template reader for web text, but not for
intent; intent needs chat inputs run through the teacher.

**WorkspaceBench** (`workspacebench_full_cost_and_judge_strictness_notes_2026-10-10.md`): a full run costs ~$340-500 per
arm (jailbreak alone ~$257), so the plan uses a cheap local preview first.

## 5. Problems met, and fixes (for the next agent)

1. **Nebius spot market.** B300 spot sat above our $2.75 cap for ~10 h (45 stops). Lesson: Nebius spot is not usable
   for a multi-hour job at that cap; on-demand Vast RTX PRO 6000 at $1.44/h ran 16 h without a stop.
2. **State lost in the move.** `preflight.json` did not come across from Nebius; the S3 gate would have crashed and
   silently skipped S3. Rebuilt from the log. Check every gate's inputs after a move.
3. **Out of GPU memory at the first student run (19:55 UTC 11 Oct).** The 27B AR (54 GB) stayed on the GPU while the
   2B student trained (full fine-tune peaks at ~84 GB on its own). Stage A (adapter only) fitted at 87.6 GB, so the
   pilot looked fine for 4 minutes, then died at stage B. Fix: `--ar-offload` on every run (AR on CPU, moved in for
   each eval, ~1 min per quick eval) and free gradients after each step. Example: pilot log
   `nld27_sftkd_qwen35-2b_..._pairs10k-pilot_20261010.log`, "Tried to allocate 1.64 GiB ... 41.62 MiB is free".
4. **Text-only control also ran out of memory** at micro-batch 32: the prompt carries 192 text tokens. Fix: micro 16
   (same batch of 32).
5. **Box failsafe could not stop the box.** `vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` failed:
   the variable was empty inside the container ("argument --api-key: expected one argument"). Harmless this time
   (it kept the box up for the fix), but it means the box-side failsafe is not a real failsafe on Vast. Stop boxes
   from the Mac, or pass a scoped key.
6. **Mac rebooted ~20:51 UTC 11 Oct**, killing the 5-minute watcher (last line 18:21). So nobody saw the 19:55 crash;
   the GPU idled ~1 h (~$1.50).
7. **Partial teacher download + offline mode.** Box 2 fetched only the AR and tokenizer files (not the 54 GB base);
   `snapshot_download` in offline mode then refuses the incomplete snapshot. Fix: `NLA_TEACHER_DIR` env var in
   `teacher_dir()` (common script).

## 6. Not done, and what to do next

In the plan's order (main path first), with rough costs on a Vast RTX PRO 6000 at ~$1.4/h:
1. **On-policy round** from the main student (100k vectors: student samples, teacher scores, one more SFT pass):
   ~3-4 h, ~$5-6. Script and queue step exist (`nladistil27b_onpolicy_20261010.py`; queue.sh after `stop_after_main`).
   Note: queue.sh expects the main run in `runs/` on the box; restore it from the HF model repo (final_lm + adapter).
2. **GRPO with the AR as reward (R1)**: plan allows up to 10 h; on the 7B it plateaued in ~6 h. ~$8-14. Watch α 0
   false alarms and injection, not just reward.
3. **WorkspaceBench preview (P)** with the local judge: ~$3-6.
4. Arms not run: 4B student (S2), two notes per vector (S4; skipped by the 02:45 cut-off), LR 5e-6 and 1.0 / 2.0-pass
   probes; a second seed or 131k check of blends. The 4B needs
   `--grad-ckpt --micro 8` plus `--ar-offload` (already in queue.sh).
5. Wire the vLLM prompt-embeds sampler (`nladistil27b_fast_student_gen_20261011.py`) into the evals and on-policy
   sampling: evals and sampling would run 2.6x faster.
6. Fix the box failsafe on Vast (empty CONTAINER_API_KEY), or always stop boxes from the Mac.
Total for 1-3: ~$16-26, ~1 day, well inside the $150 cap (~$50 spent).

## 7. Spend

| Item | $ |
|---|---|
| Nebius B300 spot (10-11 Oct; extraction, pre-flight, 10k labels) | ~10 |
| Vast box 1, RTX PRO 6000 (11 Oct 04:45 - 12 Oct 02:32 UTC, 21.8 h at $1.44) | ~31.5 |
| Vast box 2, RTX PRO 6000 (11 Oct 21:10 - 12 Oct 03:32 UTC, 6.4 h at $1.39) | ~9 |
| API | 0 (judging by subagents) |
| **Weekend total** | **~50 of the $150 cap** |

This session (11 Oct 20:55 - 12 Oct 03:32 UTC, Adam's limit $20 / 8 h): ~$17 for my two boxes (box 1 ~$8, box 2 ~$9);
Vast credit fell $58.11 -> $40.14 ($17.97, which also covers part of another agent's RTX 6000 Ada box `wk-AF-1`, run
11 Oct ~15:49 to before 02:30). Both my boxes destroyed (02:32, 03:32 UTC); no Vast instances left.
