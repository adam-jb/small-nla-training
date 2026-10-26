# Full plan: lean distillation of the 27B WorkspaceBench NLA into Qwen3.5-2B (10 Oct 2026)

Status: plan, not run. Adam chose the lean route on 10 Oct: teacher pairs -> student SFT -> GRPO (no on-policy rounds).
Background, cost reasoning and the teacher card: `04_distillation_cost_and_scaling_plan_2026-10-10.md`.
WorkspaceBench cost and judge notes: `workspacebench_full_cost_and_judge_strictness_notes_2026-10-10.md`.
Everything copies the 7B run (`91_background_7b_distillation_findings_2026-10-03.md`, scripts `scripts/nladistil_*`)
unless stated, so the two runs can be compared row by row.

## 1. How likely is it to work?

What we know: the same recipe took a 1.5B to a tie with the 7B NLA on FVE (0.735 vs 0.731) and truth, with injection at
α 0.5 of 84% vs 93%. Distilling much larger teachers into 1.5B students works in general (DeepSeek-R1 into 1.5B-7B Qwens).
But the distillation literature also finds a "capacity gap": past some size ratio a stronger teacher gives a student no
better, or worse, results (Cho & Hariharan 2019; Mirzadeh et al. 2020, "teacher assistant"; later work on distillation
scaling). Our ratio grows from ~5x (7.6B -> 1.5B) to ~13x (27B -> 2B).

Why the 27B is harder:
- The vector is from a bigger model that knows more. To name what it holds (a bridge entity, a rhyme it plans), the
  student must know the word. A 2B knows fewer rare names than a 27B.
- The teacher's explanations are richer (Sonnet-style, 80-100 words, specific quotes), so there is more to copy.
- WorkspaceBench rewards naming exact targets. That is where a small student loses most.

Why it may still work: the task is narrow, the student only maps one vector to a short text, and the text-describing part
(most of FVE) is easy for small models (a text-only 1.5B got 0.61 of 0.73 on the 7B).

My guesses (judgement, not measured):

| Outcome after GRPO | Chance |
|---|---|
| Held-out FVE ≥ 95% of the teacher's | ~75% |
| FVE tie with the teacher (inside the CI) | ~50% |
| Injection at α 0.5 ≥ 80% of the teacher's hits | ~60% |
| Blind truth within 5 points of the teacher | ~70% |
| WorkspaceBench macro pass rate ≥ 80% of the teacher's | ~50% |
| WorkspaceBench macro ≥ 95% of the teacher's | ~20-25% |

So: very likely a useful student, with a coin-flip chance of matching the teacher on WorkspaceBench. Expect the loss to
sit in the knowledge-heavy families (multihop, multilingual, association) and least in typo and text-describing ones.

## 2. Will 263k pairs be enough?

Probably not quite. On the 7B, 263k pairs gave 95.5% of the teacher after SFT and GRPO needed only ~150k rollouts to close
the rest. For the 27B I expect 88-93% after SFT at 263k and 300k-1M rollouts of GRPO, possibly levelling off below the
teacher. So the plan labels 263k, reads the ladder, and only then decides on a second 263k (section 5, step 4).

## 3. Inputs and data sizes

Same sources, positions and held-out docs as the 7B run, so every comparison is paired on the same text.

| Set | Source | Count | Notes |
|---|---|---|---|
| Train vectors | Ultra-FineWeb parts 1500-1503 (raw text) + WildChat shards 10-13 (Qwen3.6 chat template, thinking off) | 233k from ~78k docs | 3 positions per doc, ≥ 50 tokens in, docs cut at 1,024 tokens; positions mapped from the 7B run by character offset |
| Two-context mixes | w = v_a + α·|v_a|·unit(v_b − mean), α ∈ {0.25, 0.5, 1.0} | 30k | made from train vectors, no forward pass |
| Held-out | same 340 docs as the 7B run | 1,020 | never trained on |
| Concept injection | 30 concepts (`data/nladistil_concepts_20261003.json`), direction = mean L42 state over concept sentences − train mean | 10 held-out base vectors × 30 × 4 α = 1,200 cases | α 0 is the false-alarm row |
| Notes | synthetic notes with made-up name / city / job | as 7B | vector at the final token |

Sizes (one L42 vector = 5,120 × 2 bytes bf16 = 10 KB; bf16, not fp16, in case some dims exceed the fp16 range):

| Item | Size at 263k | At 525k |
|---|---|---|
| L42 vectors | 2.7 GB | 5.4 GB |
| + L28 and L52 (for later studies) | +5.4 GB | +10.8 GB |
| Teacher explanations (~170 tokens each, ~45M tokens), gzipped | ~0.3 GB | ~0.6 GB |
| Teacher top-20 per token (int32 id + fp16 log-prob) | ~5.4 GB | ~10.8 GB |
| Student checkpoints (2B bf16, keep 3-4) | 12-16 GB | same |
| Extraction prefill | ~47M tokens through 52 of 64 layers | ~94M |

Everything except checkpoints goes to a public HF dataset as 10k-row shards while the run goes (about 14 GB at 263k).
GRPO keeps per-step statistics and 1% of rollouts, not all of them.

## 4. Models, prompt and loss

- **Teacher:** `ceselder/qwen3.6-27b-nla-rl`, `av_base` + `av_rl_adapters/iter_000400` merged once on the box; T 1.0,
  max 256 new tokens, the README's prompt, norm-matched add at block 1. One sample per vector, top-20 log-probs kept.
- **Scorer:** `ar_reconstructor/` (saved at RL step 600). The same AR scores teacher and student.
- **Student:** Qwen/Qwen3.5-2B, same AV prompt, the marker replaced by 4 soft tokens from an adapter:
  LayerNorm(5120) -> Linear(5120 -> 4 × 2048). (Optional size rung: Qwen3.5-0.8B.)
- **Text-only control:** Qwen3.5-2B, same prompt, the last 192 tokens of the source text in place of the vector.

**SFT loss against the teacher**, per token t of the teacher's sampled explanation y, given the vector v:

    L_t = (1 − w) · [ −log p_s(y_t | y_<t, v) ]                         # copy the teacher's sampled token
        +    w    · [ −Σ_{k ∈ top20_t} q_t(k) · log p_s(k | y_<t, v) ]   # match the teacher's top-20
    q_t = softmax of the teacher's top-20 log-probs at t (renormalised, T 1);  w = 0.5;  L = mean over all tokens.

The second term is the forward KL to the teacher's top-20 distribution, up to a constant. Same as the 7B run
(`scripts/nladistil_train_student_20261003.py`, `batch_loss`).

Schedule: stage A, adapter only, 10k examples, lr 1e-3. Stage B, full model, lr 1e-5 (adapter 2e-4), batch 32,
1.5 passes, cosine with warm-up.

**GRPO** (as `scripts/nladistil_rl_grpo_20261006.py`): 64 vectors × 8 samples per step at T 1; reward
r = −MSE(AR(explanation), v) in the space the AR's FVE uses (check in pre-flight: the 27B AR uses no normalisation;
the 7B used unit vectors, reward −2(1 − cos)); advantage = (r − group mean) / group std; PPO clip 0.2, 2 minibatches;
k2 KL 0.01 to the start policy; LoRA r32 on every linear + the full adapter; prompts from train + mix vectors (fresh
ones, not just one pass of the same 32k as on the 7B).

## 5. Steps, stop rules, cost (one A100 80 GB, $1.99/h)

| # | Step | Box h | $ | Stop / go rule |
|---|---|---|---|---|
| 0 | Pre-flight: 80 GB check, download, merge LoRA, 50 held-out vectors through teacher + AR, speed | 3 | 6 | Stop if card < 80 GB, teacher FVE < 0.65, or < 400 tok/s (then switch to EasyNLA's vLLM path) |
| 1 | Extract L28/L42/L52 for 233k + held-out + inject + notes; make 30k mixes | 4 | 8 | Spot-check: our L42 vectors through the teacher match `data/example_activations.parquet` behaviour |
| 2 | Teacher labels 263k + held-out, inject, notes; shards to HF | 16 | 32 | Every 50k: length, parse rate, tok/s |
| 3 | Ladder: fresh 2B students at 66k, 131k, 263k; text-only 2B at 131k; (0.8B at 131k) | 9.5 | 19 | 66k student must beat shuffled clearly, else find the bug |
| 3b | Fairness check: Qwen3.5-2B vs Qwen2.5-1.5B, CE only, on the 7B teacher's 131k labels | 3 | 6 | Measures the student-family part of the 7B-vs-27B gap |
| 4 | Decide: second 263k batch? | - | +40-50 if yes | Yes if share of teacher rises > 0.03 from 131k to 263k |
| 5 | GRPO from the best SFT student | 11-18 | 22-36 | Stop at plateau (< 0.003 FVE over 100 steps) or hacking: α 0 false alarms +5 pts, length drift, injection falls |
| 6 | Evals (section 7) | 5 | 10 | |
| 7 | WorkspaceBench, macro families, teacher + final student (GPU part) | 4 | 8 | judge API ~$35-75 (macro families, both arms), see notes file |
| 8 | Destroy box, log, write findings | | | |
| | **Total** | **~56-63** | **~$110-125 GPU + ~$35-75 judge API** | + $40-50 if step 4 says yes |

Wall clock: ~3 days on one box. Box-side failsafe (auto-shutdown when the queue ends; Mac sleep must not strand it),
checks every 5 min, persistent disk in the box's region for resumes.

## 6. Logging, so input scaling can be read later

One file holds every result from both teachers: `runs/nladistil_scaling_ledger.jsonl`, one row per (teacher, student,
stage, checkpoint). The 7B rows are back-filled from the existing runs. W&B project `nla-distil` uses the same keys,
grouped by teacher.

Fields per row:
- **Run:** name (naming convention below), date, git commit, seed, box type, GPU-hours, $.
- **Teacher:** repo, checkpoint, params (total / non-embedding), subject model, layer, layer depth fraction, d_model,
  label writer, recipe, teacher FVE on our held-out set.
- **Student:** repo, params (total / non-embedding), vocab, adapter slots.
- **Data:** stage (sft / grpo / text-only), unique pairs, unique docs, mix share, examples seen, passes, tokens seen,
  rollouts seen, lr schedule id.
- **Measures:** FVE + 95% CI, shuffled FVE, text-only floor, share of teacher, vector share, injection hits α 0 / 0.25 /
  0.5 / 1.0 and their share of the teacher's, notes city / job / name, truth / topic / final-token (when judged),
  WorkspaceBench per family + macro (when run), mean length, parse rate, KL (GRPO).

Two kinds of points, kept apart with a field: separate fresh runs (the real ladder) and checkpoints inside one run (a
rough curve; their learning rates differ). The 6 Oct lesson: only fresh runs give a fair ladder.

Analysis: per teacher, fit share(n) = s∞ − b·n^(−c) on the fresh rungs; report n95 (pairs to 95% of the teacher), s∞,
the GRPO lift and rollouts to plateau. Extend `scripts/nladistil_build_scaling_page_20261006.py` to plot both teachers
on one chart (x: log pairs; y: share, vector share, injection share).

Naming (model type, data scale, size, date, step):
`nladistil_sftkd_qwen35-2b_vec_from-nla-qwen36-27b-L42-rl400_pairs263k_YYYYMMDD`,
`nladistil_textonly_qwen35-2b_..._pairs131k_YYYYMMDD`, `nladistil_grpo_qwen35-2b_..._ar-reward_YYYYMMDD/step0600`.

## 7. Evals

| # | Eval | What it shows | Size | Cost |
|---|---|---|---|---|
| E1 | Held-out FVE via the 27B AR, CI by doc bootstrap, paired vs teacher; shuffled control | How well the explanation lets the AR rebuild the vector | 1,020 vectors | GPU only |
| E2 | Text-only control | The floor from describing the text alone; gives the vector share | 1,020 | GPU only |
| E3 | Concept injection | Reads the vector itself, not the text (AR plays no part) | 1,200 cases | GPU only |
| E4 | Notes (name / city / job) | Reads specific facts | as 7B | GPU only |
| E5 | Blind truth: claims supported / contradicted / unverifiable; topic right; final token right | Are the claims true; teacher re-judged in the same pass | 80 vectors | $0 (Sonnet subagents) |
| E6 | WorkspaceBench at L42, macro families, teacher + final student (+ 263k SFT student if cheap) | The benchmark the teacher is known for | ~2k items | ~$35-75 API |
| E7 | GRPO health every 50 steps: reward, KL, length, parse, α 0 false alarms, injection | Catches reward gaming | - | GPU only |
| E8 | Later: deception twins re-answered by the 27B | Ready for the deception phase | 100-5,320 | separate approval |

## 8. Revision (10 Oct, Adam): the target is teacher parity on WorkspaceBench, and the run must map the scaling laws

Adam: "want it to be as good as the teacher"; and the training should test scaling as the 7B distillation did, because
that is how the best recipe and data mix were found. This section changes sections 5-7 where they disagree.

### 8.1 What "as good as the teacher" means

- WorkspaceBench macro pass rate (official judge, L42, k 1 sample, the repo's sampling): the student-minus-teacher gap,
  paired by item with a bootstrap CI, must include 0. No family more than ~10 points below the teacher.
- The teacher is run **twice** (different seeds). Its own run-to-run spread is the noise floor: "a tie" means the
  student's gap is no bigger than that spread.
- Also a tie on held-out FVE, injection at α 0.5 and blind truth.

### 8.2 Why the lean plan alone probably falls short, and the fixes

| Cause of a gap | Fix | Extra cost |
|---|---|---|
| Training data does not look like the bench: web/chat positions ≥ 50 tokens in, while the bench reads the last token of short prompts (facts, multihop, typos, rhymes, sums, chat renders) | **Bench-like training set:** ~10k new prompts in the style of each family, written by Sonnet subagents ($0), gated on Qwen3.6-27B on the box, de-duplicated against every bank item and entity, read at the bench's position rules (final token, last five). ~50k vectors. Fair: the teacher writes every label, so the student learns only what the teacher says | ~$10 GPU |
| One T 1 teacher sample is a noisy target where the bench cares most | 2 teacher samples per bench-like vector (100k labels) | ~$10-20 |
| The AR reward pays for describing the text, not for naming what the vector holds | **Mixed RL reward:** AR reward + β × teacher log-prob of the student's tokens (on-policy distillation: a dense, per-token "what would the teacher say" signal), β tuned on the dev set | ~+20% per GRPO step |
| On-policy rounds were dropped in the lean plan | Bring back **one** on-policy round, on all bench-like vectors + 50k general (on the 7B it added +5-7 points of injection, the measure closest to naming) | ~$12 |
| A 2B may not know the names a 27B holds | Student-size rung (0.8B / 2B / 4B at 131k) shows whether the gap is capacity. If it is, 4B is the fallback, same tokenizer | 4B rung ~$8; full 4B route +$40 |
| No way to tune without overfitting the bench | **Dev set** (below); the real bench is used only on frozen checkpoints | ~$25-50 API |
| Sampling: k 1 at T 1 is noisy | Pick each reader's own best temperature on the dev set (the 7B student was best at T 0.7); teacher gets the same chance. Keep k = 1 for both | $0 |

**Dev set ("wsb-dev"):** ~1,500 new items across the macro families, written like the bench (Sonnet subagents), gated
on Qwen3.6-27B (right ≥ 8 of 10 samples, as the authors did), disjoint from both the real bench and the bench-like
training prompts. Judged by the repo's own strict prompts with Gemini 3.8 Flash: ~$3-6 per reader. Every grid run below
gets a wsb-dev score, so the scaling ledger has a bench-shaped measure at every point.

Revised chance of a WorkspaceBench tie: ~25% for the lean plan; ~45-55% with these fixes at 2B; ~65% if the 4B fallback
is allowed. Still a judgement, not a measurement.

### 8.3 Scaling grid (fresh SFT runs, each scored on FVE, injection, wsb-dev)

All 2B unless stated. Labels made once; every run uses subsets of them. Logged to `runs/nladistil_scaling_ledger.jsonl`
with a `grid_axis` field.

| Axis | Runs | Question it answers | Box h | $ |
|---|---|---|---|---|
| A. Data size, general mix | 33k, 66k, 131k, 263k (+525k if labelled) | Ladder shape; n95 and s∞ vs the 7B | 7.5 | 15 |
| B. Student size at 131k | 0.8B, 2B, 4B | Is the gap capacity? Student-size law | 5 | 10 |
| C. Bench-like share at 131k total | 0%, 15%, 30% | Best data mix for the bench at fixed labels | 4 | 8 |
| D. Samples per vector at 131k labels | 1 × 131k vectors vs 2 × 65k | Many vectors or more samples each? | 2 | 4 |
| E. Two-context mixes at 131k | 0% vs 11% | Do the 7B-era mixes help? (untested on the 7B) | 2 | 4 |
| F. Text-only floor at 131k | 1 | Vector share | 2 | 4 |
| G. Teacher axis | 7B rows back-filled + fairness check (2 runs on 7B labels) | Teacher-size comparison, student-family confound | 3 | 6 |
| H. RL and on-policy | evals every 50 GRPO steps and after the on-policy round | Lift per rollout; plateau | (in main path) | 0 |
| | **~14 runs** | | **~26** | **~$51** + wsb-dev API ~$40 |

### 8.4 Revised steps and cost

| # | Step | $ GPU | $ API |
|---|---|---|---|
| 0 | Pre-flight (incl. getting EasyNLA's vLLM path working: the teacher now writes ~360k labels) | 8 | |
| 1 | Write bench-like prompts + wsb-dev (subagents), gate both on the 27B | 4 | 0 |
| 2 | Extract L28/L42/L52: 233k general + ~50k bench-like + held-out + inject + notes | 10 | |
| 3 | Teacher labels: 263k general + 2 × 50k bench-like + held-out sets (vLLM ~$20; HF ~$45) | 20-45 | |
| 4 | Scaling grid (8.3) | 51 | 40 |
| 5 | Main student: 2B on all labels (best mix from axis C) | 10 | |
| 6 | One on-policy round | 12 | |
| 7 | GRPO, mixed reward, eval every 50 steps | 25-40 | |
| 8 | Real WorkspaceBench macro: teacher × 2 seeds, student at 2 frozen checkpoints | 6 | 60-120 |
| 9 | Optional: second 263k general batch; or the 4B route | 40-50 | |
| | **Total** | **~$145-185** (+$40-50 optional) | **~$100-160** |

Wall clock ~4-5 days on one A100. Budget cap to agree with Adam before launch.

## 9. Current version (10 Oct, later): guard against benchmark-maxxing; arms and schedules. Replaces 8.2-8.4 where they differ

Adam is wary of training to the benchmark. Rules:
1. The main student sees only general data (web + chat) and generic objectives (copy the teacher; rebuild the vector).
2. Anything bench-shaped is a side arm, reported as such.
3. The final student is picked by a rule fixed before any WorkspaceBench run, using general measures only (held-out
   FVE, injection, truth, writing defects). WorkspaceBench is read once at the end, never tuned on. The wsb-dev set is dropped.
4. Benchmark-maxxing signature to watch: bench score up while general measures go down.
Cost of honesty: chance of a WorkspaceBench tie at 2B falls to ~35-45% (from ~45-55% with bench-shaped data).

### Arms

| Stage | Arm | What changes | Data | Cost |
|---|---|---|---|---|
| 0 | Pre-flight | 80 GB card; teacher via EasyNLA's vLLM path; teacher FVE on 50 held-out; speed | - | ~$8 |
| 1 | Data | L28/L42/L52 vectors; teacher labels (1 sample; 2 for a 65k subset) | 233k general + 30k mixes + 65k second samples (+50k bench-like for S7 only) | ~$30-55 |
| 2 | S1 ladder | pairs 33k / 66k / 131k / 263k, 2B | general | ~$15 |
| | S2 student size | 0.8B / 4B at 131k | general | ~$10 |
| | S3 mixes | 0% vs 11% two-context mixes at 131k | general | ~$4 |
| | S4 samples per vector | 1 × 131k vectors vs 2 × 65k | general | ~$4 |
| | S5 text-only | reads the text, not the vector, 131k | general | ~$4 |
| | S6 teacher axis | Qwen3.5-2B vs Qwen2.5-1.5B on the 7B's 131k labels, CE only | 7B labels | ~$6 |
| | S7 side arm (labelled) | 15% bench-like at 131k | general + bench-like | ~$4 |
| 3 | Main SFT student | = S1's 263k rung (no extra run) | general | 0 |
| 4 | R1 GRPO-AR | the teacher's own reward (as on the 7B) | general + mixes | ~$20-35 |
| | R2 GRPO-AR + teacher | reward + β × teacher log-prob of the student's tokens (on-policy distillation; replaces the separate on-policy round) | same | ~$25-40 |
| | R3 GRPO-Flow (optional) | denoising reward (Flow-NLA); see notes file | same | ~$30-45 + build time |
| 5 | Pick | fixed rule on general measures | - | 0 |
| 6 | WorkspaceBench | macro families: teacher × 2 seeds + final R1, R2 (R3) students | - | ~$100-200 API, ~$12 GPU |

GPU ~$190-260 (~$150-190 without R3 and S7); API ~$100-200; ~5 days on one A100.
Pre-registered pick: highest held-out FVE among arms whose injection at α 0.5 and truth are no worse than the SFT start
and whose writing defects are no higher than the teacher's; ties go to the simpler arm (R1).

## 10. Correction and options (10 Oct, later)

- **R2 correction:** a mixed reward (AR + teacher log-prob in the same GRPO step) needs the 27B teacher (54 GB) and the AR
  (35 GB) in memory together: 89 GB, more than one A100 80 GB. Practical R2 = one offline on-policy round (teacher top-20
  on the student's own samples, as on the 7B) and then GRPO-AR. Cost ~$32-47 as its own branch.
- **New data arm S8 (teacher's home data):** 66k vectors from the teacher's own training texts (FineFineWeb, public in
  `ceselder/qwen3-8b-nla-L24-finefineweb-100k`) vs 66k of our default web + chat, same student. Tests whether data
  matched to the teacher's training beats our 7B-matched mix. ~$10-14 (new labels + one run).
- **How general WorkspaceBench is, and options to train with it in mind:** see the 10 Oct chat summary, mirrored here.
  The bench tests one main skill: naming one hidden concept or value that Qwen3.6-27B computes at the end of a short,
  mostly synthetic prompt (thinking off), ~100 items per family. It does not test long documents, mid-document reading,
  graded or partial information, or intent beyond 32 agentic and 86 jailbreak items. Options: (A) general data only;
  (B) bench-style prompts, freshly written, teacher-labelled, reported as a side arm, ideally with half the families
  held out; (C) training on the family tasks with their true targets: no longer a distillation of the teacher, and only
  defensible with a family hold-out.

## 11. Default data changed: train on the teacher's own distribution (10 Oct, Adam's point)

The 27B teacher was trained on web text only. Checked in `ceselder/qwen3-8b-nla-L24-finefineweb-100k` (meta + 600
sampled rows):
- Source: 100k docs from FineFineWeb (m-a-p/FineFineWeb: FineWeb sorted into ~67 topic domains; Apache-2.0). Raw text,
  no chat template; 0 of 600 sampled rows look like chat.
- Positions: 10 per doc; the vector is at the last token of a prefix. Prefix length (Qwen3-8B tokens) p10 68, median
  180, p90 685, min 51, max ~3,300: mostly early in the doc.
- Labels (warm-start only): Claude Sonnet 4.6, ~80-100 words, 2-3 features "the model must be thinking about", the last
  about the final token. AV SFT 247,261 rows; AR SFT 247,358; RL 499,846 unlabelled prefixes (last 50k held out for eval;
  ~200k used by step 400). Vectors regenerated from Qwen3-8B L24 to Qwen3.6-27B L42 on the same prefixes.

The 7B run already followed this rule: its teacher was trained on 50% Ultra-FineWeb + 50% WildChat, and so was the
student. Carrying the 7B mix over to the 27B was a mistake. New default:

| Set | Content | Size |
|---|---|---|
| Train | Exact public prefixes from the teacher's dataset (AV/AR/RL splits, text already cut), random 233k | 233k + 30k two-context mixes |
| Held-out (main) | Fresh FineFineWeb docs, de-duplicated against all 100k teacher docs, same position rule | 1,020 vectors |
| Held-out (cross-teacher) | The 7B run's 1,020 Ultra-FineWeb + WildChat vectors | 1,020; scored for both students |
| Arm S9 | Home-only vs home + 25% WildChat at 131k: does chat (needed later for deception, and for the bench's chat-rendered families) cost anything on the home set? | ~$6-8 |

S8 (teacher's texts vs the 7B mix) becomes the reverse check: 66k of the 7B mix vs 66k home, ~$10-14, optional.
Extraction gets cheaper (median prefix ~180 tokens vs up to 1,024): ~$4 instead of ~$8.

## 12. Approval sheet (10 Oct, latest; this is the current plan)

Teacher `ceselder/qwen3.6-27b-nla-rl` iter_000400, L42 -> student Qwen3.5-2B. Data = the teacher's own FineFineWeb
prefixes (section 11). Option A (no bench-shaped data in the main student; section 9 rules). One A100 80 GB at
$1.99/h; teacher via vLLM (HF fallback in brackets).

| ID | Stage | $ |
|---|---|---|
| 0 | Pre-flight (stop if card < 80 GB, teacher FVE < 0.65, speed too low) | 8 |
| 1 | Data: vectors L28/42/52 + teacher labels 263k + held-out, injection, notes; shards to public HF | 19 (36) |
| S1 | Ladder 33k / 66k / 131k / 263k (263k = main student) | 15 |
| S5 | Text-only floor | 4 |
| S2 | Student size 0.8B, 4B | 10 |
| S3 | Two-document blends off | 4 |
| S4 | Two notes per vector | 8-12 |
| S9 | +25% chat | 6-8 |
| S6 | Teacher-size fairness check | 6-8 |
| S8 | 7B mix vs home data | 10-14 |
| S7 | Bench-style 15% (option B only) | 10-19 |
| X | Second 263k batch, only if the ladder says so (ask again) | 40-50 |
| R1 | GRPO-AR | 20-35 |
| R2 | On-policy round, then GRPO-AR | 32-47 |
| R3 | GRPO-Flow (+ ~1 day build) | 30-45 |
| E | General evals; judges via subagents $0 | 10 |
| W | WorkspaceBench macro: teacher x2 | 50-110 |
| W+ | each student on WorkspaceBench | 17-38 |

Bundles: minimal (0, 1, S1, S5, R1, E, W, one W+) ~$85 GPU + ~$65-140 API, ~2.5 days;
recommended (+ S2, S3, S4, S9, R2, second W+) ~$155 GPU + ~$80-175 API, ~4 days;
everything (+ S6, S8, R3, third W+) ~$225 GPU + ~$95-210 API, ~5-6 days.

## 13. Adam's choices (10 Oct) and proposed final selection (awaiting his OK)

Adam: no chat (copy the teacher's training data), no WorkspaceBench for now (cheaper internal evals first), wants R3,
S2; maybe S3, S4; no S9, no R2 unless I am sure on-policy matters.
Changes that follow:
- Two-document blends are not in the teacher's data, so the main student goes without them; S3 becomes "+11% blends"
  as a test. Caveat: blends look like the injection eval, so they may lift that score without better reading.
- On-policy is probably not skippable: on the 7B it took injection at α 0.5 from 60% to 76% (most of the student's
  gain), while GRPO added +7 more. GRPO straight from SFT was never tested. Proposal: one on-policy round (~$12) in the
  main path before R1 and R3, not a separate R2 branch.
- P (new): WorkspaceBench proxy, read only, never trained on: teacher vs student readouts on the real bench items,
  judged by Qwen3.6-27B locally with the repo's own strict prompts (~$3-6 GPU, $0 API). Predicts parity before the
  official run.
Proposed: 0, 1, S1, S5, S2, S3, S4, OP, R1, R3, E, P: ~$165 GPU (~$185 with the slow teacher path), $0 API, ~4.5-5 days.
Update (10 Oct, later): Adam drops R3 (Flow-NLA can branch later from the pre-R1 or post-R1 checkpoint; keep both),
keeps OP and S4. S3 stays a test only, gated: in pre-flight the teacher writes notes on 20 blends; if they are
incoherent, S3 is skipped. S3 counts as a win only if it lifts natural held-out FVE / truth, not injection alone.
Selection: 0, 1, S1, S5, S2, S3 (gated), S4, OP, R1, E, P: ~$127 GPU (~$145 slow path), $0 API, ~3.5-4 days.

## 14. Budget $150, cheap tests first, branch handling, storage, GPU blocker (10 Oct, latest)

**GPU blocker (checked on the Lambda API, 10 Oct):** `gpu_1x_a100_sxm4` ($1.99) is a 40 GB card; no single 80 GB A100 is
offered. The 27B teacher (54 GB bf16) does not fit. 80 GB+ types (GH200 96 GB $2.29, H100 80 GB $3.29-4.29, B200 $6.99)
were all unavailable at check time and are not A100s (CLAUDE.md says one A100). Options: (a) A100 40 GB with the teacher's
weights in 8-bit (FP8 weight-only in vLLM; AR in bf16 fits with small batches) plus a fidelity check; (b) poll for a
GH200 (bf16, faster, ARM CPU, needs Adam's OK); (c) H100 80 GB (needs OK, dearer). Recommended: (a), falling back to (b)
if the check fails. The 4B student needs memory-saving settings on 40 GB (bf16 weights, 8-bit Adam, checkpointing).

**Cheap and free tests first (≈ $0-12 total):**
| Test | Where | $ | Decides |
|---|---|---|---|
| Write and smoke-test every script with tiny stand-in models (as `NLADISTIL_SMOKE` on the 7B) | Mac | 0 | Catches bugs before GPU hours |
| Read the teacher's public data: prefix lengths and label lengths in the Qwen3.6 tokenizer; held-out dedup | Mac | 0 | Exact token budget for labels; clean held-out |
| Fit the 7B ledger curves; set rung sizes and stop rules | Mac | 0 | Wasted rungs avoided |
| FP8 vs bf16 teacher on 50 vectors (bf16 via CPU offload, slow but small): token agreement + AR FVE | A100 40 GB | ~2 | Is the 8-bit teacher the same teacher? |
| 50 teacher notes on blends, read by me | same box | ~0.5 | Run S3 or not |
| 10k-pair pilot of the 2B student | same box | ~1-2 | Loss falls, FVE rises, before the big label run |
| Order the grid so S3 and S4 (at 131k) finish before the 263k main run | - | 0 | Main student uses the winning recipe; no reruns |
| P preview with a local judge | box | 3-6 | WorkspaceBench parity before paying for it |

**Branches:** every grid arm is a separate fresh run; the main path is 263k SFT -> OP -> R1, each stage saving a
checkpoint. A branch that does not help is logged in the ledger (with its numbers) and dropped; the next stage starts
from the last good checkpoint. S3 joins the main recipe only if it lifts natural held-out FVE or truth.

**Storage:**
| What | Where |
|---|---|
| Teacher weights (~90 GB) | Box local disk only, downloaded from HF each boot; never stored |
| Vectors L28/L42/L52, teacher notes, top-20, metadata, held-out sets (~14 GB) | Public HF dataset (name to agree; needs an HF write token, none on the Mac yet) + persistent disk while running |
| Main-path weights: SFT-263k (pre-OP), post-OP = pre-R1, post-R1 (2B bf16, ~4.5 GB each) | Lambda persistent disk `claude-lens-results-20260929` (us-east-1) under `nladistil27b_<date>/models/`, plus an HF model repo (private unless Adam says public) |
| Grid runs (ladder, S2-S5) | Metrics and generations only, no weights (as on the 7B); the 4B weights kept if it wins |
| Logs, metrics, generations, judge files | Mac `runs/nladistil27b_*`; ledger `runs/nladistil_scaling_ledger.jsonl`; W&B project `nla-distil` |
| Scripts | `scripts/nladistil27b_*_<date>.py` (git) |
| Docs | this file, `experiment_log.txt`, findings file at the end, `docs.md` |

Budget: Adam's cap $150. Selection (0, 1, S1, S5, S2, S3 gated, S4, OP, R1, E, P) ~$127 (fast path) to ~$145 (slow);
cheap tests ~$5-10 inside that. If spend reaches ~$130 before R1 ends, stop and ask.

## 15. Close monitoring and learning log (10 Oct, Adam's instruction; applies to every stage)

Aim: learn as we go (how pairs, passes, learning rate and RL steps move the student, against the 7B), and never let a
broken or stalled stage burn days or $100s.

**Reference curve from the 7B (Qwen2.5-1.5B student, lr 1e-5 cosine, batch 32, 1.5 passes; held-out FVE by examples seen;
teacher 0.731):** every rung tracks the same early curve, then levels off by its unique data.

| seen | 30k | 60k | 120k | 240k | end |
|---|---|---|---|---|---|
| 66k rung | 0.305 | 0.478 | - | - | 0.540 (98k) |
| 131k rung | 0.284 | 0.494 | 0.628 | - | 0.655 (197k) |
| 263k run | 0.296 | 0.488 | 0.634 | 0.685 | 0.698 (394k) |
| as share of teacher | ~0.40 | ~0.66 | ~0.86 | ~0.94 | 0.95 |

Train loss (263k run; CE / soft CE): 2.03 / 1.91 at 10k, 1.79 / 1.64 at 30k, 1.55 / 1.45 at 120k, 1.44 / 1.35 at the
end; stage A ended at 2.78 / 2.66. The 27B's absolute losses will differ (another tokenizer and teacher), so compare
shapes and shares, not raw values. Stage times: 263k SFT 3.0 h; GRPO 33 s/step.

**What is watched, and how often:**
- Every 5 min (CLAUDE.md), from the Mac via a Monitor on a box-side heartbeat: process alive, log advancing, step rate,
  GPU memory and use, disk, $ spent so far.
- Box-side watchdog (works when the Mac sleeps): no log progress for 20 min -> dump state, stop the job, alert;
  queue finished -> sync results to the persistent disk and HF, then shut the box down.
- Every 100 steps: CE, soft CE, grad norm, LR, tokens/s.
- Every 10k examples: held-out FVE on a fixed 256-vector subset (~2 min) + 5 explanations printed for reading.
- At 30k / 60k / 120k / 240k / end: full held-out FVE (1,020), injection (1,200), notes; saved generations.
- Teacher labelling: every 10k notes: tokens/s, mean length, share hitting 256 tokens, parse rate, repetition, 3 notes read.
- GRPO: every 10 steps reward, KL, length, parse; every 50 steps FVE subset, injection, α 0 false alarms.

**Stop-and-fix rules (stop the stage, keep the last good checkpoint, write down why):**
- Loss NaN, or a spike > 3x that does not recover in 200 steps.
- Share of teacher at 30k < 0.25 or at 60k < 0.45 (7B: 0.40 / 0.66), unless the teacher label check explains it.
- FVE flat (< +0.005) over two evals while the LR is still above half its peak: probe LR before spending more.
- Throughput < 70% of the pre-flight figure, or memory creeping upward (leak).
- Teacher notes: parse failures > 2%, > 5% hitting the length cap, or repetition loops.
- GRPO: reward up while injection falls or α 0 false alarms rise 5 pts; KL > 0.1; mean length drifting > 20%; no FVE
  gain over 100 steps (plateau -> stop, it is done).
- Spend reaches ~$130 before R1 ends -> stop and ask Adam.

**LR, passes and data size: small probes, so the relationship is measured, not assumed (~$7):**
- At the 66k rung, three extra fresh runs: lr 5e-6 and 2e-5 (1.5 passes), and 1.0 vs 2.0 passes (lr 1e-5). These run
  before the 263k main run, so a better setting is used there.
- The 7B ladder already gives the size effect; the 27B ladder plus these probes give pairs x passes x LR at two teacher
  sizes.

**Learning log:** `docs/nladistil27b_live_handoff_<date>.md`, updated at every stage: what was run and
why, expected band, actual numbers, decision, what that suggests could change. Every eval also goes in
`runs/nladistil_scaling_ledger.jsonl`. Each stage ends with a short side-by-side against the 7B (FVE by examples seen,
by passes, by LR; gain per doubling; RL gain per 100k rollouts) and a few lines in `experiment_log.txt`.

**HF access:** write token at `<your HF write token file>` (account adamjb111), used only to create and push this
project's dataset and model repos; never printed or copied into logs or scripts.

## 16. Decisions (10 Oct, 11:08 UTC)

- HF dataset: `adamjb111/nla-qwen36-27b-L42-distill-data`, public. Model repo public too.
- GPU: Adam approves one GH200 (96 GB, $2.29/h) in place of the A100. Why it is likely cheaper overall: the teacher's
  writing is limited by memory speed (GH200 ~4 TB/s vs 1.55 TB/s on the 40 GB A100) and training by compute (~3x), so
  most stages should run ~2-2.5x faster for 15% more per hour; the exact bf16 teacher fits (no 8-bit check), and teacher
  + AR (89 GB) fit together. Estimate for the selection: ~$70-90 instead of ~$127-145. Risks: ARM CPU (some kernels may
  need building; test in pre-flight); availability (none free at 11:08 UTC; polling every 5 min); region may differ
  from the persistent disk (us-east-1), in which case HF is the store and a small disk is made in that region.
- Fallback: if no GH200 frees up by the time the Mac-side work is done, ask Adam: wait, or A100 40 GB with the 8-bit teacher.
- Update 10 Oct ~11:30 UTC: Adam prefers to wait up to 24 h for a GH200 (any region) rather than use an 8-bit teacher
  on an A100: the true bf16 teacher matters more than a day. Poller checks every 5 min until ~11:08 UTC 11 Oct. If the
  GH200 is outside us-east-1, results go to HF (public anyway) plus a small filesystem in that region, deleted after.
  If none by then, ask Adam again.
- Teacher data facts (Mac, Qwen3.6 tokenizer, 3k-row samples per split): AV-SFT 247k rows / 25.0k docs, AR-SFT 247k /
  25.0k docs, RL 500k / 50.0k docs; splits share no docs; ≤ 10 rows per doc; 99,956 docs in all. Prefix length p10 ~70,
  median ~190, p90 ~700-790, p99 ~2,200-2,400, mean ~340 tokens. Sonnet labels median 124 tokens (p90 139, max 179).
  Plan: sample the 233k train rows across all 100k docs (~2-3 per doc, more spread than 10 per doc), grouped by doc so
  ladder subsets nest; extraction runs each doc once up to its longest chosen prefix.

## 17. Nebius option (10 Oct, ~14:00 UTC, Adam)

Adam added Nebius (pre-emptible RTX PRO 6000 Blackwell, 96 GB) to CLAUDE.md and authorised up to $10 to test it.
Answer to my one question: if the test works, run the whole distillation on a pre-emptible RTX PRO 6000 under the same
$150 cap; fall back to the Lambda GH200 if Nebius capacity or the tests fail. Pre-emption rules (CLAUDE.md): a persistent
disk attached; every process watches for the shutdown signal and saves its latest checkpoint within the ~60 s notice.
