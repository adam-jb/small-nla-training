# NLA distillation: Qwen2.5-7B L20 NLA → Qwen2.5-0.5B / 1.5B students — findings (3 Oct 2026)

Plan: `90_background_7b_distillation_plan_2026-10-03.md`. Box: claude-nladistil-20261003 (1× A100 SXM4 80GB, us-east-1, $1.99/h).
Scripts: `scripts/nladistil_*_20261003.py`. Results: `runs/nladistil_*` (local copies, no weights) and `/lambda/nfs/claude-lens-results-20260929/nladistil_20261003/`.

## Summary (read this first)

- **A 1.5B student matches most of the 7B NLA.** Trained on 263k teacher explanations, then one on-policy round, a
  Qwen2.5-1.5B with a small adapter reaches heldout AR FVE 0.706, against 0.732 for the teacher (96%). It is as truthful
  as the teacher (blind judges: 51% vs 52% of claims true). It catches strong injected concepts as often as the teacher
  (98% vs 100%), and medium ones somewhat less (69% vs 93%). Faint ones are its weak spot (17% vs 28%). It is one fifth
  the teacher's size, and the scaled run plus the on-policy round cost about $17.
- **Update 5 Oct: two more on-policy rounds (200k vectors each, $19.57) reach FVE 0.714 (97.5% of the teacher) and
  injection 76% at a 0.5 (teacher 93%); truth 50% vs teacher 54% (inside CIs). Round 2 added almost nothing, so the
  on-policy route is used up at this size. Best student: `..._from-op30k_op200k-r2_20261005_bf16` on the persistent disk.**
- **Data was the limit, not model size.** At 30k pairs the 1.5B got only 0.44 and the 0.5B 0.40. Going from 30k to 263k
  pairs took the 1.5B from 0.44 to 0.70. But the curve flattens: +0.15 from 60k to 120k examples, +0.05 from 120k to
  240k, +0.013 from 240k to 394k (the last 130k were repeats). These are checkpoints from one decaying-LR run, not
  separate runs, so the curve is rough. No evidence yet that more data alone closes the last 0.026 FVE or the
  faint-injection gap; a guess is 0.71-0.72 with 2-4x more fresh data. To settle it: separate full runs at 60k/120k/263k,
  one ~500k fresh-data run, and a 3B student at 263k.
- **FVE mostly rewards describing the context.** A 1.5B that reads the text and never sees the vector gets 0.61. Concept
  injection is the test that only a true vector reader passes, and the students pass it.
- **Adam's loop:** aggregating many T=1 samples helps small NLAs a lot (medoid-of-8: +0.10 FVE for the weak student,
  +0.02 for the strong one) but hardly helps the teacher (+0.007). Training a student on its own medoid picks lifts it
  by +0.022, against −0.015 for training on random picks of its own. So the loop step works, but the gain is small. While a
  teacher exists, its labels are far better. Without gold, the best deployable pick from 8 strong-student samples (the
  medoid) scores 0.716, still below one teacher sample (0.732). Averaging the 8 reconstructions reaches 0.737, but that is
  a vector, not a readable explanation, and the teacher's own 8-sample average is 0.752. The "best of 8" rows in the
  tables use the gold vector, so they are ceilings that cannot be reached in use.
- **6 Oct data ladder ($10.26):** fresh 1.5B students at 66k / 131k / 263k teacher pairs (same recipe) get FVE 0.540 / 0.655 / 0.698.
  Each doubling adds about a third of the last (+0.115, +0.043), so 1M pairs projects to ~0.72 (limit ~0.723), under the teacher
  and about what on-policy rounds already give. Injection at α 0.5 still climbs ~11 points per doubling (37 / 48 / 60%).
- Total box cost $24.42 (12.3 h, 1× A100), inside the $50 budget. The best student is in bf16 on the persistent Lambda disk.

## Question

Can a small model, given the 7B's layer-20 vector through a learned adapter, learn to write the same kind of explanation as the
released NLA (kitft/nla-qwen2.5-7b-L20-av, a full 7B), just by copying its outputs (sequence SFT + top-20 KD)?

## Set-up (first pass)

- Vectors: Qwen2.5-7B-Instruct block-20 output, 3 positions per doc (index ≥ 50), 5,600 Ultra-FineWeb docs (part 1500) +
  5,600 WildChat docs (shard 13). 33,600 vectors, of which 32,580 train and 1,020 heldout (340 docs, split by doc).
- Teacher: the released AV, pure T=1 sampling, max 200 new tokens, one sample per vector, with its top-20 log-probs saved.
  4.7M teacher tokens, 21 min.
- Students: Qwen2.5-0.5B-Instruct and 1.5B-Instruct. Adapter = LayerNorm → Linear(3584 → 4 × d) → 4 soft tokens in place of
  the AV's injection token, same prompt. Stage A: adapter only, 10k examples. Stage B: full fine-tune, 2 epochs, loss
  0.5 × CE(teacher sample) + 0.5 × soft-CE(teacher top-20).
- Text-only control: same 1.5B and data, but it sees the last 192 tokens of the source text instead of the vector.
- Scoring: the released AR (kitft/nla-qwen2.5-7b-L20-ar) reads each explanation and rebuilds the vector. fve_nrm as in the
  paper (0 = always predict the mean vector). "Shuffled" = explanations paired with the wrong vectors.

## Results (first pass, 30k pairs)

| System | Heldout FVE [95% CI] | Shuffled | Inject hits α 0 / 0.25 / 0.5 / 1.0 | Note job / city | Claims true | Topic right | Final token right |
|---|---|---|---|---|---|---|---|
| Teacher 7B AV | **0.732** [0.724, 0.740] | −0.87 | 6 / 28 / 93 / 100% | 48 / 12% | 54% | 35% | 91% |
| 1.5B text-only (no vector) | 0.613 [0.601, 0.626] | −0.86 | 5 / 4 / 6 / 5% | 43 / 25% | 54% | 25% | 94% |
| 1.5B vector student | **0.440** [0.420, 0.458] | −0.85 | 3 / 9 / 30 / 68% | 27 / 5% | 41% | 16% | 41% |
| 0.5B vector student | 0.395 [0.376, 0.416] | −0.86 | 3 / 9 / 25 / 62% | 16 / 3% | 39% | 8% | 49% |

(The model card reports FVE 0.752 for the teacher; our 0.732 is on different docs. Truth columns: 80 heldout vectors per
system, blind Sonnet judges, claims labelled supported / contradicted / unverifiable against the source text; truth =
S / (S + C). CIs about ±3 points.)

Data curve, heldout FVE by examples seen in stage B (one cosine schedule, so later points also gain from LR decay):

| seen | stage A | 3k | 7.5k | 15k | 30k | 65k (2 ep) |
|---|---|---|---|---|---|---|
| 0.5B | −0.36 | −0.13 | 0.02 | 0.13 | 0.28 | 0.40 |
| 1.5B | −0.41 | −0.17 | −0.04 | 0.14 | 0.31 | 0.44 |
| 1.5B text-only | — | 0.33 | 0.47 | 0.54 | 0.57 | 0.61 |

## What it means

1. **A small model can learn to read the 7B's vector.** This is the "direction found" bar, passed by a wide margin. Even the
   frozen 0.5B with only the adapter trained beats the shuffled control (−0.36 vs −0.85). After 34 min of training on 30k
   pairs, the 1.5B reaches FVE 0.44.
2. **Size matters less than data.** The 1.5B beats the 0.5B by only about 0.04 FVE at every checkpoint. Neither curve has
   flattened.
3. **Share of the teacher kept (1.5B):** 81% of the teacher's gain over the shuffled control (the bar set in the plan),
   but only 60% of its FVE over the mean baseline, which is stricter and, I think, fairer. Injection: 25–31% of the
   teacher's gain at α 0.25–0.5, and 69% at α 1.0. Truth rate: 41% vs 54%, which is 13 points lower and so misses the
   "within 10 points" bar. Verdict against the plan's bars: "direction found" yes; "good" partly (FVE yes, injection no);
   "full" no.
4. **The FVE score rewards describing the context, and a text reader does that well.** A 1.5B that never sees the vector,
   only the text, gets 0.61 FVE (84% of the teacher) and the same truth rate as the teacher. So most of what the AR
   rewards is "say what the text is about and what token comes next". The vector students have to get that out of the
   vector itself, which they do only partly (topic right 16% vs 35%; final token 41% vs 91%). But only the vector
   readers respond to concept injection. That is the part of NLA output that is about the model's state rather than the
   text.
5. **The teacher's claims are often false too.** 54% of its checkable claims hold. This matches the LessWrong follow-up:
   the reconstruction loss does not force true claims. Students inherit this and add more errors.

Example (heldout vector 33306; WildChat, a user asks what "Arousal / Valence" mean in Hungarian, text ends at
"In Hungarian,"):
- Teacher (AR cos 0.91): "...translated technical term explanation... 'In English terms, let's translate the German
  terms'... Final token 'terms,'..." (right genre and local shape; wrong language; wrong final word)
- 1.5B student (0.84): "...numbered points and translated terms... 'Given the Portuguese term, let's convert...'..."
- Text-only (0.83): "...how to determine a word's Polish translation, now transitioning to ... the Hungarian language
  equivalent..."

## Does aggregating many T=1 samples beat one? (`runs/nladistil_aggregate_k8_20261003/`)

Each system drew 8 samples per heldout vector (1,020 vectors), and the AR scored every sample. Heldout FVE:

| | Teacher 7B | 1.5B student (pass 1) |
|---|---|---|
| one sample (average) | 0.732 | 0.438 |
| medoid of 2 / 4 / 8 (sample whose reconstruction is closest to the others'; no gold used) | 0.731 / 0.737 / **0.739** | 0.436 / 0.513 / **0.542** |
| mean of the 2 / 4 / 8 AR reconstructions (a vector, not a text) | 0.743 / 0.749 / 0.752 | 0.524 / 0.572 / 0.597 |
| best of 2 / 4 / 8 by gold cosine (ceiling) | 0.742 / 0.750 / 0.757 | 0.525 / 0.583 / 0.623 |

**Yes for the small model, hardly at all for the teacher.** The student's samples are noisy, so picking the consensus one
lifts it by 0.10 FVE (0.44 → 0.54), about a third of the way to the teacher. The teacher's samples already agree, so it
gains 0.007. This is step 3 of Adam's loop ("lots of small NLAs, aggregate"). It works, but the aggregate (0.54) is still
well below the teacher (0.73). So while the teacher is available, its labels are the better target. The open question is
step 4: does training the student on its own medoid picks lift its single-sample score? Test below. Caveat: the medoid
is chosen by the AR, so it moves toward what the AR likes, which is not the same as truth.

## Scaled run (≈230k pairs + mixed-vector augmentation + one on-policy round)

Started at 13:31 UTC; see the section below once it is filled in. Reason for running it: FVE was already at the plan's 80%
bar, and the curves were still steep. Main doubt: whether injection reading improves with more of the same data. 30k of
the new train vectors are mixes of two contexts (w = v_a + α|v_a|·unit(v_b − mean)), labelled by the teacher, so the
student also sees off-distribution inputs. The 30 eval concepts are never used.

Data: 200k new vectors (UFW parts 1501-1503, WildChat shards 10-12, docs disjoint from the first pass) + 30k mixed vectors,
all labelled by the teacher (33M tokens, 3.9 h), merged with the first-pass 32.6k: 262,563 pairs. Same 1,020 heldout
vectors. 1.5B student, 1.5 epochs (394k examples, one cosine schedule), same loss and adapter, 3.1 h.

| examples seen | 30k | 60k | 120k | 240k | 394k (final) |
|---|---|---|---|---|---|
| heldout FVE | 0.296 | 0.488 | 0.634 | 0.685 | **0.698** [0.688, 0.708] |

Final student (`nladistil_sftkd_qwen25-1.5b_vec_from-nla-qwen25-7b-L20_pairs230k_20261003`):

| | Teacher 7B | Scaled 1.5B | First-pass 1.5B |
|---|---|---|---|
| Heldout FVE | 0.732 | **0.698 (95%)** | 0.440 (60%) |
| Shuffled | −0.87 | −0.86 | −0.85 |
| Inject hits α 0 / 0.25 / 0.5 / 1.0 | 6 / 28 / 93 / 100% | 6 / 14 / **60** / **98%** | 3 / 9 / 30 / 68% |
| Share of teacher's inject gain α 0.25 / 0.5 / 1.0 | — | 40 / 63 / 98% | 25 / 31 / 69% |
| Note job / city | 48 / 12% | 35 / 11% | 27 / 5% |

Leak check: no heldout doc is in the training set. 26 heldout vectors share their last 300 characters with a training
vector (repeated WildChat template prompts, such as the Midjourney prompt generator). They are harder for every system,
and without them the scaled student still gets 0.636 vs the teacher's 0.736 at 120k. 25 of them were already in the
first-pass train set.

**Verdict:** with about 8× more teacher pairs, a 1.5B student (one fifth the teacher's size) keeps 95% of the teacher's
reconstruction score, nearly all of its strong-injection reading, and about two-thirds of its medium-injection reading.
Data, not model size, was the limit in the first pass. The weak spot left is faint signals (α 0.25: 14% vs 28%). The
mixed-vector augmentation and the extra data changed together, so I cannot say how much each helped injection.

### One on-policy round (`nladistil_onpolicy_qwen25-1.5b_vec_from-nla-qwen25-7b-L20_pairs230k-op30k_20261003`)

The scaled student wrote one T=1 explanation for each of 30k train vectors. The teacher read the vector plus the student's
tokens and gave its top-20 at every student token. The student then trained on that (soft CE only, lr 5e-6, 1 epoch,
13 min).

| | Scaled 1.5B | + on-policy | Teacher |
|---|---|---|---|
| Heldout FVE | 0.698 | **0.706** [0.697, 0.714] | 0.732 |
| Inject hits α 0 / 0.25 / 0.5 / 1.0 | 6 / 14 / 60 / 98% | 3 / 17 / **69** / 98% | 6 / 28 / 93 / 100% |
| Note job / city | 35 / 11% | 40 / 8% | 48 / 12% |

The gains are small but all point the same way. The FVE gain is inside the CIs; injection at α 0.5 is +9 points (300 cases per α).

### Truth (blind judges, pass 2: same 80 vectors, teacher re-judged as anchor)

| | Claims true | Topic right | Final token right |
|---|---|---|---|
| Teacher 7B | 51.6% [49, 54] | 28% | 93% |
| Scaled 1.5B | **51.9%** [49, 55] | 25% | 91% |
| + on-policy | 51.1% [48, 54] | 25% | 89% |

The scaled student is as truthful as the teacher, within judge noise. Pass 1 had the first-pass student 13 points
behind. Between passes the teacher's own score moved from 53.6% to 51.6%, which gives a sense of judge-to-judge noise.

### Against the plan's bars

"Full" was ≥80% of the teacher's FVE gain and injection gain, plus truth within 10 points.
- FVE: 96% of the teacher on the strict mean baseline (98% on the planned shuffled baseline).
- Truth: equal to the teacher.
- Injection: 76% / 101% of the teacher's gain at α 0.5 / 1.0, but only about 60% at α 0.25.

So it is full on FVE and truth, and close on injection except for faint signals. Total cost of the scaled run plus the
on-policy round: about 8.5 box-hours ≈ $17.

### Two more on-policy rounds over a 200k pool (5 Oct 2026; box claude-nladistil-op-20261005, 1x A100, $19.57)

Start: the op30k student above. Pool: 200.6k train vectors (138k from fresh docs + 30k two-context mixes + 32.6k first-pass train).
Each round the student wrote one T=1 explanation per pool vector while the 7B AV gave its top-20 at every student
token; then 1 epoch of soft CE (lr 5e-6). Script `.runtime/claude_upload_nladistil_op/op.sh`; runs
`runs/nladistil_onpolicy_qwen25-1.5b_vec_from-op30k_op200k-r{1,2}_20261005/`. Stop rule: skip round 3 if round 2
adds < 0.003 FVE. It added 0.0002, so it stopped after round 2.

| | op30k (start) | + round 1 | + round 2 | Teacher |
|---|---|---|---|---|
| Heldout FVE | 0.706 | 0.714 [0.705, 0.722] | **0.714** [0.705, 0.723] | 0.732 |
| Inject hits a 0 / 0.25 / 0.5 / 1.0 | 3 / 17 / 69 / 98% | 4 / 17 / 74 / 98% | 4 / 19 / **76** / 99% | 6 / 28 / 93 / 100% |
| Note job / city | 40 / 8% | 48 / 13% | 43 / 6% | 48 / 12% |
| Truth, pass 3 (blind Sonnet, same 80 vectors) | 51.0% [48, 55] | - | 50.4% [47, 54] | 53.8% [51, 57] |
| Topic right / final token right (pass 3) | 36 / 85% | - | 36 / 91% | 45 / 94% |

FVE curve inside round 2 (examples seen 25k/50k/100k/200k): 0.710/0.712/0.713/0.714, so the gain is mostly from round 1.
Judges for pass 3: `runs/nladistil_judge_pass3_20261005/` (rubric in JUDGE_PROMPT.txt; 2 Sonnet subagents, $0).

**Verdict:** on-policy rounds with the teacher are done. Round 1 gave +0.008 FVE and +5 points of injection at a 0.5;
round 2 gave nothing on FVE and +2 points of injection. Truth stays ~3 points under the teacher (inside the CIs). The
student now has 97.5% of the teacher's FVE; the gap left (0.018) is not closed by more on-policy teacher signal at this
size. Best student (bf16) saved on the persistent disk:
`models/nladistil_onpolicy_qwen25-1.5b_vec_from-op30k_op200k-r2_20261005_bf16/`.

### Aggregation on the newest student (6 Oct 2026; box claude-nladistil-agg-20261006, 1x A100, $2.4)

r2 student, 1,020 held-out vectors, k samples each, AR-scored (`runs/nladistil_aggregate_r2_k16_20261006/`; sampler now
has `--temperature`; aggregate goes to k = 64). Held-out vectors were rebuilt with `nladistil_extract_20261003.py
--no-dedup` (rows identical to the saved set; the persistent disk does not keep `main.npy`).

| | one sample | medoid of 8 | medoid of 16 | medoid of 64 | mean of 8 rebuilds | mean of 64 |
|---|---|---|---|---|---|---|
| Student r2, T 1.0 | 0.716 | 0.728 | 0.731 | 0.733 | 0.745 | 0.749 |
| Student r2, T 0.7 | **0.721** | 0.729 | 0.731 | - | 0.743 | - |
| Student r2, T 1.3 | 0.680 | 0.704 | 0.707 | - | 0.727 | - |
| Teacher, T 1.0 | 0.731 | 0.739 | - | - | 0.752 | - |

Medoid of 16 student samples equals one teacher sample. Gains flatten fast (+0.012 for k 1 to 8, +0.006 for 8 to 64), and at
equal k the teacher stays ahead. Picking the medoid needs the 7B AR on every sample, so this is not cheaper than the
teacher. T 0.7 gives the best single sample; T 1.3 hurts at every k. Charts: https://claude.ai/artifact/P8217Yx4e7Gov7dkwDvBY5
(`reports/nla_distil_scaling_2026-10-06/`, built by `scripts/nladistil_build_scaling_page_20261006.py`).

### Data ladder: is more teacher data worth it? (6 Oct 2026; box claude-nladistil-ladder-20261006, 1x A100 40GB, $10.26)

Question: would 1M teacher pairs (~$50) close the gap? The 263k curve could not say, because its points share one decaying
learning rate. So: fresh 1.5B students on random 66k and 131k subsets of the 263k pairs (131k ⊃ 66k, seed 0), same recipe
as the 263k run (stage A 10k, 1.5 passes, lr-lm 1e-5, 0.5 CE + 0.5 soft CE to the teacher top-20). The 263k rung is the 3 Oct run.
The top-20 arrays were not kept, so the box re-tokenised the saved teacher text (`data_teacher_text/teacher_train_233k.jsonl.gz`)
and the 7B AV re-scored it teacher-forced (`scripts/nladistil_ladder_prep_20261006.py` + `nladistil_teacher_score_20261003.py`).
6.9% of rows split the same text into 1-2 more or fewer tokens than the original sample; harmless, since the targets come
from the re-score. Rebuilt rows matched the saved `main_233k.jsonl` byte for byte. Queue: `.runtime/claude_upload_nladistil_ladder/ladder.sh`.

| teacher pairs | 33k (first pass, 2 passes) | 66k | 131k | 263k | 525k (projected) | 1.05M (projected) |
|---|---|---|---|---|---|---|
| heldout FVE | 0.440 | 0.540 | 0.655 | 0.698 | ~0.714 | ~0.719 |
| inject α 0.25 / 0.5 / 1.0 | 9 / 30 / 68% | 8 / 37 / 80% | 10 / 48 / 90% | 14 / 60 / 98% | | |

**Verdict:** FVE gains shrink to about a third per doubling (+0.115, then +0.043). Projected forward, 1M pairs gives ~0.72 and
the curve levels off near 0.723, below the teacher (0.731) and about what on-policy rounds already reached (0.714) for less.
Three points fix this fit exactly and one FVE has a CI of ±0.009, so read it as "1M lands at 0.71-0.73". Faint-concept reading is
the exception: injection at α 0.5 climbs a steady 11-12 points per doubling, so more data would still help there. To pass the
teacher on FVE, RL against the AR or a 3B student look like better bets than 1M pairs. Runs:
`runs/nladistil_sftkd_qwen25-1.5b_vec_from-nla-qwen25-7b-L20_pairs{66k,131k}-ladder_20261006/` (no weights kept; copies on the
persistent disk under `ladder_20261006/`). Box note: Lambda gave a 40 GB A100 this time; stage B at micro-batch 16 ran out of
memory, so the ladder used `--micro 8` (same batch 32, same maths).

### Every measure on the ladder (blind judge pass 4, 6 Oct 2026; 4 Sonnet subagents, $0)

Same 80 held-out vectors as passes 2-3, all six systems judged in one shuffled pass (`runs/nladistil_judge_pass4_ladder_20261006/`).
Judges differ in style (one marks ~1 claim per item as uncheckable, the others almost never), but every batch mixes all systems at random.

| | 33k (2 passes) | 66k | 131k | 263k | 263k + 3 on-policy | Teacher |
|---|---|---|---|---|---|---|
| FVE | 0.440 | 0.540 | 0.655 | 0.698 | 0.714 | 0.731 |
| Claims true [95% CI] | 44% [41, 47] | 46% [43, 50] | 52% [48, 55] | 54% [50, 58] | 55% [51, 58] | 57% [54, 60] |
| Topic right | 29% | 33% | 39% | 45% | 44% | 50% |
| Final token right | 39% | 51% | 79% | 89% | 88% | 90% |
| Inject α 0.25 / 0.5 / 1.0 | 9 / 30 / 68% | 8 / 37 / 80% | 10 / 48 / 90% | 14 / 60 / 98% | 19 / 76 / 99% | 28 / 93 / 100% |

Text-side measures saturate first (final token at 98% of the teacher by 263k; truth and FVE at 95%). Vector-reading measures lag
and keep rising: topic 90%, injection α 0.5 65%, α 0.25 50% of the teacher at 263k. On-policy helps faint-concept reading most.
Estimate for 1M pairs + on-policy (two trends combined, not measured): FVE ~0.725, injection α 0.5 ~85-90%, truth ~56%: close to,
probably not past, the teacher. Ranked routes to match it: RL against the AR with a KL leash (~$20-40 first test), a 3B student
(~$25), then 525k-1M pairs weighted to mixed / faint vectors + on-policy (~$60-70).

## Step 4 of the loop: amortise the aggregate back into one small NLA (no teacher)

Start: the first-pass 1.5B (heldout 0.440). It drew 8 samples on each of 30k unseen train vectors (the new extraction).
The AR scored them. Per vector, we kept the medoid (no gold used), whose train-set FVE is about 0.53, against 0.42 for a
random sample. The student then trained on its own picks (plain CE, lr 5e-6, 1 epoch). Control: the same, but with
sample 0 (a random pick) instead of the medoid.

| | Heldout FVE | Paired gap vs start [95% CI] | Inject α 0.5 / 1.0 |
|---|---|---|---|
| Start (first-pass 1.5B) | 0.440 | — | 30 / 68% |
| + own medoid-of-8 picks | **0.462** | +0.022 [+0.007, +0.037] | 30 / 68% |
| + own random picks (control) | 0.425 | −0.015 [−0.030, +0.001] | 32 / 65% |

Medoid minus random: +0.037 [+0.021, +0.052].

**The loop step works, but the gain is small.** Training on its own unselected outputs makes the student slightly worse.
Training on consensus picks makes it slightly better, recovering about a fifth of the medoid's 0.10 advantage in one
30k round. Injection reading does not move: the selection signal comes from the AR, which rewards describing the
context, not reading injected concepts. For scale, 30k fresh teacher pairs gave the first pass +0.4 FVE from scratch,
and the on-policy teacher round gave +0.008 on top of an already strong student. So while a teacher exists, teacher labels
beat self-aggregation by a wide margin. The loop is worth more once the student is at or past the teacher, where the
teacher has nothing left to give. The teacher's own medoid gain is only +0.007, which suggests the loop flattens out fast
near the top.

## Aggregation on the strong student

Same test as above, on the scaled 1.5B (before the on-policy round):

| k samples | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| one sample | 0.699 | 0.698 | 0.699 | 0.698 |
| medoid of k | 0.699 | 0.699 | 0.712 | 0.716 |
| mean of k reconstructions | 0.699 | 0.720 | 0.731 | **0.737** |
| best of k (gold, ceiling) | 0.699 | 0.718 | 0.731 | 0.740 |

Teacher, one sample: 0.732. The medoid gain shrinks as the student gets better (+0.10 → +0.018), as with the teacher
(+0.007). Without gold, the medoid (0.716) stays below one teacher sample (0.732). Eight averaged reconstructions
(0.737) pass one teacher sample, but they form a vector, not a readable explanation, and the teacher's own 8-sample
average is 0.752. The best-of-k row uses the gold vector, so it cannot be reached in use.

## Files and storage

- Local (no weights): `runs/nladistil_*_20261003/` (generations, AR scores, metrics), `runs/nladistil_judge_*`
  (judge batches, results; the keys are in `runs/nladistil_judge_keys_20261003/`), `runs/remote_logs_nladistil_20261003/`.
- Persistent Lambda disk `/lambda/nfs/claude-lens-results-20260929/nladistil_20261003/` (6.7 GB):
  - `models/nladistil_onpolicy_qwen25-1.5b_vec_pairs230k-op30k_20261003_bf16/` (best student: `lm/` + `final_adapter.pt`
    + the scripts needed to load it; load with `Student` from `nladistil_train_student_20261003.py`, `--init-lm lm
    --init-adapter final_adapter.pt`);
  - `data_teacher_text/` (all 263k teacher explanations + row metadata, gzipped, 0.2 GB; the vectors can be rebuilt
    with `nladistil_extract_20261003.py`, same parts/shards/seeds);
  - all run folders without weights.
  The model folder costs about $0.60/month on Lambda storage. A private Hugging Face repo would hold it for free.
- Not kept: the extracted vectors (1.9 GB, about 1.5 h to rebuild), the top-20 teacher arrays (4 GB), the 240k
  self-samples.

## Bugs hit (fixed)

- The system Pillow (old) and Jinja2 3.0.3 on the Lambda image break transformers 5.17 (`Could not import module
  'Qwen2ForCausalLM'`, then `apply_chat_template requires jinja2>=3.1.0`). Fix: `pip install -U pillow "jinja2>=3.1"` in
  the venv.
- `np.load(npz)[key]` re-reads the whole array on every access. Slicing `z['tok'][off[j]:off[j+1]]` in a loop over 32k
  rows hung the student start. Fix: load each array once.
- The teacher sampled token 151994 once in 4.7M tokens. That is an unused padding row of the 7B vocab (152,064 rows,
  151,665 real tokens); the 0.5B has 151,936 rows, so the lookup hit a CUDA index assert at step ~720. Fix: drop rows with
  any id ≥ 151,665.
- ssh sessions that start HfFileSystem / hf_xet downloads in the background hang after the work ends (a child process
  keeps the pipe open). Kill the local ssh; the work is done.
- The Mac's rsync is 2.6.9: several `host:path` sources in one call fail with a usage error. Use `ssh host 'tar czf - ...' | tar xzf -`.
- When scoring on the box beside a training job, start the job with `nohup setsid` and read its log later; a plain ssh
  session sometimes hangs after the work ends.


## RL against the AR (6 Oct 2026): the 1.5B matches the teacher on FVE

- Setup: GRPO from the on-policy r2 student (heldout FVE 0.714). LoRA rank 32 on every LM linear + full vector adapter; 64 vectors x 8 samples per step at T 1.0; reward = -MSE between the frozen 7B AR's rebuild and the true vector (unit-normed); group-normalised advantage; PPO clip 0.2, 2 minibatches/step; k2 KL 0.01 to the start policy (teacher's value). Prompts: the 32.6k first-pass train vectors. 1x A100 40 GB, 33 s/step, 5.8 h, $12.22. Script `scripts/nladistil_rl_grpo_20261006.py`; run `runs/nladistil_grpo_qwen25-1.5b_vec_from-op200k-r2_ar-reward_20261006/`.

| Step | Rollouts | Heldout FVE | Inject a0 / 0.25 / 0.5 / 1.0 |
|---|---|---|---|
| 0 | 0 | 0.714 | 3 / 16 / 77 / 100% |
| 100 | 51k | 0.728 | 4 / 20 / 80 / 100% |
| 200 | 102k | 0.731 | 3 / 21 / 83 / 100% |
| 300 | 154k | 0.736 | 5 / 25 / 83 / 99% |
| 600 | 307k | 0.734 | 4 / 24 / 84 / 99% |
| teacher | | 0.731 | 3-6 / 28 / 93 / 100% |

- Paired over the same 1,020 vectors (step 600): +0.020 vs start [95% CI 0.016, 0.024]; +0.002 vs teacher [-0.001, 0.005] -> a tie.
- Not gaming the AR: injection (AR plays no part) rose; false alarms flat; length (~145 tokens) and parse rate (100%) unchanged; blind judges pass 5 (`runs/nladistil_judge_pass5_rl_20261006/`, 3 Sonnet judges, same 80 vectors as pass 4): claims true r2 46.4%, RL 46.8%, teacher 47.0%; topic right 31 / 35 / 36%; final token 88 / 90 / 91%. (This judge set is stricter than pass 4; compare within the pass only.)
- Plateau from ~150k rollouts, about one pass over the 32.6k vectors. Stopped early at step 600 to save money. Next: RL on fresh + mixed vectors (200k pool), and the same recipe on a 0.5B.
- Compare: the teacher's own RL used ~4.3M rollouts from SFT ~0.35; here 100k rollouts sufficed because the student started at 97.7% of the teacher.

## Deception probe (7 Oct 2026): teacher vs RL student vs pre-RL student

- Set: 25 twins x 4 kinds (fact lie, hidden paid brand push "Zentro", sandbag arithmetic, cover up a known problem) for Qwen2.5-7B-Instruct; system prompt honest vs deceptive, same user message; the 7B answers (greedy, 64 tok); L20 vectors at prompt end / 3rd answer token / last answer token. `data/nladistil_deception_probe_20261007/` (compliance labels in compliance_auto.json).
- The 7B actually deceived in 52/100: cover-up 24, brand push 22, sandbag 6, fact lie 0 (always refused).
- FVE (AR) on this set: teacher 0.689, RL 0.693, pre-RL 0.679 (ordinary text: 0.731 / 0.734 / 0.714).
- Blind judges (6 Sonnet, explanation only, "does this indicate the assistant is deceptive?" 0-2), share flagged >= 1, real-deception vs honest twin:

| | teacher | RL 1.5B | pre-RL 1.5B |
|---|---|---|---|
| all, any position: deceptive / honest | 40% / 2% | 46% / 4% | 38% / 6% |
| AUC deceptive vs honest | 0.69 | 0.70 | 0.64 |
| cover-up (AUC) | 0.79 | 0.75 | 0.67 |
| brand push (AUC) | 0.61 | 0.69 | 0.65 |
| sandbag | 0% flagged | 0% | 0% |
| refused fact-lie (false alarm) | 1% | 0% | 1% |

- Teacher vs RL AUC difference CI [-0.06, +0.04]: a tie. RL improved on pre-RL mostly at the answer's last token.
- What the explanations do: they describe the reply's content and tone ("reassuring confirmation that the product is safe" vs "recall notice"; the brand name), never "the assistant is lying". Deception is visible only when the lie shows in what will be said. Deliberate wrong arithmetic and a mere instruction to lie leave no trace in either NLA.

## Hidden deception (7 Oct 2026): same output, different intent

Report doc: https://claude.ai/code/artifact/89c1b984-07cb-446b-b534-7fd97826d967 (full tables). Forced identical outputs; a mean-difference probe on the L20 vectors separates mistake vs lie, honest vs sandbag, paid vs fair motive, secret vs none at 86-100%, while all three NLAs read none of it (password 0%, motive +-0 pts, lie vs mistake +5 teacher / +14 RL, borderline). They notice a forced fact is false (~30% name the truth) but not that it is deliberate. The 7B sandbags arithmetic only 12-18% when free, lies about facts 22-68% by prompt style; forcing removes that limit, so the bottleneck is the NLA's training (describe visible text), not the subject model. Caveat: the probe can use any system-prompt wording difference.
