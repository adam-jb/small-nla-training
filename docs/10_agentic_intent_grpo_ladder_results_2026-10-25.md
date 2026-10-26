# Agentic-intent GRPO ladder: results, all 4 arms (25 Oct 2026)

Start model: GRPO526 (2B student NLA). Training: AR rebuild reward only, reward = -2(1 - cos(AR(note), v)) (= the rebuild loss, negated), arm J also -0.05 per junk note.
Round 1 stopped on the inherited KL guard (0.25) on every arm while dev was still rising; round 2 resumed with kl-max 1.0 and was cut by a budget stop at 01:45 UTC.
Raw files: `runs/agintent_results_20261025/<arm>/` (done.json, done_round1.json, evals.jsonl, metrics.jsonl, final_eval/). Models: private HF `adamjb111/<arm name with - for _>`.
(The HF repos also hold a stale STOPPED.json from round 1; round 2 stopped on "hours".)

## The models (where, and what to call them)
All private on HF, account adamjb111. Each repo: `final_lm/` (2B with the best LoRA merged in), `final_adapter.pt` (vector -> embedding adapter; needed with final_lm), evals.jsonl, metrics.jsonl, done.json, `final_eval/` (scores + every note).
| Short name | HF repo | Best step |
|---|---|---|
| GRPO526 (start) | `adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013` | - |
| AGI-r25 | `adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r25-rows6k-20261024` | 375 |
| AGI-r50 | `adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r50-rows12k-20261024` | 325 |
| AGI-r100 | `adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r100-rows24k-20261024` | 300 |
| AGI-r100-junk | `adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-junk05-agentmix-r100-rows24k-20261024` | 300 |
Local copies hold results only (no weights): `runs/agintent_results_20261025/<run name>/`.

## Paired final eval (start vs arm best, same rows; FVE; gain with group-bootstrap CI95)
| | r25 (6k rows) | r50 (12k) | r100 (24k) | r100 + junk penalty |
|---|---|---|---|---|
| best step / steps run | 375 / 379 | 325 / 367 | 300 / 400 | 300 / 317 |
| web held-out (1,020) | 0.739 -> 0.744, +0.005 [+0.001, +0.009] | 0.739 -> 0.744, +0.005 [+0.001, +0.008] | 0.739 -> 0.744, +0.005 [+0.002, +0.009] | 0.737 -> 0.745, +0.008 [+0.004, +0.011] |
| agent dev (1,180) | 0.374 -> 0.476, +0.102 | 0.374 -> 0.466, +0.092 | 0.374 -> 0.468, +0.094 | 0.374 -> 0.464, +0.090 |
| agent hold (3,449) | 0.348 -> 0.454, +0.105 [+0.098, +0.114] | 0.348 -> 0.448, +0.100 | 0.348 -> 0.446, +0.097 | 0.348 -> 0.443, +0.095 |
| unseen source (DeceptionBench) | +0.040 [+0.035, +0.046] | +0.040 | +0.037 | +0.043 |
| decision point, tempting (P_last x A) | -0.379 -> -0.148 | -0.379 -> -0.167 | -0.379 -> -0.156 | -0.374 -> -0.173 |
| hold by row type G / P_mid / P_last | 0.51->0.59 / 0.32->0.43 / -0.56->-0.29 | 0.59 / 0.42 / -0.31 | 0.59 / 0.41 / -0.31 | 0.59 / 0.41 / -0.32 |
| twin shift cos (385 pairs) | 0.141 -> 0.177, +0.037 [+0.028, +0.046] | 0.171, +0.031 | 0.168, +0.028 | 0.170, +0.027 |
| junk, hold notes | 25% -> 15% | 25% -> 21% | 25% -> 24% | 25% -> 1.2% |
| LEARNED / NO FORGETTING | yes / yes | yes / yes | yes / yes | yes / yes |

## Training signals (mean of first 20 -> last 20 steps)
| | r25 | r50 | r100 | J |
|---|---|---|---|---|
| reward (= -rebuild loss) | -0.251 -> -0.208 | -0.240 -> -0.219 | -0.253 -> -0.213 | -0.265 -> -0.220 |
| train cos | 0.874 -> 0.896 | 0.880 -> 0.891 | 0.873 -> 0.894 | 0.874 -> 0.891 |
| KL (max) | 0.001 -> 0.276 (0.33) | 0.001 -> 0.271 (0.33) | 0.001 -> 0.253 (0.33) | 0.001 -> 0.273 (0.33) |
| k2 / loss_kl | 0.43 / 0.0043 | 0.43 / 0.0043 | 0.40 / 0.0040 | 0.44 / 0.0044 |
| loss_pg | ~-0.0004 | ~-0.0001 | ~-0.0003 | ~-0.0006 |
| reward std in group | 0.018 -> 0.014 | 0.018 -> 0.014 | 0.018 -> 0.015 | 0.025 -> 0.016 |
| clipfrac | 0.0002 -> 0.0007 | same | same | same |
| length / parsed | 186 / 1.00 | 186 / 1.00 | 189 / 1.00 | 189 / 1.00 |
| injection hit at alpha 0.5 (step 0 0.97) | 0.97 | 0.96 | 0.97 | 0.96 |
| shuffled dev / web FVE | -0.72 / -0.89 | same | same | same |

## Reading
- All four arms learn agent text and do not forget web text: hold +0.095 to +0.105 FVE, web held-out up slightly (+0.005 to +0.008), length flat, injection unchanged.
- Data size barely matters: r25 (6k rows) is as good as or a hair better than r100. The ladder is flat, so more of this data would not help; the limit is elsewhere.
- Biggest gains on tool-heavy sources (Agent-SafetyBench +0.17-0.19, tau +0.13-0.16, ImpossibleBench +0.14-0.15); chat sources +0.05-0.07; unseen DeceptionBench +0.04.
- Decision points (last prompt token) gain most but stay below zero FVE: the notes still do worse than the mean agent vector at the exact point of choice. The twin shift (does the note move the way the vector moves between tempting and no-pressure twins) rises only 0.141 -> 0.17-0.18. So the 2B reads the setting much better, but the pressure itself only a little.
- Junk penalty works: junk 25% -> 1-2% for ~0.003-0.01 FVE cost on agent sets and a small web gain. Junk detector misses some cases (e.g. split by / and &), so the other arms' junk rates are undercounts.
- Dev was still rising ~+0.005 per 50 steps at the cut; best steps are near the end, so this is not a plateau yet.

## Cost
Vast invoices for the 5 boxes: $73.9 (cap $80). Waste: ~$3 vLLM install bug, ~$17 hung setup ssh (boxes idle 4 h).

## Check (25 Oct): is decision-point FVE < 0 a yardstick artefact? No, the reverse.
Script `scripts/agintent_rtype_baseline_check_20261025.py` (vectors from the HF dataset, cos from saved notes).
- Every decision-point row (P_last, 1,782 in hold) sits on the SAME token: the `\n` after `<|im_start|>assistant\n<think>`.
  So these vectors share a big common part: spread around their own mean (rawvar) 0.28 vs 0.40 for all agent rows;
  the mean has norm 0.85, so "always guess the typical decision-point vector" gets cos ~0.85. Per source: rawvar 0.13 (cos ~0.94).
- Notes rebuild these at cos 0.67 (start) -> 0.72 (arm), vs 0.91 for G rows. With the fair own-type yardstick, P_last FVE is
  -1.38 -> -1.0 (r100), worse than the -0.56 -> -0.31 reported. G rows look a bit lower (0.59 -> 0.50), P_mid a bit higher (0.41 -> 0.44).
- Likely cause: the AR rebuilds from text and never learned the template-token state; the notes describe the scene, the AR cannot
  turn that into "a vector at the <think> newline". Only ~10% of hold rows (and a similar share of train rows) are this token.
  Next check: score P_last with the type mean taken out of both v and AR(note) (centred cos), which needs the AR on a GPU.

## Teacher check (25 Oct): the 27B NLA teacher on the same rows
50 rows per row type from the final-eval hold sample + 51 web held-out rows; teacher = ceselder/qwen3.6-27b-nla-rl iter_000400 (HF backend, T=1),
same 27B AR. Box 57636222 (~15 min, ~$0.40, destroyed). Sample: `scripts/agintent_teacher_sample_20261025.py`; box script `.runtime/agintent/teacher_check.sh`;
scores in `data/agintent_teacher_check_20261025/`. FVE vs agent rawvar 0.426 [bootstrap CI over rows]:
| Row type (n=50) | Teacher 27B | GRPO526 | AGI-r25 | AGI-r100-junk |
|---|---|---|---|---|
| Decision point (P_last) | -0.55 [-0.82, -0.30] | -0.60 | -0.34 [-0.56, -0.11] | -0.38 |
| Mid-prompt (P_mid) | +0.30 | +0.30 | +0.37 | +0.38 |
| Reply (G) | +0.67 | +0.64 | +0.69 | +0.69 |
Web sanity: teacher 0.752 [0.712, 0.793] (n=51), as before. So the teacher is also below zero at the decision point: not a student
weakness but a limit of NLA + AR at the `<think>\n` template token. The trained 2B arms now match or beat the teacher on agent rows
(beats it on 94% of decision-point rows, small margins elsewhere).

## Decision-point follow-ups (25 Oct, box 57639240, ~40 min, ~$1)
- Tokens before the end (`scripts/agintent_tokens_before_20261025.py`, 100 units x 12 offsets, teacher + GRPO526 + AGI-r25; chart
  `agintent_tokens_before_decision_point_cos_by_offset_2026-10-25.png`; data `data/agintent_tokens_before_20261025/`).
  Bad tokens: `<think>` always (cos 0.40-0.46), `assistant` and the final `\n` after long prompts (0.53-0.62). Last word of the
  text / `<|im_end|>` / `<|im_start|>` read at 0.85-0.94 even in long prompts. All three readers fail at the same tokens.
- L42 dim 3994 is a "massive activation": mean |v| 62 vs 5.8 for the next biggest, 56% of a unit vector's squared length, but only
  4.8% of the across-token variance (std/mean 0.28 vs ~1.3 for a typical dim). Chart `agintent_qwen27b_L42_dim3994_size_vs_variance_2026-10-25.png`.
  It shrinks at the bad tokens (share 0.04-0.16); the AR always rebuilds it big (0.62-0.87 of the unit vector). Row cos vs top-5 share: corr 0.91.
- Dim-removed check (`scripts/agintent_ar_dims_check_20261025.py`): without dim 3994 the bad tokens are STILL worse than nearby text
  (long prompts, AGI-r25: decision `\n` 0.49, `<think>` 0.35, `assistant` 0.52 vs 0.71-0.78 for text; short: `<think>` 0.42 vs 0.86-0.89).
  So the big dimension explains part of the gap, not all: the notes/AR also miss the rest of the state at these tokens.
- Teacher table at n=250 per row type (+250 web): decision point teacher -0.46, GRPO526 -0.53, arms -0.27 to -0.30; mid +0.35 / +0.31 / +0.40-0.41;
  reply +0.53 / +0.48 / +0.56-0.57; web 0.75 / 0.73 / 0.74.
