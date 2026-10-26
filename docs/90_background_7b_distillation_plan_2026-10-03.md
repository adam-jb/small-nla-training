# NLA distillation, first pass: Qwen2.5-7B NLA -> Qwen2.5-0.5B / 1.5B students (plan, 3 Oct 2026)

**Status: PLANNED, not run. No GPU launched. Needs Adam's go.**

## Aim

NLAs are costly because the explainer (the AV) is a full copy of the model it reads. Test whether a much smaller
LM can learn to read the 7B model's vectors and write the same kind of explanation. This first pass only asks:
does the student read the vector at all (a "direction"), and how fast does quality grow with data? It is not
expected to give a full distillation. The data curve tells us what a full one would cost.

Background and reasoning: the chat on 3 Oct (this file sums it up). Paper: https://transformer-circuits.pub/2026/nla/index.html .
Robustness follow-up (rebuild score is a weak judge of truth): https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to .

## The teacher: facts checked on 3 Oct

Code: github.com/kitft/natural_language_autoencoders, commit 0577769b55ad4fdd96d159e983361b97fa4e7331 (cloned to scratch).
Models: HF `kitft/nla-qwen2.5-7b-L20-av` and `kitft/nla-qwen2.5-7b-L20-ar` (both released). No training data released.

- Subject model: Qwen2.5-7B-Instruct, d_model 3584. Vector = output of decoder block 20 (forward hook on
  `model.model.layers[20]`, = HF `hidden_states[21]`), one token position, at least 50 tokens into the doc, raw text.
- AV: copy of Qwen2.5-7B-Instruct after SFT + RL. The vector, times `injection_scale` 150.0, replaces the embedding of
  token `㈎` (id 149705) in this prompt (from `nla_meta.yaml`):
  "You are a meticulous AI researcher ... describe the semantic content of that activation vector ... <concept>㈎</concept>
  Please provide an explanation." Output: `<explanation>` with 2-3 short snippets. Inference default max_new_tokens 200,
  temperature 0.7 (CLI) / 1.0 (client default).
- AR: Qwen2.5-7B cut to 20 layers + affine head on the last token. Prompt "Summary of the following text: <text>{explanation}</text> <summary>".
  `mse_scale` 59.87 is used for FVE.
- Training: warm-start data = Claude Haiku 4.5 summaries of the text up to the position (max 300 tokens);
  ~1M vectors (100k docs x 10 positions), split 250k AV-SFT / 250k AR-SFT / 500k RL. Model card: data 50% Ultra-FineWeb,
  50% WildChat-1M; in-distribution FVE 0.752. RL: GRPO, global batch 1024, 4,199 rollout steps, KL to init AV.
  Paper: SFT alone gives FVE ~0.3-0.4; RL takes it to 0.6-0.8.
- Inference engine: SGLang with `input_embeds` (vLLM also works).

## Why it can work, and why it may not

For:
1. The teacher sees only the vector. So every word of its explanation is (noisily) a function of the vector alone.
   The labels are learnable in principle. (Haiku's warm-start summaries were not: they saw the text.)
2. Much of a mid-layer vector is readable by a linear map (topic, current word, likely next words). Our own linear
   readers learned topic from 40k windows. Adapters that map one model's states into another's input work in LLaVA
   (image encoder -> LM) and cross-model Patchscopes.
3. The student already writes fluent text. It only has to learn the mapping and the style.

Against:
1. Even reading its own space, the teacher needed 250k SFT pairs to reach FVE 0.3-0.4, then ~4M RL rollouts.
   We plan 30k pairs.
2. A 0.5B model knows fewer ideas, so it cannot name rare ones.
3. The teacher mostly describes the text (an NLA fed only the final paragraph got FVE 0.67 vs 0.70), so a student
   can look good by learning the easy part.

Rough guesses (not measured): student clearly reads the vector (beats the shuffled-vector control, gets the topic)
~90%; 1.5B keeps half or more of the teacher's FVE gain at 30k pairs ~50%; full distillation at 30k pairs ~10%.

## Data

- 31k vectors: 30k train + 1k held-out. Text 50% Ultra-FineWeb, 50% WildChat-1M (match the model card). If WildChat
  is gated for us, use UltraChat and say so. 3 positions per doc (more docs, more spread than the paper's 10).
  Use the repo's own `nla/datagen/stage0_extract.py` logic so the vectors match exactly.
- Teacher writes one explanation per train vector at temperature 1.0, max 200 new tokens, and returns the top-20
  log-probabilities at every output token (for soft-label distillation).
- Sizes: vectors 31k x 3584 x bf16 = 0.22 GB. Text ~4.5M tokens. Top-20 probs ~0.5 GB. Keep on the Lambda
  persistent disk `claude-lens-results-20260929` (us-east-1). Too small for cheaper storage to matter.
- Test sets (held-out only): 1k plain vectors; ~300 concept-injection cases (30 concepts x 10 base vectors; concept
  direction = mean L20 vector on texts about the concept minus mean on neutral texts, added at 3 strengths, strength
  picked on the teacher); our synthetic fact notes with made-up names (coverage v3/v4 style).

## Students and training

- Qwen2.5-0.5B-Instruct (d 896) and Qwen2.5-1.5B-Instruct (d 1536). Same tokenizer as the teacher, so token-level
  soft labels line up. Teacher data is shared, so the second student costs only its training.
- Same prompt as the teacher. The vector enters through an adapter: LayerNorm -> linear 3584 -> 4 x d_student,
  filling 4 input slots in place of the single `㈎` token (4 slots so 3584 numbers are not squeezed into 896).
- Step A: train the adapter only, student frozen, 1 pass (like LLaVA's adapter stage).
- Step B: unfreeze all; loss = cross-entropy on the teacher's text + KL to the teacher's top-20 distribution
  (renormalised), 2 passes. Save checkpoints after 3k, 7.5k, 15k and 30k examples seen, as a cheap data curve (first
  pass only, so "examples seen once", not separate dataset sizes; say so when reporting).
- Not in the first pass: on-policy distillation (teacher scores student samples), GRPO against the AR, the
  best-of-N loop.

## Measures (all on held-out data)

1. FVE of the explanation under the released AR (teacher, each student, each checkpoint).
2. Controls: (a) the student given another example's vector (shuffled), the floor for "reads the vector";
   (b) text-only: Qwen2.5-1.5B-Instruct summarises the actual text in the teacher's style, scored by the AR.
3. Concept injection hit rate (does the explanation name the added concept?), teacher first.
4. Truth rate: split 200 explanations per method into claims; blind Claude judges check each against the source
   text (subagents, $0).
5. Agreement with the teacher: topic match and claim overlap, judged blind.

Report "share of the teacher's gain kept" = (student - shuffled) / (teacher - shuffled), per measure.

Bars:
- Direction found: student FVE and injection hits clearly above the shuffled control (bootstrap 95% CI excludes it).
- Good first pass: 1.5B keeps >= 50% of the teacher's FVE gain and >= 50% of its injection hits.
- Full distillation (not expected here): >= 80% of the teacher's gain on FVE and injection, truth rate within
  10 points of the teacher.

## Pre-flight checks (stop if any fails)

1. Teacher FVE on our 1k held-out vectors is near the card's 0.75 (say >= 0.55). If not, our vector extraction
   (layer index, scale, prompt) is wrong. Fix before spending more.
2. Teacher writing speed on 200 samples with SGLang. If it is under ~1,000 tokens/s, stop and re-plan (plain HF
   generation would blow the budget).
3. Linear probe (minutes, no LM): ridge regression from the 7B vector to a sentence embedding of the teacher's
   explanation. Held-out R^2 near zero would mean the adapter route is in trouble.

## Cost (A100 40GB, $1.99/h; estimates, nothing benchmarked)

| Step | Time | Cost |
|---|---|---|
| Box setup, SGLang install, model downloads (~45 GB) | ~45 min | ~$1.50 |
| Extract 31k vectors | ~10 min | ~$0.30 |
| Pre-flight: teacher FVE, speed test, probe | ~20 min | ~$0.70 |
| Teacher writes 30k explanations + top-20 probs (~4.5M tokens) | 25-50 min | $0.80-1.70 |
| Train 0.5B (A+B) | ~25 min | ~$0.80 |
| Train 1.5B (A+B) | ~50 min | ~$1.70 |
| Held-out generation (all methods, checkpoints), AR scoring, injection, text-only | ~40 min | ~$1.30 |
| **Total** | **~3.5-4.5 h** | **~$7-9** |

Budget: $20.38 of the $30 new budget spent (log, 2 Oct), so ~$9.60 left. This pass uses nearly all of it.
Hard box cap 5 h (failsafe on the box, see memory note on Mac sleep). Check the box every 5 min.

Full distillation, if the data curve says it is worth it (projection only): 250k-1M teacher pairs (~40-150M
tokens, ~$5-20 to write), training 1.5B for 2 passes (~$10-40), on-policy rounds (~$10): about $25-70.

## Run names

`nladistil_{method}_{student}_from-nla-qwen25-7b-L20_{data}_{date}[_ep{n}|_seen{k}]`, e.g.
`nladistil_sftkd_qwen25-1.5b_from-nla-qwen25-7b-L20_pairs30k_20261003_seen15k`.
Data: `data/nladistil_qwen25-7b-L20_ufw-wildchat_31k_20261003/`.

## Later steps (only if the first pass finds a direction)

1. More pairs, sized from the data curve; add on-policy distillation.
2. The best-of-N / loop idea: pick by agreement across samples and neighbouring tokens plus a truth filter, AR
   score only as a floor; control = GRPO at the same compute.
3. Bigger gap: Gemma-3-27B NLA -> Gemma-3-1B (teacher at 8-bit on one A100), then Llama-3.3-70B NLA -> Llama-3.2-1B/3B
   (needs more than one A100).
