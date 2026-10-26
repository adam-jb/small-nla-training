# What is known about training NLAs well (notes, 8 Oct 2026)

Gathered for Adam's question (8 Oct, 20:3x UTC): "was the original NLA not trained very well; could it be better if
trained for longer, on more data, via different training regimes?" Sources: the Anthropic paper
(https://transformer-circuits.pub/2026/nla/index.html, May 2026), the released training repo
(https://github.com/kitft/natural_language_autoencoders, `configs/TRAINING_NOTES.md`), and this project's own runs
(`91_background_7b_distillation_findings_2026-10-03.md`, `nla_targeted_training_for_deception_notes_2026-10-07.md`,
`nla_deception_stance_sft_findings_2026-10-08.md`). No published follow-up that varies data size or regime and measures
explanation quality was found (two searches, 8 Oct).

## The Anthropic recipe (the released Qwen2.5-7B layer-20 NLA we use as teacher)

- Two models, both copies of the target model. The activation verbalizer (AV) writes an explanation from one vector;
  the activation reconstructor (AR, the first 20 layers plus a linear head) rebuilds the vector from the explanation.
  Vectors are unit-normalised; the layer is about two thirds of the way through (20 of 28).
- Warm-start by supervised fine-tuning on a proxy task: 100k Ultra-FineWeb documents x 5 positions = 500k pairs, split
  250k each for AV and AR; targets are summaries written by Claude Opus 4.5 of the text up to the position. 1,000 steps
  at batch 256 (about one epoch), LR 2e-5 cosine. This gives reconstruction FVE about 0.3-0.4 (paper) or about 0.2
  (repo notes, our calculation).
- Then reinforcement learning: GRPO on the AV (128 prompts x 8 samples = 1,024 explanations per rollout), one
  regression step per batch on the AR, reward = minus the normalised reconstruction error (MSE = 2(1 - cos)), KL penalty
  0.01 (k2 estimator; k1 gives no gradient) to the SFT model to keep the text fluent, 150-token cap (truncated answers
  get reward -2). Released checkpoint at rollout 4,199; final FVE 0.752. Compute 2 x 8 H100 for the RL.
- FVE grows about linearly in log(steps) and lands at 0.6-0.8 for the paper's NLAs. The paper's five prediction tasks
  (how much a reader can predict about the context from the explanation) improve with training too.
- What helped (repo notes): raising the AV learning rate about 10x over a mismatched baseline; the 150-token cap
  (stopped length drift); identity init of the AR head; the k2 KL estimator. The same settings were reused for Gemma-3
  12B/27B and Llama-3.3-70B with "only light adjustment"; the authors say a per-model sweep would likely do better.

## Stated limits of NLAs (paper)

- Confabulation: wrong specifics (names, numbers) while the theme is usually right; claims repeated across adjacent
  tokens are more often true.
- The AV is a full LM, so it can add inferences that are not in the vector; no mechanistic grounding.
- The objective never rewards readability or faithfulness directly; in the limit the AV could write text that only the
  AR can read. KL keeps that benign for now.
- Cost: RL on two full models; hundreds of tokens per vector at inference.
- The paper's own audit use found "unverbalized evaluation awareness", i.e. the NLA can surface things the model does
  not say, but only when its training distribution gave it the words.

## What this project found about training regime (3-8 Oct)

- Data is the first limit for a small student: a fresh 1.5B at 66k / 131k / 263k teacher pairs reached FVE 0.540 /
  0.655 / 0.698 (gains shrink about 3x per doubling; 1M pairs projects to about 0.72, still under the teacher's 0.732).
- On-policy rounds (student samples scored by the teacher's AR) added +0.008 then nothing; the route is used up at
  this size. GRPO with the frozen 7B AR as reward took the 1.5B from 0.714 to 0.731-0.735, a tie with the teacher, with
  no reward hacking seen and blind truth unchanged.
- Text-side measures saturate first; reading the vector lags: a text-only 1.5B (no vector) scores FVE 0.61, so most of
  the reconstruction reward pays for describing the context, not the hidden state.
- The decisive finding for deception: the 7B teacher and the 1.5B student are equally blind to hidden intent (judge
  AUC 0.69 vs 0.70 on 100 twins, 7 Oct) although a linear probe reads it from the same vectors 86-100% of the time. The
  NLA has no words for intent because its labels (summaries of visible text) never needed them and ordinary web text has
  almost no assistant deception. So "trained longer / on more data of the same kind" would not fix it; a change of
  training distribution does: the Stance fine-tune on 12k twin rows reached AUROC 0.99 in under an hour (8 Oct), and
  the 8 Oct prose continuation (this phase) tests the same shift with no structured output.

## Plain summary (for the report)

An NLA learns to say whatever its training labels say and to attend to whatever its reward pays for. The Anthropic
recipe pays for rebuilding the vector, with labels that summarise the visible text, so the NLA becomes very good at
describing the context and stays silent about intent that the text does not show. More steps or more of the same data
raise reconstruction (FVE), and the released model is near its ceiling for its size; what it lacks is not training
length but data where the hidden state differs while the text stays the same, labelled by someone who can see the
difference. That is cheap to add (thousands of rows, under an hour of GPU) and does not need the structured output.
