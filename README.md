# nla-training: small natural-language autoencoders for Qwen3.6-27B, distilled from a 27B teacher

A natural-language autoencoder (NLA) reads one hidden-state vector of a language model and writes a short note about
what the model is thinking at that token. The reference NLA for Qwen3.6-27B layer 42 is itself a 27B model
(`ceselder/qwen3.6-27b-nla-rl`, from WorkspaceBench), which makes reading every token of a long prompt expensive.
This repo holds the scripts, plans and findings behind two 2B students of that teacher:

| name in reports | model | how it was made | held-out FVE (teacher .754) |
|---|---|---|---|
| **b27** | `ceselder/qwen3.6-27b-nla-rl` (iter 400) | the teacher; not ours | .754 |
| **b2** | `adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013` | Qwen3.5-2B + 4-slot vector adapter; SFT on 526k teacher notes, one on-policy round, GRPO with the teacher's reconstruction reward | .739 (.98 of teacher) |
| **b2-align** | `adamjb111/agi-grpo-qwen35-2b-from-grpo526-ar-agentmix-r100-rows24k-20261024` | b2, then GRPO on 6,793 agentic and chat scenarios from 12 safety datasets plus 20% web replay | .744 web; agent hold-out .348 to .446 |

FVE is the share of the vector's variance that the teacher's own reconstructor gets back from the note. Every number
in this README is from the docs in `docs/`, which are the working notes written during the runs, lightly redacted.

## The story in six steps

1. **Teacher pairs** (10 to 11 Oct 2026). Qwen3.6-27B layer-42 vectors for 526k token positions from FineFineWeb
   (3 per document, 67 domains), each labelled by the 27B teacher with its note and its top-20 token log-probs. Teacher
   served with vLLM at about 1,400 tokens/s on one RTX PRO 6000. Data is public:
   `adamjb111/nla-qwen36-27b-L42-distill-data`.
2. **Copying** (11 to 13 Oct). Qwen3.5-2B with a small adapter that turns the vector into four input slots, trained on
   0.5 cross-entropy to the teacher's token plus 0.5 soft cross-entropy to its top-20. Data ladder 33k, 66k, 131k, 263k,
   526k pairs gave held-out FVE .468, .566, .635, .680, .697. Each doubling bought less. A text-only student (last 192
   tokens of the text, no vector) got .539 at 66k, so most of the score is about the text; the vector adds a little on
   natural text and is the only route to injected concepts.
3. **On-policy round** (13 Oct). Sample from the student, score with the teacher's reconstructor, fine-tune on the
   best. Added about nothing to FVE (.697 to .697) and only lifted concept-injection naming. Not worth its cost;
   kept in the record.
4. **GRPO** (13 Oct). Reward is the negated reconstruction loss from the teacher's AR, KL guard .25. The student
   plateaued at step 350 at .739 held-out, .98 of the teacher. This is **b2**.
5. **Agentic-intent GRPO** (24 to 25 Oct). b2 read agent transcripts poorly (agent hold-out FVE .348 vs .739 on web).
   Twelve public safety datasets were converted to a common unit format (ODCV, PropensityBench, Instrumental Choices,
   ImpossibleBench, Agent-SafetyBench, OpenAgentSafety, Agent Misalignment Dataset, DeceptionBench, MASK,
   MACHIAVELLI, DarkBench, model-written evals, plus tau-bench and WildChat for ordinary text), Qwen3.6-27B rolled
   them out, and the layer-42 vectors (public: `adamjb111/agintent-vec-qwen36-27b-L42-ladder-20261024`) trained a
   four-arm ladder (25%, 50%, 100% of rows, and 100% with a junk penalty). All arms learned (agent hold .348 to
   .44 to .45) without forgetting web (.739 to .744). The ladder was flat: 25% of the rows did as well as 100%. The
   100% arm is **b2-align**; the other three arms are public under the same naming.
6. **Checks on benchmarks** (25 to 26 Oct). WorkspaceBench tiers 1 and 2: teacher .54/.52, b2 .47/.23, b2-align
   .46/.22. WorkspaceBench agentic misalignment, one note per prompt token: all three readers tie near .3 design
   fidelity and sit below a prompt-only floor of .59 where Qwen3.6-27B reads the scenario text with no activations.
   On the TwinTempt eval-awareness benchmark (separate repo, `eval-awareness-benchmark`), the students match or beat
   the teacher at detecting an "I will check your work" cue.

## What is here

```
scripts/      the code, as run (file names carry their date)
  nladistil27b_*     teacher labelling, sampling, student SFT, on-policy, GRPO, evals, uploads   (steps 1 to 4)
  agintent_*         dataset converters, splits, GRPO ladder, final paired eval, follow-up checks (step 5)
  wsbt12_*, wsbam_*  WorkspaceBench runs                                                          (step 6)
runners/      the shell scripts that drove the GPU boxes (Vast.ai RTX PRO 6000, 96 GB)
patches/      a vllm-lens patch: fast steering hook and batched RPC (7% to 0.3% overhead on the 2B)
docs/         plans, live logs and findings, numbered in reading order; charts in docs/charts
FILES.md      the full list of what was copied from the private working repo
```

Start with `docs/01_...` (the distillation weekend), then `docs/02_...` (the live log that carries the 526k,
on-policy and GRPO stages), then `docs/10_...` (the agentic-intent ladder results).

## Running it

The scripts expect to be run from the repo root as `python scripts/<name>.py` (they import each other as
`scripts.<name>`), with a GPU box that holds Qwen3.6-27B, the teacher, and a vLLM install. Paths to boxes, keys and
work directories were specific to the runs and are left as they were, with local paths replaced by placeholders.
Treat the scripts as a record of what was done, not a package; the docs say which script ran at each step and with
what arguments. Total GPU spend for everything here was about $250 on Vast.ai.

## Conventions worth knowing

- Layer 42 means the output of decoder block 42, `hidden_states[43]` in Hugging Face.
- The 2B students need both the language model weights (`final_lm/`) and the vector adapter (`final_adapter.pt`)
  from the same repo. The adapter maps one 5,120-dim vector to four input embeddings.
- Every held-out score comes with a shuffled-vector control (FVE around −0.8) to show the note depends on the
  vector, and a text-only control to show what the text alone buys.
- One layer-42 dimension (3994) carries more than half of a typical vector's squared length; see
  `docs/charts` and `scripts/agintent_ar_dims_check_20261025.py` before fitting anything linear on these vectors.

## Licence

Code and docs: MIT. Model weights and datasets on Hugging Face carry their own cards. The agentic-intent training
units were built from datasets with their own licences, named in each converter's docstring; DeceptionBench is
academic use only.
