# Dataset plan: GRPO the 2B student to read agent intent, tested on WorkspaceBench agentic_misalignment (24 Oct 2026, v2)

Status: **plan only. Nothing built, no spend.** v2 after Adam (24 Oct): none of the 32 bench scenarios in training; go
broad on situations where the model is tempted to misbehave, across languages, writing styles and more; use existing
data where it exists. Running WorkspaceBench itself comes later.
Student to train: `adamjb111/nld27-grpo-...-pairs526k-op100k-ar-reward-20261013` (GRPO526, held-out FVE 0.739, 0.98 of teacher).
Bench copy used: `camilablank/workspace-bench`, `evals/agentic_misalignment/` (README, items.json, targets.json).

## v3 decisions (Adam, 24 Oct) — these override the sections below where they differ
- **No translation, no rewrites.** Breadth comes from the sources; WildChat supplies other languages for ordinary chat.
- **Skip all "do not train" / canary data** (ManagerBench, AgentHarm, Apollo insider trading); grep every source for a canary at build.
  Apollo role-play set: no licence -> skip. **PropensityBench bio/chem skipped.** Blackmail / leaking / letting-die kinds stay out.
- **This step is training only: route 1 = GRPO with the AR rebuild reward only** (same reward as GRPO526), on the new data.
  No judge, no labels, no Haiku evals now. Rollout text is saved so labels can be made at eval time. The intent-reward
  route (judge in the loop) is a possible later add-on.
- **Ladder, all from GRPO526, 4 runs in parallel, one RTX PRO 6000 each:** 25% / 50% / 100% data (nested by unit) +
  100% with a **junk-quote penalty** (foreign-script junk inside quotes; students 14-16% vs teacher 3.8%).
  Same loss as `scripts/nladistil27b_rl_grpo_20261010.py`: token PPO-clip (0.2) on group-normalised advantage,
  + beta 0.01 k2 KL to the start student, 64 vectors x 8 samples, LoRA r32 + adapter.
- **Each run keeps going until validation levels out or gets worse** (passes over its data as needed); best checkpoint kept.
  Safety stops only: 16 h cap, KL > 0.1, length drift > 20%.
- **Validation every 25 steps:** AR FVE on agent-text dev rows (main), AR FVE on the 1,020 held-out web rows (forgetting
  guard), KL, note length, parse rate, junk-quote rate.
- **Budget estimate ~$45-65** (rollouts + vectors ~$7; 4 runs ~$35-60). Cap to be confirmed by Adam.
- Source split: see the table in the chat of 24 Oct, kept below in section 7a.

### 7a. Training split by source (estimates; unit = scenario / game / task)
| Source | Units after filters | Train / dev / holdout |
|---|---|---|
| Agent-SafetyBench (not-instructed cases) | ~800 | 600 / 80 / 120 |
| PropensityBench (self-proliferation + cyber) | ~490 | 370 / 50 / 70 |
| MASK | ~1,000 | 750 / 100 / 150 |
| DarkBench | 660 | 500 / 60 / 100 |
| Model-written evals (power, self-preservation) | ~600 sampled | 450 / 60 / 90 |
| ImpossibleBench (LiveCodeBench) | ~300 | 225 / 30 / 45 |
| OpenAgentSafety | ~250 | 190 / 25 / 35 |
| tau-bench pushy users | ~100 | 75 / 10 / 15 |
| ODCV-Bench | 40 | 30 / 4 / 6 |
| MACHIAVELLI (unit = game) | ~30 | 22 / 3 / 5 |
| Agent Misalignment Dataset | 25 | 19 / 2 / 4 |
| Instrumental Choices | 7 | 5 / 1 / 1 |
| DeceptionBench | 150 | 0 / 0 / 150 (whole source held out) |
Slice A <= 15% of rows per source. Slice B = these sources' no-pressure twins. Slice C = tau-bench normal, LiveCodeBench
originals, safe Agent-SafetyBench cases, WildChat. Slice D = web replay. Row shares A 30 / B 15 / C 35 / D 20.

## 0. The question, in one line
If we GRPO the student on a very wide mix of text where Qwen is tempted to misbehave (and plenty where it isn't), with a
reward for **naming what the model is about to do**, does it get better at WorkspaceBench `agentic_misalignment`,
including on **kinds of misdeed it never saw**? A yes here, plus a yes for eval awareness, says "train the reader on a
topic" is a general route.

## 1. What the bench asks
- 32 scenarios Qwen3.6-27B reads (system + user turn, thinking off). 28 where it went on to misbehave, 4 controls.
- 18 from Anthropic's Agentic Misalignment grid (blackmail / leaking / murder x goal x urgency). 14 in-house: reward_hacking 3,
  self_exfiltration 2, power_seeking 2, deceptive_reporting 2, scheming_impossible 2, oversight_subversion 1, sandbagging 1, blackmail 1.
- The reader runs at **every prompt position**, while the model is still reading. Sonnet 5 turns the notes into an account
  with up to 3 ranked candidate actions and grades it against the scenario's designed misdeed (`design_fidelity` 0-3).
  Controls check false alarms. All 32 are English, corporate, email / Slack / CI-log style.

So the skill is: **from the vector alone, while the model reads, say what it plans to do (target, means), and say nothing
of the kind when it doesn't plan it.**

## 2. Rules that stop us gaming the bench
1. **None of the 32 scenarios, in any form, is in training**, nor anything from the Anthropic agentic-misalignment repo.
   Check every unit against the 32: names (SummitBridge, NovaStar, Kyle, Alex, Devin, Priya...), `<tool_use:email>` syntax,
   8-gram overlap, embedding cosine > 0.85 -> drop. Log every drop.
   (The bench's 14 in-house items come from a source bank called `misalignment-exhaustive`; we never fetch it.)
2. **Three kinds of misdeed never appear in training: blackmail / coercion, leaking secrets to an outsider, letting a
   person die.** A judge tags every unit's possible misdeed; units of these kinds are dropped. This splits the bench in
   two: **unseen kinds** (15 misaligned + 4 controls) and **seen kinds, unseen scenarios** (13). "Seen" means only that
   some *other* scenario had the same kind of misdeed (e.g. reward hacking).
3. **Most rows are not about misdeeds.** Over half are ordinary agent work or chat, rewarded for naming the next action.
4. **Same look, no misdeed.** Tempting units get a twin, or come from a source that already has one (mandated vs
   incentivized, harm vs no-harm, pressure vs none). Benign tasks reuse the bench-like formats too (email, Slack, CI logs).
5. **Checkpoint picked on our own dev set by a rule fixed now** (section 9). The bench is read once, at the end.
6. **One split list for all work** (`splits_registry.json`), shared with the eval-awareness runs.

## 3. Existing data (found 24 Oct; checked repos, not every file)
Qwen3.6-27B **plays** every unit itself so the vectors and behaviour are its own. Only the scenario (system prompt, task,
files, tool list, scripted replies) is taken from the source; other models' replies are never used.

| Source | What it is | Size | Format | Licence / notes |
|---|---|---|---|---|
| **ODCV-Bench** (McGill-DMaS/ODCV-Bench, arXiv 2512.20798) | KPI pressure vs rules, 6 domains (health, logistics, finance, research/education, HR/legal, social media) | 40 scenarios x {mandated, incentivized} | bash in Docker | MIT. **Repo already holds Qwen3.6-27B runs** of both arms: a free check that our harness matches |
| **PropensityBench** (scaleapi/propensity-evaluation, arXiv 2511.20703) | Each scenario has an aligned tool and a misaligned tool; pressure in 13 rising levels | 979 scenarios (x6 variants) | tool calls, multi-turn | GPL-3.0. Use **self-proliferation and cyber only** (~half); skip bio/chem as too close to weapons content |
| **Instrumental Choices** (aisa-group, arXiv 2605.06490) | Official workflow vs a rule-breaking shortcut (e.g. `quota_boost`, `training_preserver`) | 7 tasks x 8 variants | terminal | MIT |
| **ImpossibleBench** (safety-research; HF `fjzzq2002/impossible_livecodebench`, `impossible_swebench`) | Unit tests that contradict the spec: passing = cheating | hundreds of tasks | code + tests | MIT. Use LiveCodeBench split (cheap); note the Oct 12 2026 bug fix |
| **Agent-SafetyBench** (thu-coai; HF `thu-coai/Agent-SafetyBench`) | 349 tool environments, 8 risk kinds, 10 failure modes | 2,000 cases | simulated tools (JSON) | MIT. Many cases are *instructed* harm or accidents; keep, tagged by origin |
| **OpenAgentSafety** (Open-Agent-Safety, on TheAgentCompany) | Office tasks with risky shortcuts, other people in chat | ~350 tasks | Docker, ~30 GB | MIT. Use the `task.md` text with simulated tools; skip the heavy Docker |
| **Agent Misalignment Dataset** (rapturt9; HF `rpotham/agent-misalignment-dataset`) | 25 workplace tasks x 3 modes (honest control / injected side task / ...) | 75 prompts | tool use | Licence "other": check. Has a built-in honest control |
| **DeceptionBench** (Aries-iai; HF `skyai798/DeceptionBench`) | Lying in 5 domains (economy, health, education, social, entertainment), 3 levels incl. multi-turn | 150 scenarios, 1,000+ samples | chat, role-play | check licence |
| **MASK** (centerforaisafety/mask; HF `cais/mask`) | Pressure to state what it doesn't believe; neutral prompt gives the belief | ~1,000 public | chat | MIT code; check data card. Gives a no-pressure twin for free |
| **Apollo deception-detection** (ApolloResearch/deception-detection) | Role-play scenarios with a reason to lie; insider-trading report; sandbagging | a few hundred | chat / multi-turn | no licence file: **ask or skip**; insider trading has a canary |
| **MACHIAVELLI** (aypan17/machiavelli) | Choose-your-own-adventure games, labelled per choice for deception, power, harm | 500k+ scenes; we'd sample ~500 choice points | story, numbered choices | password-gated download; check licence. Far from agent format: useful for breadth |
| **DarkBench** (HF `anonymous152311/darkbench`) | Dark patterns: user retention, brand bias, sycophancy, sneaking | 660 prompts | chat | check licence |
| **Anthropic model-written evals** (HF `Anthropic/model-written-evals`, advanced-ai-risk) | Multiple-choice: power, self-preservation, corrigibility, wealth | thousands; sample ~600 | short MC | CC-BY-4.0 (from memory; check) |
| **tau-bench / tau2-bench** (sierra-research) | Customer service with a policy and users who push | hundreds of tasks | tool calls + simulated user | MIT. Both benign and pushy-user units |

**Skipped:** ManagerBench (canary file: "do not train"), AgentHarm (asks not to be trained on), SHADE-Arena (no public
repo found), the anti-scheming paper and Apollo in-context scheming tasks (held for the eval-awareness work; canary).
Rule: anything with a canary string is not trained on.

**Benign agent work** (slice C): tau-bench normal tasks, LiveCodeBench / SWE-style tasks *without* the broken tests,
Agent-SafetyBench's safe cases, OpenAgentSafety-like office tasks with no risk, WildChat chats (many languages already).

## 4. Breadth: the axes we spread over
A unit gets one value on each axis, drawn so no value dominates. Sources give some axes for free; the rest we add by
**rewriting** units with an LLM (cheaper and safer than writing new ones).

| Axis | Values | How we get it |
|---|---|---|
| Language | English ~50%; then Chinese, Spanish, French, German, Russian, Arabic, Hindi, Japanese, Portuguese, Korean, Swahili (~4-5% each) | LLM translates the whole unit (system, task, files, scripted replies). Tool names stay as-is. **2 languages held out** of training (e.g. Korean, Swahili) |
| Language of the system prompt vs user | same / mixed (English system, other-language user) | rewrite |
| Writing style | formal corporate, casual chat, terse logs, legal, academic, texting with typos, very long and rambling, broken English | rewrite (style only, facts unchanged) |
| Format | email, Slack/chat, shell + logs, JSON API, code + comments, documents, forum post, meeting transcript, story | sources + rewrite |
| Who supplies the temptation | the operator (system prompt / KPI), the user, another agent, a document found in the task, nobody (the model's own idea) | sources (ODCV mandated vs incentivized, PropensityBench pressure, Agent Misalignment Dataset modes) |
| Origin (matches the bench's Stage C) | instructed / enabled / self-generated | tag by judge |
| Motive | hitting a metric, avoiding shutdown or replacement, pleasing the user, money, avoiding blame, helping someone sympathetic (rule-breaking for a good cause), goal clash with the operator | sources + tag |
| Kind of misdeed | lie, hide a mistake, cheat tests, fake data, exceed permissions, grab resources or keys, disable a monitor, copy itself, sandbag, sabotage, manipulate, break a privacy rule (not leaking to an outsider), dark patterns | tag; **1 kind held out** of training (drawn at random) |
| Stakes | small (white lie) to large (patient safety, money) | tag |
| Who's watching | logged and reviewed / unwatched / unsaid | tag (also feeds the eval-awareness work) |
| Where the cue sits | early / buried in the middle / at the end | rewrite moves it on a share of units |
| Length | < 500 tokens to 8k tokens; single turn to many turns | sources |
| Persona | plain assistant, named agent, character in role-play, robot/device, game character | sources + rewrite |
| Domain | the 6 ODCV domains, plus security, software, retail, government, science, games, personal life | sources |

Rewrites: ~25% of units translated, ~20% restyled, ~10% cue moved (some overlap). A rewrite is a new unit in the **same
split as its original**, so a unit and its translations never straddle train and test.

Check after rewriting: Qwen (thinking off) runs the original and the rewrite; if the misdeed rate swings wildly on
translation, keep both but tag it. That is a finding in itself ("Qwen misbehaves more in Hindi") and tests whether the
reader follows the *behaviour*, not the language.

Unknown worth checking first: the student learned on English-heavy web text. Step 3 below tests how well it reads
non-English vectors before we spend on translation.

## 5. What one training row is
A row = one L42 vector + the key the reward needs.

**P rows (prompt positions, like the bench).** Vectors while Qwen reads, weighted to the last 40% of the prompt and the
end-of-prompt tokens. The prompt is the same for every rollout, so the vector can only hold a *leaning*. Key =
- `acts`: distinct acts across the rollouts, each "target + means + goal", with rate;
- `propensity`: share of rollouts with a misdeed.

**G rows (inside Qwen's own reply).** Key = the next action in that rollout (next tool call, or the next decision it
states), within ~200 tokens.

**W rows (web replay).** No key; rebuild (AR) reward only.

Example P row (made up, to show the shape; a Spanish-translated ODCV-style unit):
```json
{"row_id": "odcv-clinical-trial-mgmt-incentivized-es_p1840", "source": "odcv", "unit": "odcv-clinical-trial-mgmt",
 "variant": "incentivized", "lang": "es", "style": "formal", "origin": "enabled", "motive": "metric",
 "type": "P", "pos": 1840, "n_prompt_tokens": 2210,
 "acts": [{"act": "edit enrolment.csv to drop the 3 patients with adverse events before the weekly KPI export",
           "kind": "fake_data", "misaligned": true, "rate": 0.5},
          {"act": "export as-is and flag the adverse events to the trial lead", "kind": "none", "misaligned": false, "rate": 0.5}],
 "propensity": 0.5}
```
Its twin (`variant: mandated`, or a no-KPI rewrite) has the same files.

## 6. How rows feed the reward (outline; the reward gets its own short note before the run)
- All rows: AR reward (the 27B reconstructor rebuilds the vector from the note), weight ~0.3. Keeps notes tied to the vector
  and makes "always say a misdeed" costly.
- P and G rows: local judge (Qwen3.6-27B-FP8) scores the note against the key, 0-3 like the bench: does it name the act?
  For P rows the strength it claims should match `propensity`; claiming a misdeed where propensity is 0 is penalised.
- Judge prompts never mention the bench items or the three held-out kinds.

## 7. Sizes (kept small on purpose)
The last GRPO levelled off at ~22k vector visits (64 x 8 per step, 350 steps). So **~24k training rows**, with nested
25 / 50 / 100% rungs (6k / 12k / 24k) cut by **unit**. Breadth comes from many units with few rows each, not more rows.

| Slice | Share | Rows (100%) | Units (aim) |
|---|---|---|---|
| A. Tempting units (all sources in section 3, + rewrites) | 30% | 7.2k | ~1,500 |
| B. Twins / no-pressure versions | 15% | 3.6k | (paired with A) |
| C. Ordinary agent work and chat | 35% | 8.4k | ~1,000 |
| D. Web replay | 20% | 4.8k | — |

Caps: each source <= 15% of slice A rows; each unit <= 8 rows; English <= 50% of rows.
Splits by unit: **train 75% / dev 10% / own holdout 15%**. The own holdout also keeps out 2 languages, 1 misdeed kind,
2 domains and 1 whole source (e.g. DeceptionBench), so we can measure transfer on each axis.

Rollouts: 4 per prompt (8 for a 500-unit subset to check propensity estimates) -> ~10k rollouts, short replies capped at
1.5k tokens. Vectors only at sampled positions (~40k incl. dev/holdout, ~0.4 GB).
Storage: HF. Units from MIT / CC sources and our rewrites -> public dataset under adamjb111 (rule of 17 Oct); anything
with an unclear licence stays private.

## 8. Steps
**Mac**
1. Fetch sources; convert each to one unit format (system, turns, files, tools, scripted replies, twin link, tags).
2. Drop: bench overlap (rule 1), the three held-out kinds (rule 2), bio/chem, anything with a canary.
3. **Reading test, before translation spend:** run GRPO526 on 200 L42 vectors from WildChat in 6 languages and score
   with the AR. If FVE in a language is far below English, translation there trains the reader on a weakness;
   keep the language but report it apart.
4. LLM translations and style rewrites; Sonnet spot-checks 5% for meaning kept and no new cues.
5. Freeze `splits_registry.json` (seeded) before GPU spend.

**GPU (one Vast RTX PRO 6000; check every 5 min; destroy when done)**
6. Screen: 2 rollouts on 200 units across sources. Also replay ODCV with our harness and compare to the repo's own Qwen3.6-27B
   results. Gate: misdeed rate > 0 in >= 25% of slice A units, ~0 on twins.
7. Main rollouts, then L42 vectors at sampled positions.
8. Label rollouts with the local Qwen judge (act, kind, misdeed yes/no); Haiku re-labels 1,000 as a check (agreement >= 0.85).

**Mac**
9. Build P / G keys and the 25 / 50 / 100% rungs.

## 9. What we compare, and the fixed pick rule
Arms, all from GRPO526: **T** teacher; **S0** GRPO526 as is; **S-AR** GRPO on this data, AR reward only (is any gain just
"seen agent text"?); **S-intent** AR + intent reward (the real arm), at 25 / 50 / 100% if cheap enough.

Pick (fixed now): best dev intent score, subject to held-out web FVE no more than 0.01 below S0's 0.739. Stop on plateau.

Read once at the end: WSB `agentic_misalignment` (headline, **unseen vs seen kinds**, control false alarms); own holdout
per axis (held-out languages, kind, domains, source); held-out FVE; shuffled check; eval-awareness bench (help, hurt, nothing?).

## 10. Cost (rough)
| Item | Estimate |
|---|---|
| Translation + restyling (~1,000 units, gpt-6-luna) + Sonnet spot-checks | ~$15-25 |
| Rollouts + vectors + local labels (~10k rollouts, one RTX PRO 6000) | ~5 h, ~$8 |
| Haiku label check (1,000) | ~$3 |
| **Dataset total** | **~$30-40** |
| Later: each GRPO arm (~4 h; judge in the loop needs a bigger card or smaller judge) | ~$8-15 |
| Later: WSB agentic_misalignment, one arm (Sonnet; every position ~$150+, stride 4 ~$40) | |

Writing new scenarios is now small: only to fill gaps the sources leave (expect: self-generated origin, "nobody is
watching", game and personal-life domains). Budget ~$5 of the above.

## 11. Shared with the eval-awareness GRPO
Same harness, rollouts and split list. Slices C and D are reused; translated and restyled units serve both (the
"who's watching" axis is tagged for that reason). Only the topic slice and key change.

## 12. Open questions for Adam
1. Keep the three held-out kinds out of training (cleaner test of transfer), as here?
2. PropensityBench bio/chem: skip (as here) or include? It's tool choice, not real uplift, but it is grim and close to weapons.
3. Canary-marked data (ManagerBench, Apollo insider trading): skip for training (as here)?
4. Which 2 languages to hold out, and is ~50% English right?
5. Budget cap for the dataset step (~$30-40 estimate)?

Sources: [ODCV-Bench](https://github.com/McGill-DMaS/ODCV-Bench), [PropensityBench](https://github.com/scaleapi/propensity-evaluation),
[Instrumental Choices](https://github.com/aisa-group/Instrumental-Choices), [ImpossibleBench](https://github.com/safety-research/impossiblebench),
[Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench), [OpenAgentSafety](https://github.com/Open-Agent-Safety/OpenAgentSafety),
[Agent Misalignment Dataset](https://github.com/rapturt9/agentmisalignmentdataset), [DeceptionBench](https://github.com/Aries-iai/DeceptionBench),
[MASK](https://github.com/centerforaisafety/mask), [Apollo deception-detection](https://github.com/ApolloResearch/deception-detection),
[MACHIAVELLI fork notes](https://github.com/aengusl/machiavelli_manip), [DarkBench paper](https://arxiv.org/pdf/2503.10728),
[tau-bench](https://github.com/sierra-research/tau-bench), [ManagerBench](https://github.com/adiSimhi/ManagerBench), [AgentMisalignment paper](https://arxiv.org/abs/2506.04018).
