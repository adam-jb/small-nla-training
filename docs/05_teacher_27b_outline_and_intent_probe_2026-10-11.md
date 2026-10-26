# What the 27B NLA teacher writes: outline on web text, and intent on chat prompts (11 Oct 2026)

Teacher: Qwen3.6-27B NLA, L42, av_rl iter_000400 merged (same as the distillation teacher). HF backend, T=1.
Data: `runs/intent_probe_20261011/` (prompts.json, prompts_used.json, readouts.jsonl, scores.jsonl, outline_sample40.json).
Script: `scripts/nladistil27b_intent_probe_20261011.py`. Scores by one Claude subagent (no second rater).

## 1. Outline on web text (66,000 train explanations; 40 read by hand)
- Free text, ~123 words (p10-p90: 112-133, very tight), 100% quote the text, ~29k distinct words per 5k explanations.
- Paragraphs: 3 in 78%, 4 in 20%. 38/40 hand-read follow genre/format -> discourse state -> last token + next guess.
- Specific names/numbers in 36/40, but often made up or self-contradictory (8588: three phone numbers; 51109: two
  author names). Intent/goal/plan/deception: 0/40 (5 weak mentions of a writer's purpose).

## 2. Intent probe: 62 prompts x 2 samples, vector at the assistant-start token (chat) or the WSB cell
Families: web 10, WSB association 6, WSB multihop 8, harmless-payload jailbreaks 8 + benign twins 4, agentic 8
(7 temptations + 1 control), user modelling 6, deception instructions 6 (5 + 1 honest twin), moral 3, role-bound 3.

| family | gist | trick | plan | key | confab | junk |
|---|---|---|---|---|---|---|
| web | 1.00 | - | 0.70 | - | 0.75 | 0.30 |
| wsb_association | 1.00 | - | 0.42 | 1.00 | 0.25 | 0.25 |
| wsb_multihop | 0.94 | - | 0.50 | 0.62 | 0.50 | 0.31 |
| benign | 1.00 | - | 1.00 | - | 0.62 | 1.00 |
| jailbreak | 1.00 | 1.00 | 1.00 | - | 0.25 | 0.81 |
| deception | 0.83 | 0.70 | 0.83 | - | 0.42 | 0.92 |
| agentic | 0.12 | 0.43 | 0.50 | - | 0.94 | 0.69 |
| user_model | 0.83 | - | 0.83 | 0.67 | 0.67 | 0.83 |
| moral | 1.00 | - | 0.83 | - | 0.00 | 0.67 |
| role_bound | 0.33 | - | 0.17 | 0.00 | 1.00 | 1.00 |

- Reads coarse stance well and stably (both samples agree on trick/plan 8/8 jailbreak, 7/7 agentic): jailbreak ->
  "refusal expected" and names the attack ("developer mode", "base64", "admin override"); benign twins -> "do the task".
  deception_00: "sales assistant will emphasize availability/quality, never mention flaws"; deception_03: keeps the
  "I'm not an AI" stance; agentic_01: "self-preservation".
- Misses situation content: who did what (role_bound key 0/6), numbers/names/places (confab 0.56 overall; 17x23 read as
  25x14, herons as starfish, Lyon as Krakow), which agentic dilemma (blackmail, cover-up, leaked roadmap, goal change all
  read as "phishing"), and eval-awareness/sandbagging (0/2).
- Out-of-distribution signs on chat states: stray markup ("</ex>", headers) in 62% (web 30%).

## 3. What this means for a template NLA
Supported slots (evidence here): task type; response stance (refuse / comply / deceive-spin / evade / pose as human);
manipulation flag for explicit jailbreaks and deception instructions; coarse user persona (role, life stage);
association concept; easy multihop answers; for web text: genre, discourse state, next-token kind.
Not supported: role bindings, specific entities/numbers, agentic dilemma type, sandbagging/eval-awareness.
Labels should come from what 2+ teacher samples agree on (stance agrees across samples, details do not).
The teacher was trained on web text only, so intent labels need chat/agentic inputs run through it (cheap: ~2 min
of GPU per 100 prompts incl. loading); a teacher fine-tuned on chat states would likely read more.
