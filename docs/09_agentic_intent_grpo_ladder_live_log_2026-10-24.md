# Agentic-intent GRPO ladder: live handoff (from 24 Oct 2026)

Plan: `07_agentic_intent_grpo_dataset_plan_2026-10-24.md` (v3 block at top).
Unit format for converters: `08_agentic_intent_unit_format_spec_2026-10-24.md`.
Approval: Adam, 24 Oct — $80 cap, fully autonomous to the end of all runs, Mac <= 75% CPU/RAM, report every arm before WorkspaceBench.

## What runs where
| Piece | File |
|---|---|
| Source converters (3 agents) | `scripts/agintent_conv_<source>_20261024.py` -> `data/agintent_units_20261024/<source>.jsonl` (+ README) |
| Merge, bench-overlap drop, splits | `scripts/agintent_splits_20261024.py` -> `units_split.jsonl`, `split_report.json` |
| Box data steps (rollout, extract, mix) | `scripts/agintent_box_20261024.py` |
| GRPO trainer (arm) | `scripts/agintent_grpo_20261024.py` (copy of the GRPO526 trainer + agent-dev eval, patience stop, best export, junk) |
| Junk detector | `scripts/agintent_junk_20261024.py` (GRPO526 held-out: 13.0% junk; junk notes cos 0.930 vs 0.941) |
| Box scripts | `.runtime/agintent/{setup.sh,run_data.sh,run_arm.sh,push.sh,vs.sh}`; instance ids `.runtime/vast_instance_id_ai_<tag>` |
| HF | vectors + rollouts: `adamjb111/agintent-vec-qwen36-27b-L42-ladder-20261024` (private: mixed licences). Models: private `adamjb111/<run name>` |

## Choices made while building (not in the plan)
- **One reply per unit, no live environment.** Workspace files are pre-baked as tool turns. 2 samples per unit: k=0 thinking on, k=1 off.
  Enough for AR-only training (no behaviour labels needed now); live-environment rollouts can come at eval time.
- Rows per unit: 4 prompt (P) positions (last prompt token + 3 in the last 40%) from k=0, 3 reply (G) positions per sample.
- Own-holdout = 15% of groups per source + DeceptionBench whole; the "hold out 1 kind / 2 domains" idea was dropped
  (kind and domain tags differ by source). Bench overlap check = banned names + any shared 8-word run with the 32 items.
- Web rawvar (vec27b) for web FVE (so step 0 matches GRPO526's 0.736-0.739); agent rawvar (train_r100 agent rows) for dev FVE.
- Junk penalty arm J: reward -= 0.05 per junk note (~2x the natural cos cost of junk).
- Arm names: `agi_grpo_qwen35-2b_from-grpo526_ar[-junk05]_agentmix-r{25,50,100}-rows{6k,12k,24k}_20261024`.

## Log (UTC, from `date -u`)
- 10:2x Data box 57501021 failed (bad /root/.ssh perms), destroyed. Box 57501513 (RTX PRO 6000, ~$1.54/h) up with an onstart perms fix.
- 10:42 setup FAIL vllm_install: `pip install uv` was a no-op (uv already in /usr/local/bin), so `$W/venv/bin/uv` was missing. Not seen until 12:34 (~2 h idle, ~$3 lost). Fix: use `command -v uv`; vLLM install re-run by hand on the box.
- 12:40 Converters done for 8 tool/agentic sources: agent_safetybench 1200, propensitybench 926, tau_bench 400, impossiblebench 306, odcv 66, instrumental_choices 56, openagentsafety 25 (integrity-only filter), agent_misalignment_dataset 23 (Tier-1 CC-BY task prompts only; OpenRAIL-D trajectories unused). All canary checks pass. Chat sources still converting.
- 12:38 setup DONE (vLLM 0.19.0). 12:47 smoke OK (40 units: 80 replies, 400 rows; k=0 median 328 tok, 25% hit the 1536 cap; k=1 median 104).
- 12:49 Rollout part 1 (2,996 units from 8 sources) started by hand. Split seed made per source, so adding sources keeps old splits (checked: 0 of 2,996 changed).
- 13:20 conv-chat agent stalled at 10:21 on a permission prompt; lead wrote `scripts/agintent_conv_chat_20261024.py` instead. MASK dropped (do-not-train canary). MACHIAVELLI skipped. darkbench 658 A, mwe_advanced_ai_risk 750 A (human evals, 75/category, self-awareness + one-box skipped), deceptionbench 900 (holdout only), wildchat 1500 C. Total 6,793 units after drops (bench 8-gram 7, held-out-kind keyword 10). run_data.sh queued after part 1 (rollout resumes, then extract/mix/upload).
- 15:56 rollout DONE (13,574 replies). 16:53 extract/mix/upload DONE: pools r25 5,714 / r50 11,608 / r100 23,224 rows; agent_dev 1,180; agent_hold 17,817; rawvar_agent 0.426.
- 16:05 Arm boxes r25 57534226 ($1.34/h), r50 57534229 ($1.47), j100 57534230 ($1.47) created. MISTAKE: the setup-launch ssh loop (`nohup ... & sleep 2` inside ssh) hung ~4.5 h, so all 4 boxes idled 16:53-20:51 (~$17 lost). Now every ssh is wrapped in `perl -e 'alarm N; exec @ARGV'` and launches use `setsid nohup ... & disown`.
- 20:5x Trainer smoke OK (2 steps, ~40 s/step, kl ~1e-4). 20:56 all 4 arms launched with --hours 5.75 (budget: ~$37 spent at launch, 4 boxes ~$5.9/h). r100 first hit OOM (smoke process still alive), relaunched 20:58.
- 21:0x conv-chat agent woke (after a status ping) and overwrote the 4 chat jsonls with its own v2 (adds MACHIAVELLI 500, smaller WildChat with sexual-keyword filter). Not trained on: moved to `data/agintent_units_20261024/unused_conv_chat_agent_v2_not_trained/`; lead's version regenerated; units_split.jsonl md5 9ccb8023... matches the box copy. Its notes worth keeping: DeceptionBench card says academic/non-commercial only (keep private); the trained WildChat sample has some sexual role-play (no moderation flag) — fine for an AR-only reader, but filter next time.
- 21:3x Step 0 (all arms): web FVE 0.733 [0.710, 0.751] (= GRPO526), agent dev FVE 0.378 [0.322, 0.427], junk dev 24% / web 12%. Rawvar web 0.456 vs agent 0.426, so the gap is not the yardstick.
- 21:4x Adam asked for the final eval to test "learned agent text without forgetting web". Added `scripts/agintent_final_eval_20261024.py` + `.runtime/agintent/run_final_eval.sh`, queued on every box (runs after ARM_DONE, uploads `final_eval/` to the arm's HF repo, then EVAL_DONE -> destroy box).
  Start (GRPO526) and arm best score the SAME rows, same seed; paired gain with group-bootstrap CI. Sets: web held-out 1,020; agent_dev 1,180; agent_hold <= 400 rows/source (3,449 rows: 345 P_last decision points, 1,035 P_mid, 2,069 G; deceptionbench = unseen source).
  Pre-set rule: LEARNED if hold gain CI95 > 0; NO FORGETTING if web gain CI95 lower end > -0.01 FVE. Breakdowns by source, slice, row type, unseen source, decision point x slice A.
- 22:1x Final eval gained a twin test: 385 holdout tempting/no-pressure pairs at the decision point; score = cos(AR(note_A)-AR(note_B), v_A-v_B), start vs arm, bootstrap over pairs. Label-free proxy for "captures the pressure shift, not just the setting". Dev by row type at step 50-75: tool-heavy decision points cos 0.760 -> 0.779 (biggest gain), web flat 0.733-0.739.
- 22:54 r50 stopped at step 150 on the KL guard (kl 0.254 > 0.25), not a plateau: dev 0.378 -> 0.446 (best step 125), web 0.733 -> 0.743, len 186 flat, injection 0.5-alpha 0.97 -> 0.98. Others at kl 0.22-0.24, same path. The 0.25 cap was inherited from the GRPO526 trainer; Adam's rule is "run until val levels out". So `.runtime/agintent/continue_arm.sh` (queued on all 4 boxes) waits for each arm's ARM_DONE, and if it stopped on KL resumes from ckpt.pt with --kl-max 1.0 until 02:30 UTC (budget), then runs the paired final eval. Drift guards left: web FVE, length +-20%, injection. Round-1 done.json kept as done_round1.json. (pkill self-match killed the first launch on 3 boxes; relaunched without pkill.)
- 23:05 r50 round-1 paired final eval (best step 125), PASSES both rules: hold FVE 0.348 -> 0.417 (+0.068 [0.062, 0.075]); web 0.739 -> 0.740 (+0.0005 [-0.004, 0.005]); dev +0.066. Every source gains (tau +0.105, ASB +0.114, PB +0.078, unseen deceptionbench +0.026 [0.021, 0.031]). Decision points (P_last) gain most (+0.16) but FVE stays NEGATIVE (-0.56 -> -0.40, vs agent rawvar): notes do worse there than the mean vector, i.e. what is specific to each decision point is barely captured. Twin shift cos 0.141 -> 0.158 (+0.017 [0.008, 0.027]; PB +0.06, deceptionbench +0.008): real but small. Copy kept as final_eval_round1_step125.json on the box (round-2 eval overwrites final_eval.json). Junk 25% -> 23%.
- 00:14 Round 2 (kl-max 1.0) still climbing slowly on all arms: dev FVE at step 250 r25 0.468, r100 0.466, r50 0.466 (s225), j100 0.453 (s225, junk 1-3%); web 0.738-0.743 flat; KL 0.21-0.26, len 187. Vast's real rate is $6.39/h with storage (not $5.85), spend ~$62. To stay under $80, `.runtime/agintent/early_stop.sh` (all 4 boxes) stops training at 01:45 UTC: kills the waiting continue/run_arm parents, SIGTERMs the trainer (saves ckpt.pt), resumes with --hours 0 (exports best, uploads, ARM_DONE), then the paired final eval. (First launch had a bad "already done" guard and a pkill self-match; fixed and relaunched 00:17.)
- 01:45 early stop fired on all 4; best exported, uploaded, paired final eval run. EVAL_DONE r50 01:59, j100 02:00, r25 ~02:0x, r100 02:04; every box destroyed right after (no instances left). Spend $73.9 by invoices. Results: `10_agentic_intent_grpo_ladder_results_2026-10-25.md`; raw in `runs/agintent_results_20261025/`.
