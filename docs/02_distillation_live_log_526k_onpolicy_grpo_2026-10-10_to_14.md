# Live handoff: 27B NLA -> Qwen3.5-2B distillation (from 10 Oct 2026)

Plan: `03_distillation_plan_2026-10-10.md` (sections 12-16 are current). Budget $150 (stop and ask
at ~$130 before R1 ends). One GH200 approved (Adam waits up to 24 h for one; any region). Adam away from ~12:00 UTC 10 Oct.

## State (update at every stage)

- 10 Oct 11:08 UTC: GH200 poller started (every 5 min, 24 h): scratchpad `gh200_poll.log`; none free so far.
- 10 Oct ~12:30: input sample built (Mac, $0): `data/nladistil27b_sample_20261010/` — 263k train rows from 87,674
  teacher docs (≤ 3 per doc, nested rungs 33k/66k/131k/263k = 11.0k/22.0k/43.7k/87.7k docs; AV-SFT 65.6k, AR-SFT 65.8k,
  RL 131.6k rows, the teacher's RL eval slice excluded); held-out 1,020 vectors from 340 fresh FineFineWeb docs across all
  67 domains, 0 in the teacher's data; teacher-eval 1,020 rows from its own RL eval slice.
- 10 Oct ~13:30: all box scripts written; Mac smoke test of the whole pipeline (Qwen3.5-0.8B stand-ins, MPS) running.
  CPU was too slow for Qwen3.5 generation on Mac (80 s per forward); MPS is ~100x faster.

- 10 Oct ~13:45: Adam added Nebius (CLAUDE.md) and approved running the whole job on a Nebius pre-emptible GPU if a
  $10 test passes (Lambda GH200 as fallback). The service account only reaches uk-south1 (B300); RTX PRO 6000 regions
  need projects it cannot create. Spot B300 price ~$2.50-2.75/h at 13:00 UTC; cap policy `pricingpolicy-e03cuhkinrhr46cntv9xi`
  at $2.75. Test VM `computeinstance-e03bw197wd21sznjpj` (89.124.39.132), data disk `computedisk-e03fr7nb4q37j35h7h`
  (600 GiB, /data), guest poweroff failsafe at 15:30 UTC. Downloads 141 GB in 521 s. torch 2.14 cu130 OK on B300.
  Mac smoke test of training stopped: 16 GB RAM swaps on a 0.8B full fine-tune; remaining checks run on the B300.

## Conventions confirmed in source (EasyNLA@4d72847, nanoNLA@3e3a62a)

- Layer 42 = output of decoder block 42 (hook `layers[42]`, HF `hidden_states[43]`), last token of the prefix fed alone.
- Teacher injection: output of block 1, marker 158983 (neighbours 29 / 510): h + |h| v/|v|; chat template, thinking off.
- AR: 43-block `ar_reconstructor`, lm_head and final norm removed, last token of the critic prompt, normalise to sqrt(d),
  fp32 Linear(5120, 5120) `value_head.safetensors` (key `weight`); FVE = 1 - MSE / rawvar on sqrt(d)-normalised vectors.
- Qwen3.5-0.8B/2B/4B share the 27B's tokenizer (vocab identical); all load as `Qwen3_5ForCausalLM`; left and right padding
  give the same logits in HF (checked on Mac).
- Pre-flight checks our extraction against the 64 published L42 vectors in the teacher repo (`data/example_activations.parquet`).

## Files

| What | Where |
|---|---|
| Scripts | `scripts/nladistil27b_{build_sample,common,extract,teacher,ar_score,train_student,onpolicy,rl_grpo,decide,preflight,hf_upload,wsb_preview}_20261010.py` |
| Box queue + ops | `.runtime/claude_upload_nladistil27b/{setup.sh (zsh, Mac), queue.sh, failsafe.sh, sync.sh, finish.sh, status.sh}` |
| WorkspaceBench items for the preview | `data/wsb_items_92d763e/` (6 single-token families + the judge's system prompt) |
| Launcher | `scripts/claude_gpu.py` now takes `CLAUDE_GPU_ANY_REGION=1` (own disk `claude-nladistil27b-<region>` outside us-east-1) |

## Queue order (box) and default decisions (`nladistil27b_decide_20261010.py`, rules fixed before results)

downloads -> extract held-out -> extract train/probes/blends -> pre-flight gates -> vLLM check (else HF teacher) ->
teacher labels (held-out, teval, inject, notes, train 263k, blends 14.4k, second samples 65.5k) -> HF upload (bg) ->
pilot 10k (gate) -> 66k (+ lr 5e-6, 2e-5, 1.0 / 2.0 passes) -> 33k -> 131k (+ blends if gate, 2 notes, text-only, 0.8B, 4B)
-> main-recipe decision -> 263k main (saved) -> on-policy 100k (sample, teacher score, train) -> GRPO (R1) -> preview -> done.
Overrides: write `~/lenses/runs/remote_logs/decisions.json` on the box (`{"backend": "hf"}`, `{"run_s3": false}`,
`{"main_args": "--lr-lm 2e-5"}`, `{"pilot": true}`).
- 10 Oct 13:11 UTC: main queue started on the Nebius B300 (systemd units claude-queue + claude-failsafe; Mac watcher
  `nebius_watch.sh` every 5 min). Train extraction shard 0: 10k rows in 420 s (incl. load).
- 10 Oct 13:21: deliberate stop/start test passed: VM stopped and restarted, same IP, both units came back, queue
  re-entered extract_train at 13:22 and kept shard 000.
- 10 Oct 15:00: extraction done (train 263k + probes + blends, ~1.6 h at ~7.5k tok/s, GPU 100%). Pre-flight passed
  (extract cos mean 0.9993). Backend = HF (bs 1000). Teacher on our 1,020 fresh held-out: FVE 0.754 [0.744, 0.764],
  99.9% parsed, 182.6 tokens, 3.2k tok/s. Teacher labels for train expected ~5.5 h (to ~20:30 UTC).
- 10 Oct 15:35 and ~16:02: two real pre-emptions during teacher_train (watcher restarts; one start sat in STARTING
  ~20 min). Back RUNNING 16:06, queue resumed teacher_train from done shards. Mac router DNS fails for the Nebius
  billing host (pricing-policies.billing-cpl...); 8.8.8.8 resolves it. B300 pre-emptible availability "HIGH" at 15:48.
- 10 Oct 16:35 -> 11 Oct 04:20: STALLED. Teacher now saves every 1,000 rows (T_r0000000 names; old T_000 kept).
  But from ~17:50 UTC the B300 spot market sat above our $2.75 cap (price policy SCHEDULING_STATE_BLOCKED): 45 stops,
  watcher hit 40 restarts. Box ran only ~2.5 h in total (~$7). Train teacher labels: ~10k+ of 263k done. Watcher
  stopped; VM STOPPED (disk kept). Waiting on Adam: raise cap / on-demand / wait / Lambda.
- 11 Oct ~04:40-05:00 UTC: Adam moved the run to Vast.ai and asked to end Nebius cleanly. Data disk attached to a
  cpu-d3 VM and uploaded to HF `adamjb111/nla-qwen36-27b-L42-distill-data` under `work/` (vec27b 8.81 GB = all sets,
  L28/L42/L52; teacher27b 0.33 GB = heldout, teval, inject, notes, train rows 0-10k; logs_b300 incl. done-files).
  Then ALL Nebius resources deleted (both VMs, both disks, price policy); only the default project network remains.
  Nebius total spend ~$10. Vast: chosen RTX PRO 6000 WS, Spain, offer 29535827 (rel 1.000, 123 GB RAM, ~$1.44/h with
  400 GB); create failed "account lacks credit" -> waiting for Adam to add credit.
- 11 Oct 04:45 UTC: Vast instance 55337963 (RTX PRO 6000 WS 96 GB, Spain, ~$1.44/h incl. 400 GB disk; ssh helper
  `.runtime/vs.sh`). /data -> /workspace. Box scripts in `.runtime/claude_upload_nladistil27b/vast_*.sh`: onstart
  (links, env once, failsafe, HF sync, queue), failsafe (60 h runtime limit; stops itself 45 min after queue end via
  CONTAINER_API_KEY), hfsync (every 30 min: teacher27b, logs, keep -> HF dataset work/), restore (B300 state from HF).
  Env: torch 2.7.1 cu128 (image) + transformers 5.17 + fla + causal-conv1d OK. Teacher batch 320 (96 GB). Mac watcher
  `vast_watch.sh` -> vast_watch.log; event wait: `W=... wait_event.sh`. Queue restarted 04:47 at downloads.
- 11 Oct 05:05: Vast HF teacher at batch 192 (320 OOMs): 877 tok/s, 1,000 rows / 220 s, 100% parsed -> labels left
  (~61M tokens) ~19 h. vLLM 0.19 + vllm-lens 1.1.0 installs on this box (torch 2.10 cu128), but EasyNLA@4d72847's
  REQUIRED injection patch fails ("hunk 9 not found") -> not used; staying on HF (no fix attempted, to save tokens).
  Revised total from 05:00: ~52 h, ~$75 (plus ~$10 Nebius).
- 11 Oct 08:00-11:00: vLLM teacher fixed and switched on (09:49 UTC). Fixes: EasyNLA patch steps applied in order;
  base-model tokenizer; `/data/teacher_vllm` = original multimodal config + merged LM weights + vision weights in a
  separate file. Held-out FVE 0.756 (HF 0.754); ~1,400 tok/s vs HF ~1,010. Queue switch: flag `/data/force_vllm` ->
  queue.sh sets BACK=vllm, queue_env.sh sets VPY=/data/envs/vllm-lens/bin/python and vLLM TARGS (4k-row calls, fp16
  GDN state). Teacher script now labels only missing rows (files of any size never overlap) and records `backend`
  per row. Train rows 247k-263k were labelled during tests and are kept. vllm-lens on the box carries our fast-path
  patch (identical outputs); backups `_worker_ext.py.easynla`, `.fasthook`, `_activations_plugin.py.orig`.
  `pause_run.sh` now also kills vllm-lens python + VLLM::EngineCore, waits for the GPU and clears queue.failed
  (an orphan teacher made teacher_train FAIL at 10:00; recovered 10:06; /data/hold was set and is cleared).
  Student fast sampling: `scripts/nladistil27b_fast_student_gen_20261011.py` (FastGen, 2.6x); not yet wired in.
  GPU guide: `06_gpu_choice_and_inference_speed_2026-10-11.md`.
- 11 Oct 16:20 UTC: teacher_train at 225k/263k rows (~1,400 tok/s, 99.9% parsed). Then blends (14.4k) + train_s1
  (65.5k): teacher labels done ~20:30 UTC. Spend so far ~$27 (Vast 11.5 h ~$16.6 + Nebius ~$10). Vast credit $68.4
  (~47 h at $1.44): top-up needed before GRPO ends if the run goes past ~Tue 13 Oct 15:00 UTC.
- 11 Oct 16:57 UTC: teacher_train DONE: 263,000 rows, all unique (0..262,999), 99.92% parsed, none at the 256 cap;
  backend mix HF 69k / vLLM 194k. teacher_blends started 16:57 (14.4k rows), then train_s1 (65.5k).
- 11 Oct 17:29 UTC: teacher_blends DONE (14,400 rows, 99.95% parsed, ~1,410 tok/s); train_s1 (65.5k) running.
  S3 gate: `runs/remote_logs/preflight/preflight.json` was lost in the Nebius -> Vast move (the gate would have crashed
  and skipped S3 with no message). Rebuilt from the last JSON line of preflight.log; dry run gives run_s3=true
  (blends parsed 1.0, AR cos 0.81 vs 0.8 x 0.94 natural; blends FVE 0.18 vs 0.74 natural). My read of 50 vLLM blend
  notes (seed 7): all fluent, no broken output. About two-thirds name both documents' topics; the rest describe one
  (mostly A when alpha = 0.25). But they nearly always invent one hybrid document that links the two (e.g. "flow
  meter and shampoo", "sewing designers' race recap", "private equity room host on Airbnb"). Verdict: coherent text,
  confabulated link. S3 runs as planned; per the plan, it counts only if it lifts natural held-out FVE.
- 11 Oct 19:50-21:00 UTC: train_s1 labels DONE 19:50 (65.5k). Pilot FAILED 19:55: CUDA OOM at the first step of stage B
  (full 2B fine-tune) because the 27B AR (54 GB) stays on the GPU; stage A had peaked at 87.6 GB. GPU idle ~1 h. Box
  failsafe tried to stop the box from 20:40 but `vastai stop` failed (empty CONTAINER_API_KEY), so the box stayed up.
  Mac rebooted ~20:51 (watcher died at 18:21 log).
- 11 Oct 21:00 UTC (new session; Adam: 8 h and $20 to finish and write up): fix = `--ar-offload` on every run (AR on
  CPU between evals) + free grads after each step. /data/hold set (failsafe off; I stop the box myself). Queue cut to
  fit 8 h: pilot -> 66k -> 33k -> 131k -> 263k main (saved) -> stop (`arms_off`, `stop_after_main`). LR/pass probes and
  arms move to a second box if the pilot passes. OP + GRPO not in this window.
- 11 Oct 21:00-21:16 UTC: pilot with the fix: stage B 82.7 GB peak, ~5,960 tok/s (~33 examples/s; stage A ~10.5k);
  AR swap to GPU for evals works (~1 min per quick eval). Pilot at 5k seen: held-out FVE 0.097 (shuffled -0.78), 99.6%
  parsed -> gate passed; 66k rung started 21:12 on box 1. Box 2 = Vast 55446912 (RTX PRO 6000 WS, Spain, $1.39/h,
  ssh `.runtime/vs2.sh`), set up in 6 min from the public HF dataset (L42 only) + AR/tokenizer files only (env var
  NLA_TEACHER_DIR, since av_base weights were not fetched). Box 2 queue (`.runtime/claude_upload_nladistil27b/box2/`):
  131k rung -> text-only 66k -> lr 2e-5 66k -> 0.8B 66k -> blends 11% 66k -> 2 notes x 32.75k vectors; `stop_now`
  file ends it before the next run. 131k on box 1 is skipped (STOPPED.json "moved to box 2").
- 11 Oct 22:14 UTC: 66k rung DONE (1.03 h): held-out FVE 0.566 [0.547, 0.584] = 0.75 of teacher (7B 66k rung: 0.74);
  by seen 30k 0.405 / 60k 0.518 (7B shares 0.40 / 0.66 -> 27B student a bit ahead early). teval 0.593. Inject hit
  α 0/0.25/0.5/1.0 = 0.07/0.15/0.62/0.95 vs teacher 0.06/0.80/1.00/1.00; notes city/job 0.12/0.18 vs teacher
  0.23/0.61 (teacher refs computed on the Mac from the HF labels, same keyword rule). 131k rung (box 2) tracks the
  same curve (30k 0.407, 60k 0.513). 33k started 22:14. Results: `runs/nladistil27b_results_20261011/summary.md`
  (pull + table: `.runtime/claude_upload_nladistil27b/box2/pull_results.sh`).
- 11 Oct 22:47 UTC: 33k rung DONE: held-out 0.468 [0.445, 0.491] (0.62 of teacher; 30k 0.371). 263k main started
  22:47 on box 1 with the default recipe (lr 1e-5, 1.5 passes; decide_main had no probe results). ETA ~02:30 UTC.
  131k (box 2) at 120k seen: 0.598 (0.79; 7B 0.86 at 120k).
- 11 Oct 23:18 UTC: 131k rung DONE on box 2 (2.03 h): held-out 0.635 [0.620, 0.650] = 0.84 of teacher (7B 131k
  rung 0.90); teval 0.646; inject 0.07/0.16/0.70/0.98; notes city/job 0.13/0.27. Ladder 33k/66k/131k =
  0.468/0.566/0.635 (+0.098, +0.069 per doubling) -> 263k projected ~0.68-0.69. Text-only 66k started 23:18.
- 11 Oct 23:25 UTC: text-only 66k OOMed at micro 32 (192 text tokens in the prompt). Box 2 queue 1 ends after lr 2e-5
  (stop_now); `box2_queue_b.sh` then runs 0.8B -> text-only (micro 16) -> blends 11% -> 2 notes, none starting after
  02:45 UTC (`stop_b` file skips the rest).
- 12 Oct 00:22 UTC: lr 2e-5 at 66k DONE: 0.578 [0.559, 0.597] vs 0.566 at lr 1e-5 (+0.012, CIs overlap; passes the
  plan's +0.01 rule, so a re-run main would use 2e-5; the running main uses 1e-5). Main at 120k seen: 0.602 (0.80).
  0.8B at 66k started 00:22 (queue b).
- 12 Oct 01:03 UTC: 0.8B at 66k DONE (0.68 h): 0.463 [0.441, 0.484] = 0.61 of teacher (2B 0.566); teval 0.501;
  inject 0.07/0.13/0.50/0.81; notes 0.10/0.14. Text-only 66k (micro 16) started 01:03. Main at 240k quick 0.655.
- 12 Oct 02:32 UTC: 263k main DONE (3.71 h): 0.680 [0.665, 0.692] = 0.90; teval 0.695 (teacher 0.769, scored now);
  inject 0.07/0.22/0.83/1.00; notes 0.14/0.42. Uploaded public: adamjb111/nld27-sftkd-qwen35-2b-from-nla-qwen36-27b-
  L42-pairs263k-ep1.5-20261012 (final_lm, final_adapter.pt, metrics). Text-only 66k 0.539. Box 1 (55337963) DESTROYED
  02:32 after pulling all runs (no grid weights kept, by plan). Ledger: runs/nladistil_scaling_ledger.jsonl.
  Next (not started): on-policy -> GRPO -> preview; restore the main student from HF first. Write-up:
  01_distillation_weekend_findings_sft_ladder_2026-10-12.md
- 12 Oct 03:32 UTC: blends 11% at 66k DONE: 0.584 [0.565, 0.602] vs 0.566 (+0.018); inject α .5 0.73 vs 0.62; job
  0.25 vs 0.18. 2-notes skipped (02:45 cut-off). Box 2 (55446912) DESTROYED 03:32. No Vast instances left. Session
  spend ~$17 (credit 58.11 -> 40.14 incl. part of another agent's box). Weekend ~$50 of $150. Run paused here.
- 12 Oct 06:53 UTC: Adam: "do A+B" (on-policy round, then GRPO), then C (more pairs) or D (4B) only if GRPO levels off
  early. Box 3 = Vast 55512962, RTX PRO 6000 Server 96 GB, France, $1.43/h, 350 GB disk, 566 GB RAM. Scripts:
  .runtime/claude_upload_nladistil27b/box3/ (box3_setup.sh, box3_queue.sh); ssh helper .runtime/vs3.sh. Queue: merge
  teacher on CPU || smoke (256-row sample, GRPO 2 steps) -> sample 100k -> smoke score+8 SFT steps -> score 100k ->
  on-policy SFT (lr 5e-6, soft-w 1.0, 1 pass) -> GRPO-AR (10 h cap, plateau rule) -> full eval of the RL student.
  Vast credit at start $40.09. No box-side stop works on Vast: Claude checks every 5 min and destroys from the Mac.
- 12 Oct 10:29 UTC (box 3): smoke tests passed (256-row sample/score/8 SFT steps; GRPO 2 steps 41 s/step, 77 GB, full
  eval 0.681 = main). Sample 100k 34 min; teacher score 100k 1.87 h (~2.7k tok/s, bs 32). On-policy SFT (100k, 1 pass,
  lr 5e-6, soft only) 58 min: held-out 0.686 [0.670, 0.700] (0.91; main 0.680); teval 0.702 (0.93); inject
  0.07/0.29/0.91/1.00 (main 0.07/0.22/0.83/1.00); notes 0.13/0.43 (same). GRPO from it started 10:29 (10 h cap).
- 12 Oct 11:43 UTC: GRPO evals (256 held-out subset) step 0 / 50 / 100: FVE 0.680 / 0.710 / 0.721; inject a.25 .29/.27/.31,
  a.5 .94/.93/.96; a0 .09/.07/.06; len 183/187/185. Mean KL per step rose 0.001 -> 0.07 -> 0.085 (step 96): the
  script's 0.1/step KL stop would have ended a healthy run. The 7B GRPO ran at mean KL 0.10-0.15 for 500 steps with
  gains (runs/nladistil_grpo_qwen25-1.5b_..._20261006). Changed (Claude): killed at the step-100 ckpt, resumed with
  --kl-max 0.25, all other guards as planned (box3_resume_kl.sh). Deviation from plan s.15, logged here.
- 12 Oct 12:55 UTC: GRPO eval step 150: 0.724 (step 100 0.721); train FVE flat ~0.755 since step 75; KL 0.13. Likely
  plateau stop at step 200-250 (~96% of teacher on the subset). Per Adam: if GRPO levels off early -> C or D. Vast
  credit covers D, not C, so box3_queue_d.sh waits on the box: uploads the on-policy + GRPO students (public HF), then
  (only if GRPO reason != hours) a 4B smoke test and the 4B SFT at 263k, same recipe as the 2B main. Trainer: new
  --opt-offload (AdamW state to CPU during evals; 4B state ~34 GB + AR 35 GB do not fit on 96 GB).
- 12 Oct 13:32 UTC: GRPO stopped on plateau at step 250 (evals 100/150/200/250: .721/.724/.725/.722 on the 256 subset).
  Full held-out: 0.730 [0.719, 0.740] = 0.97 of teacher (0.754 [0.744, 0.764]); inject 0.08/0.34/0.96/1.00. rl_eval
  (notes, teval) running. Adam: no 4B for now (D held: hold_d on the box; box3_queue_d.sh only uploads OP1 + GRPO1);
  credit auto-tops up. Adam's go (13:45): round 2 = second on-policy round from the GRPO student (rows 100k-200k), then
  GRPO with LoRA r128, 16 samples/vector (64x16), KL stop 0.25, min 150 steps, 8 h cap (box3_queue_r2.sh).
- 12 Oct 17:01 UTC: OP2 done (sample rows 100k-200k from the GRPO student 34 min, score 1.86 h, SFT 58 min). Held-out
  0.694 [0.681, 0.706] (0.92), DOWN from GRPO1 0.729; teval 0.703; inject 0.09/0.28/0.92/1.00; notes 0.21/0.45. Read:
  soft KD pulls the student back toward copying the teacher's words and undoes most of what GRPO gained on the AR
  reward. Uploaded (adamjb111/nld27-onpolicy2-...). GRPO2 (r128, g16) started 17:01 from OP2.
- 12 Oct 18:10 UTC: Box 4 = Vast 55595472, RTX PRO 6000 S, Virginia, $1.53/h, 300 GB disk, 128 GB RAM (7B-teacher side + 27B hard-label runs).
- 12 Oct ~18:00 UTC: Adam's go: (1) a second 263k of the teacher's own rows -> 526k student -> on-policy -> GRPO;
  (2) teacher-size point: the 7B NLA (kitft Qwen2.5-7B L20) also teaches Qwen3.5-2B. Adam ended GRPO round 2 (stopped
  at step 43; subset FVE at step 0 was 0.689, no later eval). Facts found: the teacher's data = 944k usable rows from
  99,956 docs (~10 cut points/doc); the first 263k took <= 3 rows from 87,674 docs, so ~680k rows were unused.
  Qwen2.5 vs Qwen3.5 tokenizers share 281 of 151,665 ids -> top-20 KD cannot cross; the teacher comparison uses hard
  labels (soft-w 0) for BOTH teachers (27B soft runs show the KD bonus).
  Code (12 Oct): scripts/nladistil27b_build_sample526k_20261012.py (Mac; rebuilt the first 263k row for row, then
  train2 = 36.8k rows from the 12.3k unreached docs + rows 4-6 of every doc); extract script --layers and set train2;
  common.py NLA_PROFILE=7b (teacher side only: Qwen2.5-7B-Instruct L20, D 3584, kitft AR, no input norm before its
  head); scripts/nladistil27b_teacher7b_20261012.py (7B notes re-spelled in Qwen3.5 tokens, one-hot top-20 = hard).
  Box 3: hard-label 27B 66k (nld27_sfthard_..._pairs66k) running; then box3_queue_526k.sh (Qwen3.6-27B download ->
  train2 L42 extraction -> set train526 via symlinks -> teacher labels rows 263k-526k (HF unless /workspace/use_vllm)
  -> 526k SFT -> on-policy op526_r1 -> GRPO (kl-max 0.25) -> full eval).
  Box 4 (box4_setup.sh, box4_queue.sh): 7B L20 vectors for the same rows and held-out texts -> 64-row smoke -> 7B
  labels (heldout, teval, inject, notes, train 263k) -> 7B teacher FVE -> hard students 7B 66k/131k/263k -> 27B hard
  131k/263k.
- 12 Oct 19:16 UTC: 27B hard-label 66k (soft-w 0): held-out 0.563 [0.545, 0.580] vs top-20 KD (soft-w 0.5) 0.566 ->
  KD adds ~nothing at 66k; teval 0.581; inject 0.07/0.13/0.58/0.94; notes 0.12/0.21. vLLM teacher rebuilt on box 3
  by a helper agent (box3_vllm_prep.sh; env /workspace/envs/vllm-lens, /workspace/teacher_vllm with symlinked
  weights): GPU check on 1,020 held-out FVE 0.7549 (HF 0.7545), parsed 0.999, 182.5 tok -> /workspace/use_vllm set;
  train2 labels will use vLLM. train2 extraction started 19:16. Box 4: 7B vectors train 263k + probes in 57 min;
  7B teacher on 64 of our FFW held-out: FVE 0.738 [0.709, 0.764] (its own UFW/WildChat held-out: 0.732), rawvar
  0.726, parsed 1.0, 144 tok (148 in Qwen3.5 tokens). 7B labels running.
- 12 Oct 20:10:43 UTC: the managing chat (c5cce600) and the tmplae chat (8526b0c8) both died at the same second (cause
  unknown; Mac awake, no Claude update; likely the terminal closed). Their Mac watchers died too. Box queues ran on.
- 13 Oct 07:40 UTC (new chat 127bb851 takes over; Adam: keep boxes 3 and 4 running). Box 4 results (hard labels,
  soft-w 0; FVE on 1,020 held-out, "share" = FVE / own teacher FVE):
  7B teacher (0.733): 66k 0.675 (0.92) / 131k 0.693 (0.95) / 263k 0.701 (0.96); teval 0.694/0.707/0.714;
  inject a.5 0.40/0.49/0.55, a1 0.82/0.90/0.91; notes job 0.32/0.31/0.35.
  27B teacher (0.754): 66k 0.563 (0.75, box 3) / 131k 0.615 (0.82) / 263k running (quick 256-subset ~0.65 at 340k seen);
  inject a.5 0.58/0.73, a1 0.94/1.00.
  Read: by share-of-teacher the 2B learns the 7B much faster (0.92 at 66k vs 0.75); in raw FVE the 7B-taught student is
  ahead too, but each FVE uses its own AR on its own layer, so raw FVEs are not like for like. The 27B-taught student
  is better at injection (concept steering). Box 3 at 07:40: t_train2 labels 511k/526k (vLLM ~1,500 tok/s); queue
  then runs 526k SFT (--save, ~7.5 h) -> op526 sample/score/SFT (~3.5 h) -> GRPO (<= 10 h) -> eval. The queue does NOT
  upload weights: upload the 526k / OP / GRPO students before destroying box 3.
  Vast credit $21.93 at 07:40 (auto top-up said to be on, 12 Oct). Spend estimate for this plan ~$108 so far; box 3 to
  the end ~ +$20-30 -> ~$130-140, at the plan's "stop and ask" line: flagged to Adam.
- 13 Oct 08:05 UTC: box 4 ALL_DONE. 27B hard 263k: held-out 0.658 [0.644, 0.671] (0.87 of teacher). Ladder 27B-taught
  66k/131k/263k = 0.563/0.615/0.658 (0.75/0.82/0.87) vs 7B-taught 0.675/0.693/0.701 (0.92/0.95/0.96). Pulled to
  runs/nld27_box4_teacher7b_vs_27b_hard_results_20261013/ (5 runs, remote_logs, 7B teacher text labels: all 263k train
  rows + heldout/teval/inject/notes; the one-hot _top.npz files were not kept, they rebuild from the text). No student
  weights were saved on box 4 (by design). Box 4 (55595472) DESTROYED 08:15 UTC.
- 13 Oct 08:00 UTC: box 3 t_train2 DONE: 263,000 new rows (263k-525,999), all unique, 99.93% parsed, 0 at cap. 526k SFT
  started 07:58 (stage A ok, ce 2.6 -> stage B at lr 1e-5).
- 13 Oct 09:05 UTC: 526k SFT at 114k seen; held-out 30k 0.444 / 60k 0.531 (263k main: 0.423 / 0.521) -> on track.
  Vast credit $12.70 and falling ~$6.3/h (box 3 + decision-engine boxes): no auto top-up seen; asked Adam to top up.
- 13 Oct ~09:20 UTC: Adam: run box 3 to the end of its queue (OK to reach ~$130-140 of the $150 cap); Vast auto top-up IS on (low credit is not an alarm).
- 13 Oct 10:56 UTC: 526k SFT at 319k of 789k seen; held-out 120k 0.592 (main 0.602) / 240k 0.643; quick256 280k 0.646, 320k 0.645. Tracking the 263k main within noise; the gain from more data should show after ~400k seen, where the main run ended. ETA ~15:15 UTC.
- 13 Oct 12:47 UTC: 526k SFT at 528k of 789k seen; held-out at 480k 0.680 [0.665, 0.692] = the 263k main's FINAL score, with 40% of the run left (LR still ~half). Quick256 520k 0.674. ETA ~15:15 UTC.
- 13 Oct 15:06 UTC: 526k SFT DONE (7.1 h): held-out 0.697 [0.683, 0.709] = 0.92 of teacher (263k main 0.680
  [0.665, 0.692]); teval 0.714 (main 0.695); inject 0.06/0.22/0.88/1.00 (main 0.07/0.22/0.83/1.00); notes city/job
  0.21/0.45 (main 0.14/0.42). Ladder 33k/66k/131k/263k/526k = 0.468/0.566/0.635/0.680/0.697: the gain per doubling
  shrinks (+0.10, +0.07, +0.045, +0.017), so more pairs alone has nearly run out. Uploading (public):
  adamjb111/nld27-sftkd-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-ep1.5-20261013. op526_sample started 15:06.
- 13 Oct 15:50 UTC: 526k student uploaded (public, verified files): adamjb111/nld27-sftkd-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-ep1.5-20261013. First try failed: the upload script reads ~/.hf_token, box 3 has /workspace/.hf_token -> pass HF_TOKEN=$(cat /workspace/.hf_token). op526_sample DONE 15:41 (34 min); op526_score running (~11 min per 10k rows, ~1.9 h).
- 13 Oct 20:20 UTC: OP526 DONE 18:31: held-out 0.697 [0.683, 0.710] (= M526 0.697; OP adds ~0 FVE, as in round 1 +0.006), inject 0.06/0.30/0.94/1.00, notes 0.13/0.48. Uploading adamjb111/nld27-onpolicy-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-20261013. GRPO from 18:31: subset evals step 0/50/100 0.699/0.714/0.726 (GRPO1 0.680/0.710/0.721). Mac slept ~16:45-20:15 -> watcher gap; started caffeinate 18 h.
- 13 Oct 22:44 UTC: GRPO526 stopped on plateau at step 350 (subset 0.699 -> 0.738 at step 300). Full held-out (eval run)
  0.739 [0.727, 0.750] = 0.98 of teacher (GRPO1 from the 263k path: 0.730); teval 0.750; inject 0.09/0.39/0.97/1.00;
  notes city/job 0.24/0.50. ALL_DONE_526K 22:48 (box idle to 00:41: Claude usage limit).
- 14 Oct 00:50 UTC: uploaded (public): adamjb111/nld27-onpolicy-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-20261013,
  adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013 (+ eval_full_done.json);
  dataset work/teacher27b/train526_rows263000-525999 (132 files, 6.5 GB) + work/vec27b/train2 (3.2 GB). Small results
  pulled to runs/nld27_box3_526k_onpolicy_grpo_results_20261013/. Not kept: op526_r1/op_r1/op_r2 scored samples
  (rebuildable). Box 3 (55512962) DESTROYED. No lenses GPU boxes left. Plan spend ~ $140 of $150.
