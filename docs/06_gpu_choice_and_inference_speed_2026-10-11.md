# GPU choice and inference speed for the 27B NLA distillation (measured 11 Oct 2026, Vast RTX PRO 6000 WS)

Box: Vast instance 55337963, RTX PRO 6000 Blackwell Workstation (96 GB, 188 SMs, cc 12.0), Spain, $1.44/h incl.
400 GB disk. Scripts: `scripts/gpu_profile_matmul_bandwidth_20261011.py`, `scripts/vllm_teacher_ceiling_bench_20261011.py`,
`scripts/vllm_teacher_steered_forced_len_bench_20261011.py`, `scripts/vllm_lens_overhead_bench_small_model_20261011.py`,
`scripts/nladistil27b_fast_student_gen_20261011.py`. vllm-lens patch: `patches/vllm_lens_fast_steering_hook_and_batched_rpc_20261011/`.

## 1. What the card really does (torch 2.7.1 cu128)

| Format | Dense matmul 8192³ | Decode shape 192x5120x17408 |
|---|---|---|
| fp32 (no TF32) | 65 TFLOPS | - |
| tf32 | 154 | - |
| bf16 | 369 | 201 |
| fp16 | 355 | - |
| fp8 e4m3 | 719 | 460 |
Memory copy: 1,467 GB/s (Vast lists 1,398).

So: never leave a matmul in fp32 (5.7x slower than bf16). Our trainer uses bf16 autocast and allows TF32, which is right.
fp8 doubles matmul speed, but Adam ruled out an 8-bit teacher (10 Oct), so it is used nowhere.

## 2. 27B teacher (NLA explanations, ~182 tokens each, top-20 logprobs kept)

| Setup | tok/s | $ per M tokens at $1.44/h |
|---|---|---|
| HF generate, batch 192 (320 runs out of memory) | ~1,010 | 0.40 |
| vLLM + vllm-lens, 1,020-row calls | ~1,090-1,110 | 0.37 |
| vLLM + vllm-lens, 4,000-row calls, fp16 GDN state, 512 seqs, len 512 (**in use since 09:49 UTC**) | ~1,310-1,430 | 0.29 |
| same, our vllm-lens patch | 1,376-1,408 (no change on 27B) | 0.29 |
| same, no per-row seeds | 1,292 (no gain) | - |
| ceiling: forced 182 tokens, steered | 1,795 | 0.22 |
| ceiling: forced 182 tokens, plain vLLM eager / CUDA graphs / no logprobs | 1,831 / 1,744 / 1,756 | - |
| plain vLLM, fp8 weights (not used) | 3,020 | 0.13 |
| (B300 on Nebius, HF batch 1000, 10 Oct) | 3,200 | ~0.24 at the $2.75 cap |

Quality is the same: held-out FVE 0.756 (vLLM) vs 0.754 (HF), 99.9% parsed, 182.5 tokens.
Where the time goes:
- vllm-lens costs ~2%.
- CUDA graphs and logprobs cost nothing at this batch.
- Small calls cost ~25%: the last requests drain with the batch half empty.
- Varied output lengths cost ~25% (1,795 -> ~1,400). Most steps mix new prompts with ongoing decoding. We found no
  cheap fix.

Effect on this run: the remaining teacher labels (train rest + blends 14.4k + second samples 65.5k) take ~9.5 h, not
~13 h. That saves ~$5.

## 3. 2B student sampling (evals, on-policy round, GRPO rollouts)

| Setup (1,020 soft-token prompts, 182 forced tokens) | tok/s |
|---|---|
| HF `sample_embeds` loop, bf16, batch 512 | 7,519 |
| vLLM with prompt embeddings (`FastGen`), M-RoPE dropped | **19,234** (2.6x) |
| plain vLLM on token prompts (4,000 x 182), eager / CUDA graphs | 12,741 / 12,646 |

- vLLM refuses prompt embeddings on Qwen3.5 because M-RoPE needs token ids. For text-only input, M-RoPE gives every
  token the same position on all three axes, which is exactly 1-D RoPE, so `FastGen` drops the M-RoPE keys.
- Check: greedy outputs match HF on the first 5 tokens for 32/32 prompts. The median shared prefix is 27 of 64 tokens;
  after that, bf16 kernel differences take over.
- With natural lengths the gap should grow: the HF loop runs every batch until its longest sample finishes.
- Not yet wired into the on-policy, eval or GRPO scripts. Next: weight loading (`load_lm_state`) needs a test with a
  trained student.

## 4. Which GPU for which job (Vast prices 11 Oct ~11:00 UTC, single GPU, reliability ≥ 0.98)

Read the price with the disk added. Vast's list price leaves out storage, which can be $0.2-1.0 per GB-month: 400 GB at
$0.87 adds ~$0.48/h. Our $1.44 Spain box was the best real deal; an offer listed at "$0.94" was ~$1.41 with disk.

| Job | What limits it | Good choice | Why |
|---|---|---|---|
| 27B teacher labelling (decode, bf16) | memory bandwidth, then compute; needs ≥ 80 GB (54 GB weights + KV and GDN state) | RTX PRO 6000 now (measured $0.29/M tok). **Try H100 NVL** (94 GB, 3.4 TB/s, from $2.07/h): if it reaches ~2.5x our tok/s (estimate, not measured) it costs ~$0.19/M tok. | Bandwidth per $: H100 NVL 1.6 TB/s per $/h, RTX PRO 6000 1.0, H200 1.0, B200 1.0 |
| AR scoring (27B prefill only) | bf16 compute | any ≥ 80 GB card. RTX PRO 6000 is fine (369 TFLOPS for $1.44). | - |
| 2B student SFT (full fine-tune, fp32 master weights + Adam ≈ 32 GB + activations) | bf16 compute | RTX PRO 6000 or H100. A 32 GB RTX 5090 ($0.38/h) would need bf16 weights + 8-bit Adam (untested). | - |
| GRPO (2B policy + 27B AR + rollout engine in one card) | memory first | ≥ 94 GB: RTX PRO 6000 or H100 NVL | 80 GB needs AR offload |
| Student evals / sampling only | launch overhead | any card, with vLLM (`FastGen`) | 2.6x over the HF loop |
| Small jobs < 40 GB (probes, 0.8B students) | - | A100 40 GB from $0.27-0.43/h, RTX 5090 $0.38/h | - |

Listed bandwidth (GB/s) and cheapest price: RTX PRO 6000 WS 1,398 ($0.94 + disk) · H100 SXM 2,461 ($1.74) · H100 NVL 3,389
($2.07) · H200 4,058 ($4.08) · B200 6,166 ($6.20) · A100 40 GB 1,310 ($0.27) · RTX 5090 1,455 ($0.38) · L40S 625
($0.54) · RTX 6000 Ada 806 ($0.54).

## 5. Rules of thumb we learned

1. Measure before you rewrite. The guess "vllm-lens forces eager mode, so it is slow" was wrong for big batches
   (graphs gave 0%). The real costs were call size and output-length mix.
2. Give vLLM big calls (≥ 4k requests) and an fp16 GDN state (`mamba_ssm_cache_dtype='float16'`; bf16 is not
   allowed). Logprobs are free.
3. For our NLA work, vllm-lens's own overhead only shows on small models (7% on a 2B). Our patch removes it.
4. Never pass a tensor view as a steering vector: pickling it copies the whole base tensor (32.8 MB instead of
   0.009 MB per request; 4,000 requests used up 123 GB of RAM).
5. `pause_run.sh` must kill both Python envs and `VLLM::EngineCore`. Missing that left an orphan teacher on the GPU and
   marked the queue failed (fixed 10:06 UTC).
