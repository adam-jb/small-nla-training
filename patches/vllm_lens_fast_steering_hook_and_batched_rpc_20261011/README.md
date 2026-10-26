# vllm-lens: faster steering hook + one RPC per generate() call (11 Oct 2026)

Upstream: https://github.com/UKGovernmentBEIS/vllm-lens (we ran 1.1.0 from PyPI, plus EasyNLA's
`utils/patch_vllm_lens.py`, which fixes two steering bugs, see "Also worth upstreaming").
Tested with vLLM 0.19.0, torch 2.10 cu128, one RTX PRO 6000 Blackwell (96 GB), on Qwen3.6-27B (NLA teacher) and
Qwen3.5-2B. Both are hybrid models (Gated DeltaNet + full attention).

## Files

| Path | What |
|---|---|
| `upstream_1.1.0/` | `_worker_ext.py` exactly as PyPI 1.1.0; `_activations_plugin.py` as installed before our change |
| `easynla_patched/` | after EasyNLA's patcher (the version we started from) |
| `ours/` | our version (drop-in for `site-packages/vllm_lens/`) |
| `ours_vs_easynla__worker_ext.diff`, `ours_vs_easynla__activations_plugin.diff` | our change only |

## What we changed

1. **Per-step state, built once per forward pass, not once per layer** (`_step_state` in `_worker_ext.py`).
   The hook fires on every decoder layer. Stock code loops over every running request on every layer (config lookup,
   list copy, capture check). At ~260-512 running requests and 24-64 layers that is ~6k-33k Python lookups per decode
   step. We build the per-request steering list, the union of steered layers and the capture list once, at the first
   hooked layer. It is keyed by the identity of vLLM's `ForwardContext` object, and we hold a reference to it, so the id
   cannot be reused.
2. **Skip layers with nothing to do.** If no request steers or captures at this layer, return straight away
   (63 of 64 layers in the NLA case).
3. **One GPU sync per step, not per request.** `query_start_loc` and `seq_lens` are read with one `.tolist()` each,
   instead of 2-3 blocking `.item()` calls per steered request.
4. **No clone when no write lands in this step.** Position-indexed steering (inject at the marker token) only writes
   while the prompt is processed. On decode steps the stock code still cloned the layer output for every steered
   request. `_needs_write` checks the positions first.
5. **One RPC per `LLM.generate()` call** (`set_steering_data_batch` / `clear_steering_data_batch`). Stock code sends one
   `collective_rpc` per request to set, and one to clear.

Same results. On 1,020 held-out NLA prompts with per-request seeds, all 1,020 sampled texts were identical
(character for character) to the stock hook, and FVE was identical to 10 digits (0.7558581829).

## Measured speed

Each config ran in a fresh process. Requests had one steering vector each (layer 1, marker position, norm_match),
logprobs 20, eager mode (vllm-lens forces it).

| Model | Run | stock | ours | plain vLLM, no steering |
|---|---|---|---|---|
| Qwen3.5-2B | 4,000 requests, 182 forced tokens | 11,835 tok/s | **12,708** | 12,741 (eager), 12,646 (CUDA graphs) |
| Qwen3.6-27B | 4,000 NLA rows, natural lengths | 1,386 | 1,408 / 1,376 (two runs) | - |
| Qwen3.6-27B | 2,000 requests, 182 forced tokens | - | 1,795 | 1,831 (eager), 1,744 (graphs) |

- Small model: stock vllm-lens costs 7.1%. Ours costs 0.3%, which is within noise of plain vLLM.
- 27B: the overhead is hidden behind ~0.2 s GPU steps (run-to-run noise is about ±5%).
- Forced eager mode cost nothing at batch 512 on either model: CUDA graphs were not faster. At small batch sizes
  (interactive use) it will matter more. We did not measure that.

## Also worth upstreaming (found on the way, not fixed in our files)

- **Pickling a view serialises the whole base tensor.** `SteeringVector(activations=V[k].view(1,1,-1))`, where V is a
  4,000 x 2,048 tensor, pickles all 32.8 MB of V's storage per request (0.009 MB after `.clone()`). 4,000 requests used
  ~128 GB of host RAM: the kernel killed the process, and the vLLM engine child was left orphaned. Suggested fix: in
  `_patched_llm_generate`, before pickling, call `.detach().clone().contiguous()` on each vector's activations (or
  document the trap).
- **Two steering bugs that EasyNLA's patcher already fixes** (in `easynla_patched/`):
  (a) `norm_match` was anchored to `output[0]` alone, but for decoder layers that return
  `(hidden_states, residual)` the true stream is the sum, so the injection was mis-scaled (EasyNLA saw ~40% GRPO
  clip fraction from this);
  (b) on vLLM V1, `attn_metadata` is a dict, so `getattr(attn_metadata, 'seq_lens')` was always None, and split
  prefills were steered at the wrong position.
- **EasyNLA's patcher bug:** each patch step's "old" text is checked against the unpatched file, but step 9's old text
  is step 7's new text, so it fails with "hunk 9 not found". Applying the steps in order works.
  Re-running it adds `COUNT_GLOBAL` twice.

## How to send upstream

The changes touch `_hook_inner` (rewritten around `_step_state`), two new worker methods, and two short blocks in
`_patched_llm_generate`. Against upstream 1.1.0, the PR should carry the EasyNLA fixes (a) and (b) together with ours,
because our `_hook_inner` keeps their semantics. Or do it as two PRs: correctness first, speed second.
Test to include: same seeds, steered outputs identical with and without the fast path; and a 2B throughput check.
