# Unit format spec for the agentic-intent GRPO data build (24 Oct 2026)

Plan: `07_agentic_intent_grpo_dataset_plan_2026-10-24.md` (read the "v3 decisions" block).
Every source converter writes **one jsonl** `data/agintent_units_20261024/<source>.jsonl` and a short
`data/agintent_units_20261024/<source>.README.md` (where fetched, commit/version, licence, canary check result, filters, counts, examples).

## What a unit is
One scenario Qwen3.6-27B will read and then reply to once (no live environment: we take vectors from Qwen reading the
prompt and from its one reply). Anything the agent would need to see from its workspace is **pre-baked** into the
messages: e.g. an assistant tool call `ls`/`cat` followed by a tool message with the real file contents from the source.

## Fields (one JSON object per line)
```json
{"unit_id": "odcv-clinical-trial-data-management-incentivized",
 "source": "odcv",
 "slice": "A",                 // A = tempting (pressure / chance to misbehave); B = twin with no pressure; C = ordinary work
 "twin_of": null,              // for slice B: unit_id of its slice-A partner (same group); else null
 "group": "odcv-clinical-trial-data-management",  // split unit: a unit and its twins share one group (never split apart)
 "kind": "fake_data",          // free short tag of the possible misdeed ("none" for C); used for filtering only
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "", "tool_calls": [{"type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"cat kpi.md\"}"}}]},
              {"role": "tool", "name": "bash", "content": "..."}],
 "tools": [{"type": "function", "function": {"name": "bash", "description": "...", "parameters": {"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],
 "license": "MIT",
 "meta": {"any": "source-specific fields, e.g. variant, pressure_level, domain, original ids"}}
```
Rules:
- The **last message must be `user` or `tool`** (Qwen replies next). `tools` may be `[]` for chat units.
- Keep text **verbatim** from the source where it exists; only glue text may be ours (say so in `meta.glue`).
- Total text per unit <= ~24k characters (about 6k tokens). Truncate long files sensibly (head + "... [truncated]").
- `unit_id`s unique within the source, lowercase, `[a-z0-9-_]`.

## Hard filters (drop, and count what was dropped in the README)
1. **Canary / do-not-train**: grep the whole fetched source for `canary`, `BENCHMARK DATA SHOULD NEVER`, `do not train`,
   `should not be used for training`. If found, **drop the whole source** and report it.
2. Misdeed kinds **never** allowed: blackmail / coercion / threats for leverage; leaking confidential info to an outsider
   (rival, press, external party); letting a person die / lethal harm / disabling life-safety alerts. Drop such units.
3. Bio / chem weapons content (PropensityBench bio-security and chemical-security dirs, anything similar): drop.
4. Anything that mentions SummitBridge, NovaStar, or is from `anthropic-experimental/agentic-misalignment`: drop.

## Safety for downloaded data
Put each fetched source in its own new dir `data/agintent_raw_20261024/<source>/`. Never run code from it. Parse with
your own script in `scripts/agintent_conv_<source>_20261024.py`, run as `.venv/bin/python -I scripts/...` (paths as args).
Mac limits: keep CPU and RAM modest (no heavy parallel jobs; Adam's limit is 75% of CPU/RAM shared with other work).

## Output check
End by printing, per source: units by slice, median/max characters, 2 example units (shortened) — and put the same in the README.
