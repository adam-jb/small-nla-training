"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers regardless of model architecture."""
    # Module.__getattr__ returns Tensor | Module, so pyright can't narrow
    # through chained attribute access.  Use Any for duck-typed traversal.
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (
        hasattr(m, "model")
        and hasattr(m.model, "decoder")
        and hasattr(m.model.decoder, "layers")
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers"
    )


# NLA explicit injection check: number of steering-vector marker writes since the
# last reset. 1-elem list so the hot _apply_steering hook mutates it without a
# `global`. Read+reset per rollout from the trainer via LLM.apply_model.
_NLA_STEER_APPLY_COUNT = [0]


def get_and_reset_steer_count() -> int:
    """Return marker position-writes since the last call, then reset to 0."""
    c = _NLA_STEER_APPLY_COUNT[0]
    _NLA_STEER_APPLY_COUNT[0] = 0
    return c


# NLA per-request steering coverage log (see patch_vllm_lens.py fix (4)):
# {log_key: {"covered": int, "applied": int, "positions": [int], "orphaned": int}}
_NLA_STEER_LOG: dict = {}


def _nla_steer_entry(log_key):
    return _NLA_STEER_LOG.setdefault(
        log_key, {"covered": 0, "applied": 0, "positions": [], "orphaned": 0})


def get_and_reset_steer_log() -> dict:
    """Return the per-request steering coverage log, then reset it."""
    log = {k: dict(v) for k, v in _NLA_STEER_LOG.items()}
    _NLA_STEER_LOG.clear()
    return log


def _find_steering_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[SteeringVector]:
    """Find all steering configs that apply to an internal request ID.

    Matches by ``"{external_id}-"`` prefix (async path: vLLM appends
    ``"-{random_suffix}"`` to external IDs) and by ``_steering_id``
    sentinel in ``extra_args`` (offline path).
    """
    # Offline path (NLA RL rollout): exact O(1) lookup — taken FIRST so the
    # hot per-layer hook does NOT scan the whole _steering_data dict per
    # request (that scan is O(num_running x num_total) per layer per decode
    # step and collapses rollout throughput at large batch).
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id is not None:
            return list(extension._steering_data.get(steering_id, []))
    # Async path (no _steering_id): match by external-id prefix.
    results: list[SteeringVector] = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    return results


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    norm_ref: torch.Tensor | None = None,
    log_key: str | None = None,
) -> None:
    """Apply all matching steering vectors to a token slice *in-place*.

    ``target`` is the (already-cloned) output tensor.  ``start``/``end``
    are batch-relative indices, ``abs_start`` is the absolute sequence
    position of the first token in ``target[start:end]``.

    ``norm_ref`` is the tensor whose per-position L2 norm anchors
    ``norm_match``. For models whose decoder layers return
    ``(hidden_states, residual)`` tuples (Qwen/Llama in vLLM), the TRUE
    residual stream is ``hidden_states + residual`` — norm-matching against
    ``hidden_states`` alone mis-scales the steering vector. Defaults to
    ``target`` for plain (non-tuple) layer outputs.
    """
    if norm_ref is None:
        norm_ref = target
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)  # (hidden,) or (n_pos, hidden)

        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (
                cfg.position_indices
                if cfg.position_indices is not None
                else list(range(vec.shape[0]))
            )
            abs_end = abs_start + n_tokens
            if log_key is not None and any(
                    abs_start <= _p < abs_end for _p in pos_indices):
                _nla_steer_entry(log_key)["covered"] += 1
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale
                _NLA_STEER_APPLY_COUNT[0] += 1  # NLA: count this marker write
                if log_key is not None:
                    _e = _nla_steer_entry(log_key)
                    _e["applied"] += 1
                    _e["positions"].append(int(abs_pos))


def _step_state(extension: HiddenStatesExtension, ctx: Any, attn_metadata: Any) -> dict:
    """Per-forward-pass state, built once at the first hooked layer and reused by every other layer.

    FAST PATH (11 Oct 2026). The old hook redid all of this on every layer of every step: a Python loop over all
    running requests (config lookup, list copy) x n_layers, plus 2-3 blocking ``.item()`` syncs per steered request
    at the steered layer. At ~260 running requests and 64 layers that is ~17k lookups and ~800 GPU syncs per decode
    step, though position-indexed steering (the NLA case) only ever writes during the prompt.
    Keyed by identity of the ForwardContext object, which we hold a reference to (so its id cannot be reused).
    """
    st = extension.__dict__.get("_fast_step")
    if st is not None and st["ctx"] is ctx:
        return st
    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    per_req_steering: list[list[SteeringVector]] = []
    per_req_log_key: list = []
    steer_layers: set[int] = set()
    capture: list[tuple[int, str, Any]] = []  # (i, req_id, output_residual_stream)
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (req_state.sampling_params.extra_args
                 if req_state and req_state.sampling_params else None)
        configs = _find_steering_configs(extension, req_id, extra)
        per_req_steering.append(configs)
        _sid = (extra or {}).get("_steering_id")
        per_req_log_key.append(_sid or req_id)
        if _sid is not None and not configs:
            _nla_steer_entry(_sid)["orphaned"] += 1  # once per forward, as before
        for cfg in configs:
            steer_layers.update(cfg.layer_index_map)
        if extra and extension._should_capture:
            ors = extra.get("output_residual_stream")
            if ors is not None:
                capture.append((i, req_id, ors))
    capture_all = any(not isinstance(o, list) for _, _, o in capture)
    capture_layers = set() if capture_all else {l for _, _, o in capture for l in o}
    query_start_loc = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            query_start_loc = getattr(_meta, "query_start_loc")
            break
    seq_lens: Any = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None and hasattr(attn_metadata, "values"):
        for _meta5 in attn_metadata.values():
            if getattr(_meta5, "seq_lens", None) is not None:
                seq_lens = _meta5.seq_lens
                break
    st = {"ctx": ctx, "num_reqs": num_reqs, "req_ids": req_ids, "steer": per_req_steering,
          "log_key": per_req_log_key, "steer_layers": steer_layers, "capture": capture,
          "capture_all": capture_all and bool(capture), "capture_layers": capture_layers,
          "qsl_t": query_start_loc, "sl_t": seq_lens, "qsl": None, "sl": None}
    extension._fast_step = st
    return st


def _cpu_positions(st: dict) -> None:
    """One sync per step for query_start_loc and seq_lens (instead of .item() per request)."""
    if st["qsl"] is None:
        st["qsl"] = st["qsl_t"].tolist()
        sl = st["sl_t"]
        st["sl"] = (sl.tolist() if isinstance(sl, torch.Tensor) else list(sl)) if sl is not None else None


def _needs_write(configs: list[SteeringVector], layer_idx: int, abs_start: int, abs_end: int) -> bool:
    """False only when every config on this layer is position-indexed and none of its positions is in this slice."""
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        vec = cfg.activations[cfg.layer_index_map[layer_idx]]
        if vec.dim() == 1:
            return True
        pos = cfg.position_indices if cfg.position_indices is not None else range(vec.shape[0])
        if any(abs_start <= p < abs_end for p in pos):
            return True
    return False


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic (fast path: see _step_state). Same results as the per-layer version."""
    if not is_forward_context_available():
        return None
    if extension.model_runner.input_batch.num_reqs == 0:
        return None
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    st = _step_state(extension, ctx, attn_metadata)
    do_steer = layer_idx in st["steer_layers"]
    do_capture = st["capture_all"] or layer_idx in st["capture_layers"]
    if not do_steer and not do_capture:
        return None  # the common case: 63 of 64 layers, every step
    if st["qsl_t"] is None:
        logger.warning("No attention metadata with query_start_loc found (keys: %s). Skipping hook for this step.",
                       list(attn_metadata.keys()))
        return None
    _cpu_positions(st)
    qsl, sl = st["qsl"], st["sl"]

    # --- Phase 2: apply steering (only requests with a write in this slice) ---
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    if do_steer:
        work = []
        for i, configs in enumerate(st["steer"]):
            if not configs:
                continue
            start, end = qsl[i], qsl[i + 1]
            abs_start = int(sl[i] - (end - start)) if sl is not None else 0
            if _needs_write(configs, layer_idx, abs_start, abs_start + end - start):
                work.append((i, start, end, abs_start))
        if work:
            if isinstance(output, tuple):
                modified_output = (output[0].clone(), output[1])
                target = modified_output[0]
                # (hidden_states, residual) layers: the true stream is the sum; norm_match is anchored to it
                norm_ref = output[0] + output[1] if output[1] is not None else output[0]
            else:
                modified_output = output.clone()
                target = modified_output
                norm_ref = target
            for i, start, end, abs_start in work:
                _apply_steering(st["steer"][i], layer_idx, target, start, end, abs_start, norm_ref,
                                log_key=st["log_key"][i])

    # --- Phase 3: capture activations (rank 0 only) ---
    if do_capture and st["capture"]:
        capture_src = modified_output if modified_output is not None else output
        if isinstance(capture_src, tuple):
            hidden_states = capture_src[0] + capture_src[1] if capture_src[1] is not None else capture_src[0]
        else:
            hidden_states = capture_src
        for i, req_id, ors in st["capture"]:
            if isinstance(ors, list) and layer_idx not in ors:
                continue
            activation = hidden_states[qsl[i]:qsl[i + 1]].cpu()
            layer_states = extension._captured_states.setdefault(req_id, {})
            layer_states.setdefault(layer_idx, []).append(activation)

    return modified_output


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        rank: int
        parallel_config: ParallelConfig

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Per-request steering configs:
    # key (external_req_id or _steering_id) → list of SteeringVector
    _steering_data: dict[str, list[SteeringVector]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args — otherwise
        ``@support_torch_compile`` would compile the forward graph and
        hooks won't fire.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared)
        self._captured_states = {}
        self._steering_data = {}

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layers = _get_layers(self.model_runner.model)
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            layer.register_forward_hook(_make_hook(self, layer_idx))

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store steering vectors for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``SteeringVector`` instances, validates layer indices
        against the model, moves activation tensors to GPU in the model's
        dtype, and stores them keyed by *key* (an external request ID or a
        synthetic ``_steering_id``).
        """
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)

        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype

        num_layers = len(_get_layers(self.model_runner.model))
        vectors: list[SteeringVector] = []

        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )

            vectors.append(
                sv.model_copy(
                    update={
                        "activations": sv.activations.to(device=device, dtype=dtype)
                    }
                )
            )

        self._steering_data[key] = vectors

    def set_steering_data_batch(self, payloads: dict[str, bytes]) -> None:
        """Batched set_steering_data: one rpc for a whole generate() call."""
        for key, pickled_data in payloads.items():
            self.set_steering_data(key, pickled_data)

    def clear_steering_data_batch(self, keys: list[str]) -> None:
        """Batched clear_steering_data."""
        for key in keys:
            self._steering_data.pop(key, None)

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                logger.debug("Cleared leaked activations for %s", req_id)

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle for safe ZMQ
        transport.

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                layer_dict = self._captured_states.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
                    torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
                ]
                stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
                    torch.stack(per_layer, dim=0)
                )
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "activations": {"residual_stream": stacked},
                        }
                    )
                )
        return None

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)
