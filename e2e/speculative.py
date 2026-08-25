"""PARD2-TI/TD speculative decoding for the fused dense QuaRot runtime.

The target always remains the W4A4KV4 fused runtime.  The official PARD2
drafter is loaded from a pinned local Hugging Face snapshot; TI deliberately
does not load ``warp_model.bin``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Iterable

import torch


def _normalized_hadamard_cpu(value: torch.Tensor) -> torch.Tensor:
    """Reference normalized Sylvester transform used to recover rotation metadata."""
    result = value.float().contiguous().clone()
    width, stride = result.shape[-1], 1
    if width <= 0 or width & (width - 1):
        raise ValueError("PARD2 target hidden width must be a power of two")
    while stride < width:
        pair = result.view(*result.shape[:-1], -1, 2, stride)
        left, right = pair[..., 0, :].clone(), pair[..., 1, :].clone()
        pair[..., 0, :], pair[..., 1, :] = left + right, left - right
        stride *= 2
    return result / math.sqrt(width)


def fold_td_projection_weight(weight: torch.Tensor, rotation_signs: torch.Tensor,
                              final_norm_weight: torch.Tensor,
                              hidden_size: int) -> torch.Tensor:
    """Fold QuaRot's ``D H`` basis restoration into the four TD weight blocks.

    PARD2 consumes row-vector features.  For rotated target activations
    ``x_rot = x D H``, the equivalent projection weights are
    ``W0' = W0 Gamma D H`` for the final normalized tap and
    ``Wi' = Wi D H`` for the remaining raw taps.  The transform is performed
    blockwise on CPU in FP32 and never constructs a dense Hadamard matrix.
    """
    hidden_size = int(hidden_size)
    if weight.ndim != 2 or weight.shape[1] != 4 * hidden_size:
        raise ValueError(
            "TD projection weight must have shape [output, 4 * hidden_size]")
    if hidden_size <= 0 or hidden_size & (hidden_size - 1):
        raise ValueError("TD hidden size must be a positive power of two")
    signs = rotation_signs.detach().cpu().float().reshape(-1)
    gamma = final_norm_weight.detach().cpu().float().reshape(-1)
    if signs.numel() != hidden_size:
        raise ValueError("rotation_signs width does not match TD hidden size")
    if gamma.numel() != hidden_size:
        raise ValueError("final_norm_weight width does not match TD hidden size")
    source = weight.detach().cpu().float()
    folded = []
    for index, block in enumerate(source.split(hidden_size, dim=1)):
        multiplier = signs * gamma if index == 0 else signs
        folded.append(_normalized_hadamard_cpu(block * multiplier))
    return torch.cat(folded, dim=1)


def _checkpoint_tensor(snapshot, key, rows=None):
    """Read one tensor (or a few leading rows) without materializing a model."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filename = weight_map.get(key)
        if filename is None:
            raise KeyError(f"{key} is absent from {index_path}")
    else:
        filename = "model.safetensors"
    with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        return (tensor[:rows] if rows is not None else tensor[:]).clone()


def load_td_target_basis(target, source_snapshot):
    """Recover x'=x D H metadata and the pre-fusion final RMSNorm scale."""
    saved_signs = getattr(target.config, "quarot_rotation_signs", None)
    saved_norm = getattr(target.config, "quarot_final_norm_weight", None)
    if saved_signs is not None and saved_norm is not None:
        if len(saved_signs) != target.config.hidden_size:
            raise ValueError("invalid quarot_rotation_signs checkpoint metadata")
        if len(saved_norm) != target.config.hidden_size:
            raise ValueError("invalid quarot_final_norm_weight checkpoint metadata")
        return torch.tensor(saved_signs, dtype=torch.float32), torch.tensor(
            saved_norm, dtype=torch.float32)
    source_rows = _checkpoint_tensor(
        source_snapshot, "model.embed_tokens.weight", rows=8).float()
    rotated_rows = target.get_input_embeddings().weight[:8].detach().cpu().float()
    restored_columns = _normalized_hadamard_cpu(rotated_rows)
    signs = torch.sign((restored_columns * source_rows).sum(dim=0))
    signs[signs == 0] = 1
    reconstructed = _normalized_hadamard_cpu(source_rows * signs)
    rmse = (reconstructed - rotated_rows).square().mean().sqrt().item()
    if rmse > 0.003:
        raise RuntimeError(
            f"cannot recover fused target rotation signs (embedding RMSE {rmse:.5f})")
    final_norm = _checkpoint_tensor(source_snapshot, "model.norm.weight").float()
    return signs, final_norm


@dataclass(frozen=True)
class Pard2Spec:
    model_id: str = "amd/PARD2-Qwen3-8B"
    revision: str = "67a1516c8f6fc145cda99916799a0cbb3a4af135"
    target_model_id: str = "Qwen/Qwen3-8B"
    target_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    upstream_commit: str = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
    pard_token: int = 151670
    draft_k: int = 15
    target_layers: tuple[int, ...] = (-1, -8, -16, -24)
    target_dim: int = 16384
    projection_scale: float = 0.02

    def validate(self, target_config, draft_config) -> None:
        if getattr(target_config, "model_type", None) not in ("qwen3", "qwen3_quarot"):
            raise ValueError("selected PARD2 target must be a dense Qwen3 runtime")
        if int(target_config.hidden_size) != 4096 or int(target_config.num_hidden_layers) != 36:
            raise ValueError("PARD2-Qwen3-8B requires a 36-layer, 4096-wide target")
        if int(target_config.vocab_size) != int(draft_config.vocab_size):
            raise ValueError("target and PARD2 vocabularies differ")
        expected = {
            "pard_token": self.pard_token,
            "pard2_target_dim": self.target_dim,
        }
        for name, value in expected.items():
            if int(getattr(draft_config, name, -1)) != value:
                raise ValueError(f"draft {name} does not match pinned PARD2 spec")
        if tuple(draft_config.pard2_target_layers) != self.target_layers:
            raise ValueError("draft hidden taps do not match pinned PARD2 spec")
        if abs(float(draft_config.pard2_scale) - self.projection_scale) > 1e-12:
            raise ValueError("draft projection scale does not match pinned PARD2 spec")


@dataclass
class GenerationResult:
    output_ids: list[int]
    ttft_ms: float
    steady_decode_ms: float
    total_ms: float
    target_forwards: int
    draft_forwards: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    verifier_steps: int
    emitted_tokens_per_step: list[int]
    accept_length_by_step: list[int]
    stage_ms: dict[str, float] = field(default_factory=dict)
    peak_vram_bytes: int = 0
    proposal_lengths_by_step: list[int] = field(default_factory=list)

    def metrics(self) -> dict:
        generated = len(self.output_ids)
        first = self.emitted_tokens_per_step[0] if self.emitted_tokens_per_step else 0
        steady = max(generated - first, 0)
        verified = self.proposal_lengths_by_step
        if not verified and self.verifier_steps:
            average = self.proposed_draft_tokens // self.verifier_steps
            verified = [average] * self.verifier_steps
        accepted = [max(length - 1, 0) for length in self.accept_length_by_step]
        reached = sum(min(count + 1, width)
                      for count, width in zip(accepted, verified))
        conditional_by_position = [
            (sum(count >= position for count in accepted)
             / sum(width >= position and count >= position - 1
                   for count, width in zip(accepted, verified)))
            if any(width >= position and count >= position - 1
                   for count, width in zip(accepted, verified)) else 0.0
            for position in range(1, max(verified, default=0) + 1)]
        return {
            **asdict(self),
            "generated_tokens": generated,
            "steady_tokens_per_s": (
                1000.0 * steady / self.steady_decode_ms
                if steady and self.steady_decode_ms else 0.0),
            "end_to_end_tokens_per_s": (
                1000.0 * generated / self.total_ms if self.total_ms else 0.0),
            "mean_accept_length": (
                sum(self.accept_length_by_step) / len(self.accept_length_by_step)
                if self.accept_length_by_step else 1.0),
            "draft_acceptance": (
                self.accepted_draft_tokens / self.proposed_draft_tokens
                if self.proposed_draft_tokens else 0.0),
            "conditional_acceptance": (
                self.accepted_draft_tokens / reached if reached else 0.0),
            "conditional_acceptance_by_position": conditional_by_position,
        }


def greedy_accept(candidates: torch.Tensor, predictions: torch.Tensor) -> int:
    """Return the longest greedy candidate prefix accepted by the target."""
    if candidates.ndim != 2 or predictions.ndim != 2:
        raise ValueError("candidates and predictions must have shape [batch, tokens]")
    if candidates.shape[0] != 1 or predictions.shape[0] != 1:
        raise ValueError("PARD2 v1 supports batch size one")
    if predictions.shape[1] < candidates.shape[1] + 1:
        raise ValueError("predictions must include the correction/bonus token")
    equal = candidates[0].eq(predictions[0, : candidates.shape[1]])
    mismatch = torch.nonzero(~equal, as_tuple=False)
    return candidates.shape[1] if mismatch.numel() == 0 else int(mismatch[0, 0])


class SelectedHiddenCollector:
    """Collect only the four PARD2 layer outputs, never a full hidden tuple."""

    def __init__(self, target, taps: Iterable[int], rotation_signs=None,
                 final_norm_weight=None, cache_basis=False,
                 folded_basis=False):
        layers = target.model.layers
        count = len(layers)
        self.indices = tuple(count + int(tap) for tap in taps)
        self.values = {}
        self.rotation_signs = rotation_signs
        self.final_norm_weight = final_norm_weight
        self.folded_basis = bool(folded_basis)
        self.basis_cached = False
        self.rms_norm_eps = float(
            getattr(getattr(target, "config", None), "rms_norm_eps", 1e-6))
        self.handles = []
        for position, index in enumerate(self.indices):
            if not 0 <= index < count:
                raise ValueError(f"hidden tap {index} is outside {count} layers")
            module = (target.model.norm
                      if self.folded_basis and position == 0 else layers[index])
            self.handles.append(module.register_forward_hook(self._hook(index)))
        if cache_basis and rotation_signs is not None and not self.folded_basis:
            parameter = next(target.parameters())
            self.cache_basis(parameter.device, parameter.dtype)

    def cache_basis(self, device, dtype):
        """Move the TD basis to the target device once, outside decode steps."""
        if self.rotation_signs is None:
            return
        if self.final_norm_weight is None:
            raise ValueError("final_norm_weight is required with rotation_signs")
        self.rotation_signs = self.rotation_signs.to(device=device, dtype=dtype)
        self.final_norm_weight = self.final_norm_weight.to(
            device=device, dtype=dtype)
        self.basis_cached = True

    def _hook(self, index):
        def capture(_module, _inputs, output):
            self.values[index] = output[0] if isinstance(output, tuple) else output
        return capture

    def reset(self):
        self.values.clear()

    def features(self, rows=None):
        missing = [index for index in self.indices if index not in self.values]
        if missing:
            raise RuntimeError(f"selected hidden taps were not produced: {missing}")
        values = [self.values[index] for index in self.indices]
        if rows is not None:
            values = [value[:, rows] for value in values]
        if self.folded_basis:
            return torch.cat(values, dim=-1)
        if self.rotation_signs is None:
            return torch.cat(values, dim=-1)
        from quarot.transformers.kv_cache import matmul_had_HIP

        signs = (self.rotation_signs if self.basis_cached else
                 self.rotation_signs.to(
                     device=values[0].device, dtype=values[0].dtype))
        restored = [matmul_had_HIP(value, value.dtype) * signs for value in values]
        final = restored[0]
        dtype = final.dtype
        final = final.float()
        final = final * torch.rsqrt(
            final.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps)
        norm_weight = (self.final_norm_weight if self.basis_cached else
                       self.final_norm_weight.to(
                           device=final.device, dtype=dtype))
        final = final.to(dtype) * norm_weight
        return torch.cat([final, *restored[1:]], dim=-1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TargetFeatEmbedWarp(torch.nn.Module):
    def __init__(self, base_model, target_dim, scale, proj_bias, calibration=None):
        super().__init__()
        self.base_model = base_model
        self.target_proj = torch.nn.Linear(
            target_dim, base_model.config.hidden_size, bias=proj_bias)
        self.scale = float(scale)
        self.config = base_model.config
        calibration = calibration or {}
        for name in ("raw_scale", "raw_bias", "projected_scale", "projected_bias"):
            self.register_buffer(name, calibration.get(name), persistent=False)
        self._td_basis_folded = False

    @property
    def td_basis_folded(self):
        return self._td_basis_folded

    def fold_target_basis_(self, rotation_signs, final_norm_weight):
        """Install an in-memory folded projection; the checkpoint is untouched."""
        if self._td_basis_folded:
            raise RuntimeError("TD target projection basis is already folded")
        if self.raw_scale is not None or self.raw_bias is not None:
            raise ValueError(
                "TD basis folding does not yet support raw feature calibration; "
                "disable --td-basis-fold or omit raw calibration")
        folded = fold_td_projection_weight(
            self.target_proj.weight, rotation_signs, final_norm_weight,
            self.target_proj.in_features // 4)
        with torch.no_grad():
            self.target_proj.weight.copy_(
                folded.to(dtype=self.target_proj.weight.dtype))
        self._td_basis_folded = True
        return self

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def project_features(self, target_feat):
        """Apply calibration and projection before replicated PARD mask rows."""
        features = target_feat.to(self.target_proj.weight.dtype)
        if self.raw_scale is not None:
            features = features * self.raw_scale.to(features.dtype) + self.raw_bias.to(features.dtype)
        projected = self.target_proj(features)
        if self.projected_scale is not None:
            projected = (projected * self.projected_scale.to(projected.dtype)
                         + self.projected_bias.to(projected.dtype))
        return projected

    def forward(self, input_ids=None, target_feat=None,
                projected_target_feat=None, **kwargs):
        if (target_feat is None) == (projected_target_feat is None):
            raise ValueError(
                "provide exactly one of target_feat or projected_target_feat")
        embeds = self.base_model.get_input_embeddings()(input_ids)
        projected = (self.project_features(target_feat)
                     if projected_target_feat is None
                     else projected_target_feat.to(embeds.dtype))
        return self.base_model(inputs_embeds=embeds + projected * self.scale, **kwargs)


class AdaptiveK:
    """Phase-2 confidence/EMA selector; disabled unless explicitly requested."""

    choices = (8, 12, 15)

    def __init__(self, low=0.55, high=0.80, decay=0.9):
        self.low, self.high, self.decay = float(low), float(high), float(decay)
        self.ema = 1.0

    def choose(self):
        return 8 if self.ema < self.low else (12 if self.ema < self.high else 15)

    def update(self, accepted, proposed):
        observed = accepted / proposed if proposed else 0.0
        self.ema = self.decay * self.ema + (1.0 - self.decay) * observed


class _StageTimer:
    def __init__(self):
        self.events = []

    def record(self, name, fn):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        value = fn()
        end.record()
        self.events.append((name, start, end))
        return value

    def totals(self):
        torch.cuda.synchronize()
        result = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end)
        return result


class _ExactSmallChunkRows(torch.nn.Module):
    """Run a terminal module with the decode (M=1) numerical contract."""

    def __init__(self, module, max_rows):
        super().__init__()
        self.module = module
        self.max_rows = int(max_rows)

    def forward(self, value):
        rows = value.shape[-2]
        if rows <= 1 or rows > self.max_rows:
            return self.module(value)
        outputs = [
            self.module(value[..., index:index + 1, :])
            for index in range(rows)]
        return torch.cat(outputs, dim=-2)


class _RowIndependentRMSNorm(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mean_dim = module.mean_dim
        self.eps = module.eps

    def forward(self, value):
        if value.is_cuda and value.dtype == torch.float16:
            import quarot
            return quarot._HIP.rms_norm_rows(
                value.contiguous(), self.mean_dim, self.eps)
        return self.module(value)


class _RowIndependentRMSNormQuant(torch.nn.Module):
    """Fuse exact row-independent RMSNorm with symmetric INT4 packing."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mean_dim = module.mean_dim
        self.eps = module.eps

    def forward(self, value):
        if not value.is_cuda or value.dtype != torch.float16:
            raise RuntimeError(
                "fused RMSNorm+INT4 quantization requires CUDA FP16 input")
        import quarot
        packed, scales = quarot._HIP.rms_norm_quant_i4_rows(
            value.contiguous(), self.mean_dim, self.eps)
        return quarot.PackedQuantizedTensor(
            packed, scales, logical_shape=value.shape)


class FusedPardRuntime:
    def __init__(self, *, mode, target, draft, tokenizer, spec=Pard2Spec(),
                 max_cache_len=4096, page_size=128, compile_mode="max-autotune",
                 ignore_eos=False, adaptive_k=None, native_gqa=True,
                 fused_decode_append=True, exact_row_norm=True,
                 rowwise_lm_head=False, exact_small_chunk=None,
                 td_target_basis=None, td_cache_basis=True,
                 td_lazy_features=False, td_unique_projection=False,
                 td_basis_fold=False, fused_norm_quant=False):
        if mode not in ("ar", "pard2-ti", "pard2-td"):
            raise ValueError("mode must be ar, pard2-ti, or pard2-td")
        self.mode, self.target, self.draft, self.tokenizer = mode, target, draft, tokenizer
        self.spec, self.max_cache_len, self.page_size = spec, max_cache_len, page_size
        self.ignore_eos, self.adaptive_k = ignore_eos, adaptive_k
        self.native_gqa = bool(native_gqa)
        self.fused_decode_append = bool(fused_decode_append)
        self.td_cache_basis = bool(td_cache_basis)
        self.td_lazy_features = bool(td_lazy_features)
        self.td_unique_projection = bool(td_unique_projection)
        self.td_basis_fold = bool(td_basis_fold)
        if self.td_basis_fold and mode != "pard2-td":
            raise ValueError("TD basis folding is valid only in pard2-td mode")
        # exact_small_chunk is the legacy coupled switch. Keep accepting it
        # when explicitly supplied, but use the independent defaults otherwise.
        # Exact row-wise RMSNorm is required for cache/token parity, while a
        # batched LM-head is parity-safe and substantially cheaper.
        if exact_small_chunk is not None:
            exact_row_norm = bool(exact_small_chunk)
            rowwise_lm_head = bool(exact_small_chunk)
        self.exact_row_norm = bool(exact_row_norm)
        self.rowwise_lm_head = bool(rowwise_lm_head)
        self.fused_norm_quant = bool(fused_norm_quant)
        self.exact_small_chunk = (
            self.exact_row_norm and self.rowwise_lm_head)
        if self.fused_norm_quant and not self.exact_row_norm:
            raise ValueError("fused_norm_quant requires exact_row_norm=True")
        if (self.fused_norm_quant and
                getattr(target, "cache_dtype", None) != "int4"):
            raise ValueError(
                "fused_norm_quant requires a quantized target "
                "(cache_dtype='int4')")
        if (self.exact_row_norm and hasattr(target, "model")
                and hasattr(target.model, "norm")):
            if not isinstance(target.model.norm, _RowIndependentRMSNorm):
                target.model.norm = _RowIndependentRMSNorm(target.model.norm)
            for layer in target.model.layers:
                wrapper = (_RowIndependentRMSNormQuant
                           if self.fused_norm_quant else _RowIndependentRMSNorm)
                if not isinstance(layer.input_layernorm, wrapper):
                    layer.input_layernorm = wrapper(layer.input_layernorm)
                if not isinstance(layer.post_attention_layernorm, wrapper):
                    layer.post_attention_layernorm = wrapper(
                        layer.post_attention_layernorm)
        if self.rowwise_lm_head and hasattr(target, "lm_head"):
            if not isinstance(target.lm_head, _ExactSmallChunkRows):
                target.lm_head = _ExactSmallChunkRows(
                    target.lm_head, spec.draft_k + 1)
        eos = target.config.eos_token_id
        self.eos_ids = {int(x) for x in (eos if isinstance(eos, list) else [eos]) if x is not None}
        basis = td_target_basis or (None, None)
        self.collector = (SelectedHiddenCollector(
            target, spec.target_layers,
            None if self.td_basis_fold else basis[0],
            None if self.td_basis_fold else basis[1],
            self.td_cache_basis, folded_basis=self.td_basis_fold)
                          if mode == "pard2-td" else None)
        self.draft_forward = draft.forward if draft is not None else None
        if draft is not None and compile_mode != "eager":
            # Proposal decoding deliberately specializes the drafter for the
            # bounded sequence lengths draft_k..(2 * draft_k - 1). Dynamo's
            # default recompile limit (8) is smaller than PARD2's 15 shapes.
            import torch._dynamo
            required_recompiles = spec.draft_k + 1
            if torch._dynamo.config.recompile_limit < required_recompiles:
                torch._dynamo.config.recompile_limit = required_recompiles
            self.draft_forward = torch.compile(
                draft.forward, mode=compile_mode, fullgraph=True, dynamic=False)

    def close(self):
        if self.collector is not None:
            self.collector.close()

    def _target_cache(self):
        return self.target.build_cache(1, self.page_size, self.max_cache_len,
            native_gqa=self.native_gqa,
            fused_decode_append=self.fused_decode_append)


    def _draft_cache(self):
        if getattr(self.draft.config, "model_type", "").endswith("_quarot"):
            return self.draft.build_cache(
                1, self.page_size, self.max_cache_len,
                native_gqa=self.native_gqa,
                fused_decode_append=self.fused_decode_append)
        from transformers import StaticCache
        return StaticCache(config=self.draft.config, max_batch_size=1,
                           max_cache_len=self.max_cache_len,
                           device=next(self.draft.parameters()).device,
                           dtype=next(self.draft.parameters()).dtype)

    def _target_call(self, ids, cache, positions, materialize_features=True):
        if self.collector is not None:
            self.collector.reset()
        output = self.target(input_ids=ids, past_key_values=cache,
                             cache_position=positions, use_cache=True,
                             attention_mask=None, return_dict=True,
                             output_hidden_states=False)
        features = (self.collector.features()
                    if self.collector is not None and materialize_features else None)
        return output, features

    def _project_td_features(self, features, mask_rows=0):
        """Project unique real-token rows before expanding the final mask row."""
        projected = self.draft.project_features(features)
        if not mask_rows:
            return projected
        padding = projected[:, -1:].expand(-1, mask_rows, -1)
        return torch.cat((projected, padding), dim=1)

    def _accepted_td_features(self, prefill_features, emitted_count,
                              first_round, timer):
        """Materialize exactly the rows consumed by the next draft call."""
        rows = emitted_count - (1 if first_round else 0)
        restored = (timer.record(
            "td_feature_restore",
            lambda: self.collector.features(slice(0, rows)))
            if rows else None)
        if first_round:
            prompt_tail = prefill_features[:, -1:]
            return (prompt_tail if restored is None else
                    torch.cat((prompt_tail, restored), dim=1))
        if restored is None:
            raise RuntimeError(
                "a non-first speculative round must emit at least one token")
        return restored

    def _stop(self, emitted):
        return bool(emitted and not self.ignore_eos and emitted[-1] in self.eos_ids)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=256):
        if input_ids.shape[0] != 1:
            raise ValueError("PARD2 v1 supports batch size one")
        if input_ids.shape[1] + max_new_tokens + self.spec.draft_k > self.max_cache_len:
            raise ValueError("generation exceeds configured cache capacity")
        return self._generate_ar(input_ids, max_new_tokens) if self.mode == "ar" else self._generate_spec(input_ids, max_new_tokens)

    def _generate_ar(self, input_ids, max_new_tokens):
        cache, current, generated = self._target_cache(), input_ids, []
        forwards, first_at = 0, None
        timer = _StageTimer()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
        while len(generated) < max_new_tokens:
            pos = torch.arange(cache.length, cache.length + current.shape[1], device=current.device)
            output, _ = timer.record("target", lambda: self._target_call(current, cache, pos))
            cache = output.past_key_values
            forwards += 1
            token = int(output.logits[:, -1].argmax(-1))
            generated.append(token)
            if first_at is None:
                torch.cuda.synchronize(); first_at = time.perf_counter()
            if self._stop(generated): break
            current = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
        stages = timer.totals(); ended = time.perf_counter(); first_at = first_at or ended
        return GenerationResult(generated, (first_at-started)*1000, (ended-first_at)*1000,
            (ended-started)*1000, forwards, 0, 0, 0, forwards, [1]*len(generated),
            [1]*len(generated), stages, torch.cuda.max_memory_allocated())

    def _generate_spec(self, input_ids, max_new_tokens):
        target_cache, draft_cache = self._target_cache(), self._draft_cache()
        target_input, draft_input = None, input_ids
        draft_cache_len = 0
        draft_features = None
        target_forwards = draft_forwards = proposed = accepted_total = 0
        generated, emitted_steps, accept_lengths, proposal_lengths = [], [], [], []
        timer, first_at = _StageTimer(), None
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()

        # Keep target prefill identical to AR.  A combined unquantized
        # "prompt + candidates" prefill can disagree with sequential KV4 AR.
        # The prefill logit supplies the first pending prediction; the first
        # verifier contains k candidates, then later rounds contain pending+k.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        prefill_out, prefill_features = timer.record(
            "target_feature_prefill" if self.mode == "pard2-td" else "target_prefill",
            lambda: self._target_call(input_ids, target_cache, positions))
        target_cache = prefill_out.past_key_values
        pending_prediction = prefill_out.logits[:, -1:].argmax(-1)
        target_forwards += 1
        first_round = True

        if self.mode == "pard2-td":
            features = prefill_features
            zero = torch.zeros_like(features[:, :1])
            draft_features = torch.cat((zero, features[:, :-1]), dim=1)

        # Keep arbitrary prompt lengths out of the compiled proposal graph.
        # Prefill all but the final prompt token eagerly; the first PARD call
        # is then the reusable fixed M=15 shape (one real token + 14 masks).
        if input_ids.shape[1] > 1:
            prefix = input_ids[:, :-1]
            prefix_pos = torch.arange(prefix.shape[1], device=input_ids.device)
            prefix_kwargs = {}
            if self.mode == "pard2-td":
                prefix_features = draft_features[:, :-1]
                if self.td_unique_projection:
                    prefix_kwargs["projected_target_feat"] = timer.record(
                        "td_feature_project_prefill",
                        lambda: self._project_td_features(prefix_features))
                else:
                    prefix_kwargs["target_feat"] = prefix_features
            timer.record("draft_prefill", lambda: self.draft(
                input_ids=prefix, past_key_values=draft_cache,
                cache_position=prefix_pos, use_cache=True, attention_mask=None,
                return_dict=True, **prefix_kwargs))
            draft_forwards += 1
            draft_cache_len = prefix.shape[1]
            draft_input = input_ids[:, -1:]
            if self.mode == "pard2-td":
                draft_features = draft_features[:, -1:]

        while len(generated) < max_new_tokens:
            k = self.adaptive_k.choose() if self.adaptive_k is not None else self.spec.draft_k
            masks = torch.full((1, k-1), self.spec.pard_token,
                               device=input_ids.device, dtype=input_ids.dtype)
            draft_ids = torch.cat((draft_input, masks), dim=1)
            draft_pos = torch.arange(draft_cache_len, draft_cache_len + draft_ids.shape[1], device=input_ids.device)
            if hasattr(draft_cache, "length"):
                draft_cache.length = draft_cache_len
            kwargs = {}
            if self.mode == "pard2-td":
                if self.td_unique_projection:
                    kwargs["projected_target_feat"] = timer.record(
                        "td_feature_project",
                        lambda: self._project_td_features(draft_features, k - 1))
                else:
                    pad = draft_features[:, -1:].expand(-1, k-1, -1)
                    kwargs["target_feat"] = torch.cat((draft_features, pad), dim=1)
            draft_out = timer.record("draft", lambda: self.draft_forward(
                input_ids=draft_ids, past_key_values=draft_cache,
                cache_position=draft_pos, use_cache=True, attention_mask=None,
                return_dict=True, **kwargs))
            draft_forwards += 1; draft_cache_len += draft_ids.shape[1]
            candidates = draft_out.logits[:, -k:].argmax(-1)
            proposed += k
            proposal_lengths.append(k)

            verify_ids = (candidates if first_round else
                          torch.cat((target_input, candidates), dim=1))
            target_pos = torch.arange(target_cache.length,
                                      target_cache.length + verify_ids.shape[1],
                                      device=input_ids.device)
            base_length = target_cache.length
            target_out, new_features = timer.record("target_verify",
                lambda: self._target_call(verify_ids, target_cache, target_pos,
                    materialize_features=not self.td_lazy_features))
            if first_round:
                predictions = torch.cat((
                    pending_prediction,
                    target_out.logits[:, -k:].argmax(-1)), dim=1)
            else:
                predictions = target_out.logits[:, -(k+1):].argmax(-1)
            accepted = greedy_accept(candidates, predictions)
            # Transformers may return a shallow cache wrapper instead of
            # mutating the object passed by the caller.  Commit on the returned
            # physical cache; rejected slots remain stale and are overwritten.
            target_cache = target_out.past_key_values
            target_cache.length = base_length + (0 if first_round else 1) + accepted
            target_forwards += 1; accepted_total += accepted
            if self.adaptive_k is not None: self.adaptive_k.update(accepted, k)
            accept_lengths.append(accepted + 1)
            emitted = [int(x) for x in predictions[0, :accepted+1].tolist()]
            emitted = emitted[:max_new_tokens-len(generated)]
            if not self.ignore_eos:
                for i, token in enumerate(emitted):
                    if token in self.eos_ids:
                        emitted = emitted[:i+1]; break
            generated.extend(emitted); emitted_steps.append(len(emitted))
            if self.mode == "pard2-td":
                # Target features are shifted left by one token.  First-round
                # candidate[0] uses the final prompt hidden state; later rounds
                # use the pending correction hidden state.
                if self.td_lazy_features:
                    draft_features = self._accepted_td_features(
                        prefill_features, len(emitted), first_round, timer)
                else:
                    source = (
                        torch.cat((prefill_features[:, -1:], new_features), dim=1)
                        if first_round else new_features)
                    draft_features = source[:, :len(emitted)]
            if first_at is None:
                torch.cuda.synchronize(); first_at = time.perf_counter()
            if self._stop(generated) or len(generated) >= max_new_tokens: break

            target_input = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=input_ids.dtype)
            draft_cache_len = max(0, draft_cache_len - (k-1))
            if hasattr(draft_cache, "length"):
                draft_cache.length = draft_cache_len
            draft_input = torch.tensor([emitted], device=input_ids.device, dtype=input_ids.dtype)
            first_round = False

        stages = timer.totals(); ended = time.perf_counter(); first_at = first_at or ended
        return GenerationResult(generated, (first_at-started)*1000, (ended-first_at)*1000,
            (ended-started)*1000, target_forwards, draft_forwards, proposed,
            accepted_total, len(emitted_steps), emitted_steps, accept_lengths,
            stages, torch.cuda.max_memory_allocated(), proposal_lengths)


def load_runtime(*, mode, target_checkpoint, draft_snapshot, tokenizer_path,
                 max_cache_len=4096, page_size=128, compile_mode="max-autotune",
                 ignore_eos=False, calibration_path=None, quantized_draft=None,
                 adaptive_k=False, native_gqa=True,
                 fused_decode_append=True, exact_row_norm=True,
                 rowwise_lm_head=False, exact_small_chunk=None,
                 td_cache_basis=True, td_lazy_features=False,
                 td_unique_projection=False, td_basis_fold=False,
                 fused_norm_quant=False):
    """Load a pinned local runtime without network access or checkpoint copies."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from e2e.model_registry import runtime_types

    spec = Pard2Spec()
    config_cls, target_cls, _ = runtime_types(target_checkpoint, local_files_only=True)
    target_config = config_cls.from_pretrained(target_checkpoint, local_files_only=True,
                                               attn_implementation="flash_attention_2")
    target = target_cls.from_pretrained(target_checkpoint, config=target_config,
        torch_dtype=torch.float16, local_files_only=True).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    draft = None
    td_target_basis = None
    if mode != "ar":
        draft_config = AutoConfig.from_pretrained(draft_snapshot, local_files_only=True)
        spec.validate(target_config, draft_config)
        if quantized_draft:
            draft_config_cls, draft_cls, _ = runtime_types(
                quantized_draft, local_files_only=True)
            quantized_draft_config = draft_config_cls.from_pretrained(
                quantized_draft, local_files_only=True,
                attn_implementation="flash_attention_2")
            draft = draft_cls.from_pretrained(
                quantized_draft, config=quantized_draft_config,
                torch_dtype=torch.float16, local_files_only=True)
            draft_dtype = torch.float16
        else:
            draft = AutoModelForCausalLM.from_pretrained(
                draft_snapshot, torch_dtype=torch.bfloat16, local_files_only=True,
                attn_implementation="eager")
            draft_dtype = torch.bfloat16
        if mode == "pard2-td":
            td_target_basis = load_td_target_basis(target, tokenizer_path)
            calibration = (torch.load(calibration_path, map_location="cpu", weights_only=True)
                           if calibration_path else None)
            draft = TargetFeatEmbedWarp(draft, spec.target_dim, spec.projection_scale,
                                        bool(draft_config.pard2_proj_bias), calibration)
            state = torch.load(Path(draft_snapshot)/"warp_model.bin",
                               map_location="cpu", weights_only=True)
            draft.target_proj.load_state_dict({
                key.removeprefix("target_proj."): value for key, value in state.items()
                if key.startswith("target_proj.")})
            if td_basis_fold:
                draft.fold_target_basis_(*td_target_basis)
        draft = draft.eval().to("cuda", dtype=draft_dtype)
    selector = AdaptiveK() if adaptive_k else None
    return FusedPardRuntime(mode=mode, target=target, draft=draft,
        tokenizer=tokenizer, spec=spec, max_cache_len=max_cache_len,
        page_size=page_size, compile_mode=compile_mode, ignore_eos=ignore_eos,
        adaptive_k=selector, native_gqa=native_gqa,
        fused_decode_append=fused_decode_append,
        exact_row_norm=exact_row_norm,
        rowwise_lm_head=rowwise_lm_head,
        exact_small_chunk=exact_small_chunk,
        td_target_basis=td_target_basis,
        td_cache_basis=td_cache_basis,
        td_lazy_features=td_lazy_features,
        td_unique_projection=td_unique_projection,
        td_basis_fold=td_basis_fold,
        fused_norm_quant=fused_norm_quant)
