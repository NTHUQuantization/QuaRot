from __future__ import annotations

import time
from dataclasses import dataclass

from .measurement import gpu_snapshot, parameter_bytes


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
    memory: list[dict]

    def metrics(self) -> dict:
        tokens = len(self.output_ids)
        first_step_tokens = self.emitted_tokens_per_step[0] if self.emitted_tokens_per_step else 0
        steady_tokens = max(tokens - first_step_tokens, 0)
        return {
            "generated_tokens": tokens,
            "first_step_tokens": first_step_tokens,
            "steady_generated_tokens": steady_tokens,
            "ttft_ms": self.ttft_ms,
            "steady_decode_ms": self.steady_decode_ms,
            "steady_ms_per_token": self.steady_decode_ms / steady_tokens if steady_tokens else 0.0,
            "steady_tokens_per_s": 1000.0 * steady_tokens / self.steady_decode_ms if self.steady_decode_ms else 0.0,
            "total_ms": self.total_ms,
            "end_to_end_tokens_per_s": 1000.0 * tokens / self.total_ms if self.total_ms else 0.0,
            "target_forwards": self.target_forwards,
            "draft_forwards": self.draft_forwards,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "draft_acceptance_rate": (
                self.accepted_draft_tokens / self.proposed_draft_tokens
                if self.proposed_draft_tokens
                else 0.0
            ),
            "verifier_steps": self.verifier_steps,
            "mean_emitted_tokens_per_step": (
                sum(self.emitted_tokens_per_step) / len(self.emitted_tokens_per_step)
                if self.emitted_tokens_per_step
                else 1.0
            ),
        }


class TargetFeatEmbedWarp:
    """Constructed lazily so importing static tests does not require PyTorch."""

    @staticmethod
    def build(nn, base_model, target_dim: int, scale: float, proj_bias: bool):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = base_model
                self.scale = float(scale)
                self.target_proj = nn.Linear(target_dim, base_model.config.hidden_size, bias=proj_bias)
                self.config = base_model.config

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.base_model, name)

            def forward(self, input_ids=None, inputs_embeds=None, target_feat=None, **kwargs):
                if target_feat is None:
                    raise ValueError("target_feat is required by PARD2 target-dependent mode")
                embeds = self.base_model.get_input_embeddings()(input_ids)
                projected = self.target_proj(target_feat.to(embeds.device, embeds.dtype)) * self.scale
                return self.base_model(input_ids=None, inputs_embeds=embeds + projected, **kwargs)

        return Wrapper()


class PardRuntime:
    def __init__(
        self,
        *,
        torch,
        mode: str,
        tokenizer,
        target,
        draft,
        target_config,
        draft_config,
        draft_k: int,
        max_cache_len: int,
        compile_mode: str,
        load_memory: list[dict],
        stop_on_eos: bool = False,
    ):
        self.torch = torch
        self.mode = mode
        self.tokenizer = tokenizer
        self.target = target
        self.draft = draft
        self.target_config = target_config
        self.draft_config = draft_config
        self.draft_k = draft_k
        self.max_cache_len = max_cache_len
        self.load_memory = load_memory
        self.stop_on_eos = stop_on_eos
        self.eos_ids = target_config.eos_token_id
        if not isinstance(self.eos_ids, (list, tuple, set)):
            self.eos_ids = [self.eos_ids]
        self.eos_ids = {int(x) for x in self.eos_ids if x is not None}
        self.target_forward = target.forward
        self.draft_forward = draft.forward if draft is not None else None
        if compile_mode != "eager":
            self.target_forward = torch.compile(target.forward, mode=compile_mode, fullgraph=True, dynamic=False)
            if draft is not None:
                self.draft_forward = torch.compile(draft.forward, mode=compile_mode, fullgraph=True)

    def _new_cache(self, model):
        from transformers import StaticCache

        return StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=self.max_cache_len,
            device=model.device,
            dtype=model.dtype,
        )

    def _truncate_for_eos(self, token_ids: list[int], remaining: int) -> list[int]:
        out = token_ids[:remaining]
        if not self.stop_on_eos:
            return out
        for idx, token_id in enumerate(out):
            if token_id in self.eos_ids:
                return out[: idx + 1]
        return out

    def generate(self, input_ids, max_new_tokens: int, capture_memory: bool = True) -> GenerationResult:
        with self.torch.inference_mode():
            if self.mode == "ar":
                return self._generate_ar(input_ids, max_new_tokens, capture_memory)
            return self._generate_speculative(input_ids, max_new_tokens, capture_memory)

    def _generate_ar(self, input_ids, max_new_tokens: int, capture_memory: bool) -> GenerationResult:
        torch = self.torch
        cache = self._new_cache(self.target)
        memory = [gpu_snapshot(torch, "cache_initialized")] if capture_memory else []
        torch.cuda.reset_peak_memory_stats()
        generated: list[int] = []
        cache_length = 0
        current = input_ids
        target_forwards = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        first_at = None
        while len(generated) < max_new_tokens:
            positions = torch.arange(cache_length, cache_length + current.shape[1], device=current.device)
            forward = self.target.forward if cache_length == 0 else self.target_forward
            out = forward(
                input_ids=current,
                past_key_values=cache,
                cache_position=positions,
                use_cache=True,
                attention_mask=None,
                return_dict=True,
            )
            target_forwards += 1
            cache_length += current.shape[1]
            token = int(out.logits[:, -1:].argmax(-1).item())
            generated.append(token)
            if first_at is None:
                torch.cuda.synchronize()
                first_at = time.perf_counter()
            if self.stop_on_eos and token in self.eos_ids:
                break
            current = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if capture_memory:
            memory.append(gpu_snapshot(torch, "decode_peak"))
        first_at = first_at or end
        return GenerationResult(
            generated,
            (first_at - start) * 1000,
            (end - first_at) * 1000,
            (end - start) * 1000,
            target_forwards,
            0,
            0,
            0,
            target_forwards,
            [1] * len(generated),
            memory,
        )

    def _target_features(self, output):
        torch = self.torch
        selected = [int(x) for x in self.draft_config.pard2_target_layers]
        return torch.cat([output.hidden_states[i] for i in selected], dim=-1)

    def _generate_speculative(self, input_ids, max_new_tokens: int, capture_memory: bool) -> GenerationResult:
        torch = self.torch
        target_cache = self._new_cache(self.target)
        draft_cache = self._new_cache(self.draft)
        memory = [gpu_snapshot(torch, "cache_initialized")] if capture_memory else []
        torch.cuda.reset_peak_memory_stats()
        target_cache_length = 0
        draft_cache_length = 0
        target_input_ids = input_ids
        draft_input_ids = input_ids
        draft_target_feat = None
        all_target_feat = None
        target_dependent = self.mode == "pard2-td"
        target_forwards = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        if target_dependent:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            prefill = self.target(
                input_ids=input_ids,
                past_key_values=target_cache,
                cache_position=positions,
                use_cache=True,
                attention_mask=None,
                return_dict=True,
                output_hidden_states=True,
            )
            feat = self._target_features(prefill)
            zero = torch.zeros_like(feat[:, :1])
            all_target_feat = torch.cat([zero, feat[:, :-1]], dim=1)
            draft_target_feat = all_target_feat
            target_cache.reset()
            target_forwards += 1
        generated: list[int] = []
        draft_forwards = 0
        proposed = 0
        accepted = 0
        emitted_per_step: list[int] = []
        first_target = True
        first_at = None
        while len(generated) < max_new_tokens:
            masks = torch.full(
                (1, self.draft_k - 1),
                int(self.draft_config.pard_token),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            draft_call_ids = torch.cat([draft_input_ids, masks], dim=1)
            draft_positions = torch.arange(
                draft_cache_length,
                draft_cache_length + draft_call_ids.shape[1],
                device=input_ids.device,
            )
            draft_kwargs = {}
            if target_dependent:
                pad_feat = draft_target_feat[:, -1:].repeat(1, self.draft_k - 1, 1)
                draft_kwargs["target_feat"] = torch.cat([draft_target_feat, pad_feat], dim=1)
            draft_out = self.draft_forward(
                input_ids=draft_call_ids,
                past_key_values=draft_cache,
                cache_position=draft_positions,
                use_cache=True,
                attention_mask=None,
                return_dict=True,
                **draft_kwargs,
            )
            draft_forwards += 1
            draft_cache_length += draft_call_ids.shape[1]
            candidates = draft_out.logits[:, -self.draft_k :].argmax(-1)
            proposed += self.draft_k

            verify_ids = torch.cat([target_input_ids, candidates], dim=1)
            target_positions = torch.arange(
                target_cache_length,
                target_cache_length + verify_ids.shape[1],
                device=input_ids.device,
            )
            target_forward = self.target.forward if first_target else self.target_forward
            target_out = target_forward(
                input_ids=verify_ids,
                past_key_values=target_cache,
                cache_position=target_positions,
                use_cache=True,
                attention_mask=None,
                return_dict=True,
                output_hidden_states=target_dependent,
            )
            first_target = False
            target_forwards += 1
            target_cache_length += verify_ids.shape[1]
            predictions = target_out.logits[:, -(self.draft_k + 1) :].argmax(-1)
            accepted_this_step = 0
            for idx in range(self.draft_k):
                if int(predictions[0, idx]) != int(candidates[0, idx]):
                    break
                accepted_this_step += 1
            accepted += accepted_this_step
            emitted_tensor = predictions[:, : accepted_this_step + 1]
            emitted = self._truncate_for_eos(
                [int(x) for x in emitted_tensor[0].tolist()],
                max_new_tokens - len(generated),
            )
            if not emitted:
                break
            generated.extend(emitted)
            emitted_per_step.append(len(emitted))
            if target_dependent:
                new_feat = self._target_features(target_out)[:, -(self.draft_k + 1) :]
                keep_feat = new_feat[:, : len(emitted)]
                all_target_feat = torch.cat([all_target_feat, keep_feat], dim=1)
            if first_at is None:
                torch.cuda.synchronize()
                first_at = time.perf_counter()
            if (self.stop_on_eos and generated[-1] in self.eos_ids) or len(generated) >= max_new_tokens:
                break

            # The final emitted token is not cached by the target verifier.  Rewind
            # logical cache lengths; stale StaticCache slots are overwritten later.
            total_length = input_ids.shape[1] + len(generated)
            target_cache_length = min(total_length - 1, target_cache_length)
            target_input_ids = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=input_ids.dtype)

            draft_cache_length = max(0, draft_cache_length - (self.draft_k - 1))
            draft_input_ids = torch.tensor([emitted], device=input_ids.device, dtype=input_ids.dtype)
            if target_dependent:
                draft_target_feat = all_target_feat[:, -len(emitted) :]
        torch.cuda.synchronize()
        end = time.perf_counter()
        if capture_memory:
            memory.append(gpu_snapshot(torch, "decode_peak"))
        first_at = first_at or end
        return GenerationResult(
            generated,
            (first_at - start) * 1000,
            (end - first_at) * 1000,
            (end - start) * 1000,
            target_forwards,
            draft_forwards,
            proposed,
            accepted,
            len(emitted_per_step),
            emitted_per_step,
            memory,
        )


def load_runtime(
    *,
    mode: str,
    target_spec,
    draft_spec,
    draft_k: int,
    max_cache_len: int,
    compile_mode: str,
    local_files_only: bool,
    token: str | None,
    stop_on_eos: bool = False,
):
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .compat import validate_configs

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device is not available")
    dtype = torch.bfloat16
    common = {"token": token, "local_files_only": local_files_only}
    target_config = AutoConfig.from_pretrained(target_spec.model_id, revision=target_spec.revision, **common)
    draft_config = None
    if draft_spec is not None:
        draft_config = AutoConfig.from_pretrained(draft_spec.model_id, revision=draft_spec.revision, **common)
    compatibility = validate_configs(mode, target_config, draft_config, target_spec.model_id)
    if not compatibility.compatible:
        raise RuntimeError(compatibility.reason)
    tokenizer = AutoTokenizer.from_pretrained(target_spec.model_id, revision=target_spec.revision, **common)
    load_memory: list[dict] = [gpu_snapshot(torch, "before_model_load")]
    target = AutoModelForCausalLM.from_pretrained(
        target_spec.model_id,
        revision=target_spec.revision,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        **common,
    ).eval().to("cuda")
    target.parameter_bytes = parameter_bytes(target)
    load_memory.append(gpu_snapshot(torch, "target_loaded"))
    draft = None
    if draft_spec is not None:
        draft = AutoModelForCausalLM.from_pretrained(
            draft_spec.model_id,
            revision=draft_spec.revision,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            **common,
        )
        if mode == "pard2-td":
            draft = TargetFeatEmbedWarp.build(
                nn,
                draft,
                target_dim=int(draft_config.pard2_target_dim),
                scale=float(draft_config.pard2_scale),
                proj_bias=bool(draft_config.pard2_proj_bias),
            )
            warp_path = hf_hub_download(
                repo_id=draft_spec.model_id,
                filename="warp_model.bin",
                revision=draft_spec.revision,
                token=token,
                local_files_only=local_files_only,
            )
            state = torch.load(warp_path, map_location="cpu", weights_only=True)
            draft.target_proj.load_state_dict(
                {k.removeprefix("target_proj."): v for k, v in state.items() if k.startswith("target_proj.")}
            )
        draft = draft.eval().to("cuda", dtype=dtype)
        draft.parameter_bytes = parameter_bytes(draft)
        load_memory.append(gpu_snapshot(torch, "draft_loaded"))
    runtime = PardRuntime(
        torch=torch,
        mode=mode,
        tokenizer=tokenizer,
        target=target,
        draft=draft,
        target_config=target_config,
        draft_config=draft_config,
        draft_k=draft_k,
        max_cache_len=max_cache_len,
        compile_mode=compile_mode,
        load_memory=load_memory,
        stop_on_eos=stop_on_eos,
    )
    return runtime, compatibility
