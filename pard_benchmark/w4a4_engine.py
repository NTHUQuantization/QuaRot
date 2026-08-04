from __future__ import annotations

import time

from .engine import GenerationResult
from .measurement import gpu_snapshot, parameter_bytes


class W4A4PardRuntime:
    """PARD/PARD2-TI drafter with the packed QuaRot target verifier."""

    def __init__(
        self,
        *,
        torch,
        tokenizer,
        target,
        draft,
        draft_config,
        draft_k: int,
        max_cache_len: int,
        load_memory: list[dict],
        stop_on_eos: bool = False,
    ):
        if not 1 <= draft_k <= 15:
            raise ValueError("packed verifier requires draft_k in [1, 15]")
        self.torch = torch
        self.tokenizer = tokenizer
        self.target = target
        self.draft = draft
        self.draft_config = draft_config
        self.draft_k = draft_k
        self.max_cache_len = max_cache_len
        self.load_memory = load_memory
        self.stop_on_eos = stop_on_eos
        eos = getattr(target.config, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos]) - {None}

    def _new_draft_cache(self):
        from transformers import StaticCache

        return StaticCache(
            config=self.draft.config,
            max_batch_size=1,
            max_cache_len=self.max_cache_len,
            device=self.draft.device,
            dtype=self.draft.dtype,
        )

    def generate(self, input_ids, max_new_tokens: int, capture_memory: bool = True) -> GenerationResult:
        if self.draft is None:
            return self._generate_ar(input_ids, max_new_tokens, capture_memory)
        torch = self.torch
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("PARD integration currently requires batch size 1")
        if input_ids.size(1) + max_new_tokens + self.draft_k > self.max_cache_len:
            raise ValueError("max_cache_len does not include speculative headroom")
        with torch.inference_mode():
            draft_cache = self._new_draft_cache()
            memory = [gpu_snapshot(torch, "cache_initialized")] if capture_memory else []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prefill_logits, target_cache = self.target.prefill(
                input_ids,
                max_new_tokens=self.max_cache_len - input_ids.size(1),
            )
            target_forwards = 1
            draft_cache_length = 0
            draft_input_ids = input_ids
            generated: list[int] = []
            draft_forwards = proposed = accepted = 0
            emitted_per_step: list[int] = []
            first_at = None
            first_step = True
            pending = None

            while len(generated) < max_new_tokens:
                masks = torch.full(
                    (1, self.draft_k - 1),
                    int(self.draft_config.pard_token),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                draft_call_ids = torch.cat([draft_input_ids, masks], dim=1)
                positions = torch.arange(
                    draft_cache_length,
                    draft_cache_length + draft_call_ids.size(1),
                    device=input_ids.device,
                )
                draft_out = self.draft(
                    input_ids=draft_call_ids,
                    past_key_values=draft_cache,
                    cache_position=positions,
                    use_cache=True,
                    attention_mask=None,
                    return_dict=True,
                )
                draft_forwards += 1
                draft_cache_length += draft_call_ids.size(1)
                candidates = draft_out.logits[:, -self.draft_k :].argmax(-1)
                proposed += self.draft_k

                if first_step:
                    verified, transaction = self.target.verify_chunk(candidates, target_cache)
                    predictions = torch.cat((prefill_logits[:, -1:], verified), dim=1).argmax(-1)
                    commit_base = 0
                    first_step = False
                else:
                    verify_ids = torch.cat((pending, candidates), dim=1)
                    verified, transaction = self.target.verify_chunk(verify_ids, target_cache)
                    predictions = verified.argmax(-1)
                    commit_base = 1
                target_forwards += 1

                accepted_step = 0
                for index in range(self.draft_k):
                    if int(predictions[0, index]) != int(candidates[0, index]):
                        break
                    accepted_step += 1
                accepted += accepted_step
                transaction.commit(commit_base + accepted_step)

                emitted = [int(x) for x in predictions[0, : accepted_step + 1].tolist()]
                emitted = emitted[: max_new_tokens - len(generated)]
                if self.stop_on_eos:
                    for index, token in enumerate(emitted):
                        if token in self.eos_ids:
                            emitted = emitted[: index + 1]
                            break
                if not emitted:
                    break
                generated.extend(emitted)
                emitted_per_step.append(len(emitted))
                pending = torch.tensor([[emitted[-1]]], device=input_ids.device, dtype=input_ids.dtype)
                if first_at is None:
                    torch.cuda.synchronize()
                    first_at = time.perf_counter()
                if (self.stop_on_eos and emitted[-1] in self.eos_ids) or len(generated) >= max_new_tokens:
                    break

                draft_cache_length = max(0, draft_cache_length - (self.draft_k - 1))
                draft_input_ids = torch.tensor([emitted], device=input_ids.device, dtype=input_ids.dtype)

            torch.cuda.synchronize()
            ended = time.perf_counter()
            if capture_memory:
                memory.append(gpu_snapshot(torch, "decode_peak"))
            first_at = first_at or ended
            return GenerationResult(
                generated,
                (first_at - started) * 1000,
                (ended - first_at) * 1000,
                (ended - started) * 1000,
                target_forwards,
                draft_forwards,
                proposed,
                accepted,
                len(emitted_per_step),
                emitted_per_step,
                memory,
            )

    def _generate_ar(self, input_ids, max_new_tokens: int, capture_memory: bool) -> GenerationResult:
        torch = self.torch
        with torch.inference_mode():
            memory = [gpu_snapshot(torch, "cache_initialized")] if capture_memory else []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits, cache = self.target.prefill(input_ids, max_new_tokens=max_new_tokens)
            generated = []
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            first_at = time.perf_counter()
            while len(generated) < max_new_tokens:
                token = int(next_token.item())
                generated.append(token)
                if self.stop_on_eos and token in self.eos_ids:
                    break
                logits, cache = self.target.decode_one(next_token, cache)
                next_token = logits.argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            ended = time.perf_counter()
            if capture_memory:
                memory.append(gpu_snapshot(torch, "decode_peak"))
            return GenerationResult(
                generated,
                (first_at - started) * 1000,
                (ended - first_at) * 1000,
                (ended - started) * 1000,
                1 + len(generated),
                0,
                0,
                0,
                len(generated),
                [1] * len(generated),
                memory,
            )


def load_w4a4_runtime(
    *,
    checkpoint_path,
    draft_spec,
    tokenizer_spec,
    draft_k: int,
    max_cache_len: int,
    local_files_only: bool,
    token: str | None,
    stop_on_eos: bool = False,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from w4a4_runtime import QuaRotW4A4LlamaForCausalLM, audit_checkpoint
    from .compat import validate_configs

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device is not available")
    audit = audit_checkpoint(checkpoint_path, strict_model=True)
    target = QuaRotW4A4LlamaForCausalLM.from_quantized(checkpoint_path, device="cuda")
    target.parameter_bytes = int(audit["tensor_bytes"])
    target_config = target.config
    common = {"token": token, "local_files_only": local_files_only}
    draft_config = None
    if draft_spec is not None:
        draft_config = AutoConfig.from_pretrained(draft_spec.model_id, revision=draft_spec.revision, **common)
    mode = "ar" if draft_config is None else (
        "pard2-ti" if bool(getattr(draft_config, "pard2", False)) else "pard"
    )
    compatibility = validate_configs(mode, target_config, draft_config, target_id=str(checkpoint_path))
    if not compatibility.compatible:
        raise RuntimeError(compatibility.reason)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_spec.model_id, revision=tokenizer_spec.revision, **common
    )
    load_memory = [gpu_snapshot(torch, "before_model_load"), gpu_snapshot(torch, "target_loaded")]
    draft = None
    if draft_spec is not None:
        draft = AutoModelForCausalLM.from_pretrained(
            draft_spec.model_id,
            revision=draft_spec.revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **common,
        ).eval().to("cuda")
        draft.parameter_bytes = parameter_bytes(draft)
        load_memory.append(gpu_snapshot(torch, "draft_loaded"))
    runtime = W4A4PardRuntime(
        torch=torch,
        tokenizer=tokenizer,
        target=target,
        draft=draft,
        draft_config=draft_config,
        draft_k=draft_k,
        max_cache_len=max_cache_len,
        load_memory=load_memory,
        stop_on_eos=stop_on_eos,
    )
    return runtime, compatibility
