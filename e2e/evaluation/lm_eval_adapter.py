"""lm-evaluation-harness adapter for loaded QuaRot/Hugging Face models."""
import torch
from lm_eval.models.huggingface import HFLM


class QuaRotHarnessLM(HFLM):
    """Use the pinned harness scoring logic with an already-loaded model."""

    def __init__(self, model, tokenizer, *, batch_size=1, max_length=2048,
                 kv_cache_dtype=None):
        if kv_cache_dtype is not None:
            model.cache_dtype = kv_cache_dtype
        model.config.use_cache = kv_cache_dtype is not None
        self.kv_cache_dtype = kv_cache_dtype
        super().__init__(
            pretrained=model, tokenizer=tokenizer, backend="causal",
            batch_size=batch_size, max_length=max_length,
            use_fast_tokenizer=False)
        self.forward_calls = 0
        self.forward_sequences = 0
        self.max_forward_tokens = 0

    @torch.inference_mode()
    def _model_call(self, inps, attn_mask=None, labels=None):
        if attn_mask is not None or labels is not None:
            raise ValueError("QuaRotHarnessLM supports causal models only")
        if self.kv_cache_dtype is not None:
            return self._model_call_with_cache(inps)
        self.forward_calls += 1
        self.forward_sequences += inps.shape[0]
        self.max_forward_tokens = max(
            self.max_forward_tokens, inps.shape[-1])
        return self.model(inps, use_cache=False).logits

    def _model_call_with_cache(self, inps):
        """Score every position through the real paged INT4 KV decode path."""
        logits = []
        cache = None
        self.model._expected_max_length = inps.shape[-1]
        for position in range(inps.shape[-1]):
            output = self.model(
                inps[:, position:position + 1], past_key_values=cache,
                use_cache=True)
            cache = output.past_key_values
            logits.append(output.logits)
            self.forward_calls += 1
            self.forward_sequences += inps.shape[0]
        self.max_forward_tokens = max(self.max_forward_tokens, inps.shape[-1])
        return torch.cat(logits, dim=1)

    def runtime_stats(self):
        return {
            "forward_calls": self.forward_calls,
            "forward_sequences": self.forward_sequences,
            "max_forward_tokens": self.max_forward_tokens,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
