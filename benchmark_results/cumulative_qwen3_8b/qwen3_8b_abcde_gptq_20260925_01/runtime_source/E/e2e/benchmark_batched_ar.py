"""True batched greedy AR workloads for the shared phase benchmark."""
import inspect
from types import SimpleNamespace

import torch

from e2e.benchmark_pard2_phases import PhasedRuntime


class BatchedAR(PhasedRuntime):
    def __init__(self, runtime, prompt, tokens):
        if runtime.mode != 'ar' or not runtime.ignore_eos:
            raise ValueError('BatchedAR requires AR and fixed-length generation')
        self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
        self.expected_ids, self.checks = None, []
        self.cache = None
        self.generated_source = inspect.getsource(type(self))

    def _cache(self):
        if self.cache is None:
            self.cache = self.target.build_cache(self.prompt.shape[0], self.page_size,
                self.max_cache_len, native_gqa=self.native_gqa,
                fused_decode_append=self.fused_decode_append)
            # The runtime's persistent arange indices are only correct for
            # B=1 or a single page. Multi-batch prefill needs the existing
            # interleaved-page fallback. Static decode metadata stays enabled.
            if self.prompt.shape[0] > 1:
                self.cache._persistent_metadata_enabled = False
        else:
            self.cache.length = 0
            self.cache._needs_init = [True] * self.cache.n_layers
            self.cache._active_chunk_metadata = None
            self.cache._metadata_prepared_key = None
        return self.cache

    def _generate(self):
        cache, current = self._cache(), self.prompt
        generated = [[] for _ in range(self.prompt.shape[0])]
        for step in range(self.tokens):
            positions = torch.arange(cache.length, cache.length + current.shape[1],
                                     device=current.device)
            output, _ = self._target_call(current, cache, positions)
            cache = output.past_key_values
            if step == 0:
                yield 'prefill_complete'
            # Keep the batch-1 runtime's per-step host token transfer in timing.
            tokens = output.logits[:, -1].argmax(-1).tolist()
            for row, token in zip(generated, tokens):
                row.append(token)
            current = torch.tensor(tokens, device=self.prompt.device,
                                   dtype=self.prompt.dtype).unsqueeze(1)
        return SimpleNamespace(output_ids=generated, target_forwards=self.tokens,
            draft_forwards=0, accepted_draft_tokens=0, proposed_draft_tokens=0)

    def begin(self):
        state = self._generate()
        assert next(state) == 'prefill_complete'
        return state

    def validate(self, result):
        ids = result.output_ids
        if len(ids) != self.prompt.shape[0] or any(len(row) != self.tokens for row in ids):
            raise RuntimeError('Wrong batched output shape')
        if self.expected_ids is None:
            self.expected_ids = [list(row) for row in ids]
        if ids != self.expected_ids:
            raise RuntimeError('Batched output differs from oracle or another repeat')
        self.checks.append({'batch_size': len(ids), 'tokens_per_sequence': self.tokens,
                            'exact': True, 'target_forwards': result.target_forwards})
