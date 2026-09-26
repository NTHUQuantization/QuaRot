class BaselinePhases:
    def __init__(self, runtime, prompt, tokens):
        self.runtime, self.prompt, self.tokens = runtime, prompt, tokens
        self.generated_source = inspect.getsource(type(self))

    def generator(self):
        import torch
        runtime = self.runtime
        cache = runtime._benchmark_cache
        if cache is None:
            cache = runtime.target.build_cache(1, 128, 8192)
            runtime._benchmark_cache = cache
        # All branch cache request-local state is reset, not just length.
        cache.length = 0
        cache._needs_init = [True] * cache.n_layers
        ids = self.prompt
        positions = torch.arange(ids.shape[1], device=ids.device)
        output = runtime.target(input_ids=ids, past_key_values=cache,
            cache_position=positions, use_cache=True, attention_mask=None,
            return_dict=True, output_hidden_states=False, logits_to_keep=1)
        yield 'prefill_complete'
        generated = []
        while len(generated) < self.tokens:
            token = int(output.logits[:, -1].argmax(-1))
            generated.append(token)
            if token in runtime.eos_ids or len(generated) >= self.tokens:
                break
            ids = torch.tensor([[token]], dtype=self.prompt.dtype, device=self.prompt.device)
            positions = torch.arange(cache.length, cache.length+1, device=ids.device)
            output = runtime.target(input_ids=ids, past_key_values=cache,
                cache_position=positions, use_cache=True, attention_mask=None,
                return_dict=True, output_hidden_states=False, logits_to_keep=1)
            cache = output.past_key_values
        return SimpleNamespace(output_ids=generated, target_forwards=len(generated),
            draft_forwards=0, proposed_draft_tokens=0, accepted_draft_tokens=0,
            verifier_steps=0, emitted_tokens_per_step=[1]*len(generated),
            ar_fallback_tokens=0, fallback_after_verifier_steps=None)

    def begin(self):
        state = self.generator()
        assert next(state) == 'prefill_complete'
        return state

    def finish(self, state):
        try:
            next(state)
        except StopIteration as stopped:
            return stopped.value
        raise RuntimeError('Unexpected additional phase boundary')
