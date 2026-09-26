def _generate_ar(self, input_ids, max_new_tokens):
    (cache, current, generated) = (self._target_cache(), input_ids, [])
    (forwards, first_at) = (0, None)
    timer = _StageTimer()
    started = 0.0
    while len(generated) < max_new_tokens:
        pos = torch.arange(cache.length, cache.length + current.shape[1], device=current.device)
        (output, _) = timer.record('target', lambda : self._target_call(current, cache, pos))
        if not generated:
            yield 'prefill_complete'
        cache = output.past_key_values
        forwards += 1
        token = int(output.logits[:, -1].argmax(-1))
        generated.append(token)
        if first_at is None:
            first_at = 0.0
        if self._stop(generated):
            break
        current = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
    stages = timer.totals()
    ended = 0.0
    first_at = first_at or ended
    return GenerationResult(generated, (first_at - started) * 1000, (ended - first_at) * 1000, (ended - started) * 1000, forwards, 0, 0, 0, forwards, [1] * len(generated), [1] * len(generated), stages, 0.0)
