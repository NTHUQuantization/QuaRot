def _generate_spec(self, input_ids, max_new_tokens):
    (target_cache, draft_cache) = (self._target_cache(), self._draft_cache())
    (target_input, draft_input) = (None, input_ids)
    draft_cache_len = 0
    draft_features = None
    target_forwards = draft_forwards = proposed = accepted_total = 0
    (generated, emitted_steps, accept_lengths, proposal_lengths) = ([], [], [], [])
    fallback_threshold = getattr(self, 'ti_zero_accept_fallback', 0)
    ar_fallback_tokens = 0
    fallback_after_verifier_steps = None
    (timer, first_at) = (_StageTimer(), None)
    started = 0.0
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    (prefill_out, prefill_features) = timer.record('target_feature_prefill' if self.mode == 'pard2-td' else 'target_prefill', lambda : self._target_call(input_ids, target_cache, positions))
    target_cache = prefill_out.past_key_values
    target_forwards += 1
    first_round = True
    if self.mode == 'pard2-td':
        features = prefill_features
        zero = torch.zeros_like(features[:, :1])
        draft_features = torch.cat((zero, features[:, :-1]), dim=1)
    if input_ids.shape[1] > 1:
        prefix = input_ids[:, :-1]
        prefix_pos = torch.arange(prefix.shape[1], device=input_ids.device)
        prefix_kwargs = {}
        if self.mode == 'pard2-td':
            prefix_features = draft_features[:, :-1]
            if self.td_unique_projection:
                prefix_kwargs['projected_target_feat'] = timer.record('td_feature_project_prefill', lambda : self._project_td_features(prefix_features))
            else:
                prefix_kwargs['target_feat'] = prefix_features
        timer.record('draft_prefill', lambda : self.draft(input_ids=prefix, past_key_values=draft_cache, cache_position=prefix_pos, use_cache=True, attention_mask=None, return_dict=True, logits_to_keep=getattr(self, 'draft_prefill_logits_to_keep', 1), **prefix_kwargs))
        draft_forwards += 1
        draft_cache_len = prefix.shape[1]
        draft_input = input_ids[:, -1:]
        if self.mode == 'pard2-td':
            draft_features = draft_features[:, -1:]
    yield 'prefill_complete'
    pending_prediction = prefill_out.logits[:, -1:].argmax(-1)
    while len(generated) < max_new_tokens:
        if fallback_threshold and accepted_total == 0 and (len(accept_lengths) >= fallback_threshold):
            if fallback_after_verifier_steps is None:
                fallback_after_verifier_steps = len(accept_lengths)
            target_pos = torch.arange(target_cache.length, target_cache.length + 1, device=input_ids.device)
            (target_out, _) = timer.record('target_ar_fallback', lambda : self._target_call(target_input, target_cache, target_pos))
            target_cache = target_out.past_key_values
            token = int(target_out.logits[:, -1].argmax(-1))
            generated.append(token)
            emitted_steps.append(1)
            target_forwards += 1
            ar_fallback_tokens += 1
            if self._stop(generated):
                break
            target_input = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
            continue
        k = self.adaptive_k.choose() if self.adaptive_k is not None else self.spec.draft_k
        masks = torch.full((1, k - 1), self.spec.pard_token, device=input_ids.device, dtype=input_ids.dtype)
        draft_ids = torch.cat((draft_input, masks), dim=1)
        draft_pos = torch.arange(draft_cache_len, draft_cache_len + draft_ids.shape[1], device=input_ids.device)
        if hasattr(draft_cache, 'length'):
            draft_cache.length = draft_cache_len
        kwargs = {}
        if self.mode == 'pard2-td':
            if self.td_unique_projection:
                kwargs['projected_target_feat'] = timer.record('td_feature_project', lambda : self._project_td_features(draft_features, k - 1))
            else:
                pad = draft_features[:, -1:].expand(-1, k - 1, -1)
                kwargs['target_feat'] = torch.cat((draft_features, pad), dim=1)
        draft_out = timer.record('draft', lambda : self.draft_forward(input_ids=draft_ids, past_key_values=draft_cache, cache_position=draft_pos, use_cache=True, attention_mask=None, return_dict=True, **kwargs))
        draft_forwards += 1
        draft_cache_len += draft_ids.shape[1]
        candidates = draft_out.logits[:, -k:].argmax(-1)
        proposed += k
        proposal_lengths.append(k)
        verify_ids = candidates if first_round else torch.cat((target_input, candidates), dim=1)
        target_pos = None if self.verification_graph and getattr(target_cache, '_static_metadata_enabled', False) and (verify_ids.shape[1] in (15, 16)) else torch.arange(target_cache.length, target_cache.length + verify_ids.shape[1], device=input_ids.device)
        base_length = target_cache.length
        (target_out, new_features) = timer.record('target_verify', lambda : self._target_call(verify_ids, target_cache, target_pos, materialize_features=not self.td_lazy_features))
        if first_round:
            predictions = torch.cat((pending_prediction, target_out.logits[:, -k:].argmax(-1)), dim=1)
        else:
            predictions = target_out.logits[:, -(k + 1):].argmax(-1)
        accepted = greedy_accept(candidates, predictions)
        target_cache = target_out.past_key_values
        target_cache.length = base_length + (0 if first_round else 1) + accepted
        target_forwards += 1
        accepted_total += accepted
        if self.adaptive_k is not None:
            self.adaptive_k.update(accepted, k)
        accept_lengths.append(accepted + 1)
        emitted = [int(x) for x in predictions[0, :accepted + 1].tolist()]
        emitted = emitted[:max_new_tokens - len(generated)]
        if not self.ignore_eos:
            for (i, token) in enumerate(emitted):
                if token in self.eos_ids:
                    emitted = emitted[:i + 1]
                    break
        generated.extend(emitted)
        emitted_steps.append(len(emitted))
        if self.mode == 'pard2-td':
            if self.td_lazy_features:
                draft_features = self._accepted_td_features(prefill_features, len(emitted), first_round, timer)
            else:
                source = torch.cat((prefill_features[:, -1:], new_features), dim=1) if first_round else new_features
                draft_features = source[:, :len(emitted)]
        if first_at is None:
            first_at = 0.0
        if self._stop(generated) or len(generated) >= max_new_tokens:
            break
        target_input = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=input_ids.dtype)
        draft_cache_len = max(0, draft_cache_len - (k - 1))
        if hasattr(draft_cache, 'length'):
            draft_cache.length = draft_cache_len
        draft_input = torch.tensor([emitted], device=input_ids.device, dtype=input_ids.dtype)
        first_round = False
    stages = timer.totals()
    ended = 0.0
    first_at = first_at or ended
    return GenerationResult(generated, (first_at - started) * 1000, (ended - first_at) * 1000, (ended - started) * 1000, target_forwards, draft_forwards, proposed, accepted_total, len(accept_lengths), emitted_steps, accept_lengths, stages, 0.0, proposal_lengths, ar_fallback_tokens, fallback_after_verifier_steps)
