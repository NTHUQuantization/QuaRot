"""Exercise request-local TI fallback with a prefix-sensitive CPU target."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from e2e import speculative
from e2e.benchmark_pard2_phases import PhasedRuntime, UntimedStages
from test_benchmark_pard2_phases import toy_runtime


def runtime_with_bad_draft(threshold=4, eos=None):
    runtime = toy_runtime('pard2-ti')
    runtime.ti_zero_accept_fallback = threshold
    runtime._target_cache = lambda: SimpleNamespace(length=0, tokens=[])

    def target(input_ids, past_key_values, **kwargs):
        cache = past_key_values
        cache.tokens = cache.tokens[:cache.length] + input_ids[0].tolist()
        logits = torch.zeros(1, input_ids.shape[1], 128)
        for i in range(input_ids.shape[1]):
            # Every prediction depends on the whole committed KV prefix.
            token = sum(cache.tokens[:cache.length + i + 1]) % 61 + 1
            logits[0, i, token] = 1
        cache.length += input_ids.shape[1]
        if kwargs.get('logits_to_keep') == 1:
            logits = logits[:, -1:]
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def draft(input_ids, past_key_values, **kwargs):
        past_key_values.length += input_ids.shape[1]
        logits = torch.zeros(1, input_ids.shape[1], 128)
        logits[:, :, 0] = 1  # Target never emits zero.
        return SimpleNamespace(logits=logits)

    runtime.target = target
    runtime._target_call = lambda ids, cache, positions, materialize_features=True: (
        target(ids, cache), None)
    runtime.draft = runtime.draft_forward = draft
    runtime.ignore_eos = eos is None
    runtime.eos_ids = set() if eos is None else {eos}
    runtime._stop = lambda ids: bool(ids and ids[-1] in runtime.eos_ids)
    return runtime


@pytest.fixture(autouse=True)
def cpu_timing():
    with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
            torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0):
        yield


@pytest.mark.parametrize('prompt', [[[3]], [[3, 4, 5, 6]]])
@pytest.mark.parametrize('tokens', [1, 4, 5, 19])
def test_fallback_preserves_prefix_and_counts(prompt, tokens):
    ids = torch.tensor(prompt)
    runtime = runtime_with_bad_draft()
    expected = speculative.FusedPardRuntime._generate_ar(runtime, ids, tokens)
    actual = speculative.FusedPardRuntime._generate_spec(runtime, ids, tokens)
    assert actual.output_ids == expected.output_ids
    assert actual.ar_fallback_tokens == max(0, tokens - 4)
    assert actual.verifier_steps == min(tokens, 4)
    assert actual.draft_forwards == min(tokens, 4) + (len(prompt[0]) > 1)
    assert actual.proposed_draft_tokens == 3 * min(tokens, 4)
    assert actual.metrics()['mean_accept_length'] == 1
    assert sum(actual.emitted_tokens_per_step) == tokens
    assert actual.fallback_after_verifier_steps == (4 if tokens > 4 else None)
    # The phase adapter uses the same cache handoff and request-local state.
    adapter = PhasedRuntime(runtime, ids, tokens)
    phased = adapter.finish(adapter.begin())
    assert phased.output_ids == expected.output_ids
    assert phased.ar_fallback_tokens == actual.ar_fallback_tokens


def test_fallback_honors_eos_and_resets_for_each_request():
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = speculative.FusedPardRuntime._generate_ar(runtime_with_bad_draft(), ids, 20)
    eos = expected.output_ids[6]
    runtime = runtime_with_bad_draft(eos=eos)
    for _ in range(2):
        actual = speculative.FusedPardRuntime._generate_spec(runtime, ids, 20)
        assert actual.output_ids == expected.output_ids[:7]
        assert actual.ar_fallback_tokens == 3
        assert actual.verifier_steps == 4


def test_disabled_guard_keeps_pure_ti_and_good_draft_never_falls_back():
    ids = torch.tensor([[3, 4, 5, 6]])
    bad = speculative.FusedPardRuntime._generate_spec(runtime_with_bad_draft(0), ids, 19)
    assert bad.verifier_steps == 19
    assert bad.ar_fallback_tokens == 0
    runtime = toy_runtime('pard2-ti')
    runtime.ti_zero_accept_fallback = 4
    good = speculative.FusedPardRuntime._generate_spec(runtime, ids, 19)
    assert good.output_ids == list(range(7, 26))
    assert good.ar_fallback_tokens == 0
    assert good.accepted_draft_tokens > 0


def test_any_initial_success_disables_the_guard_even_if_later_rounds_reject():
    runtime = toy_runtime('pard2-ti')
    runtime.ti_zero_accept_fallback = 4
    original = runtime.draft_forward
    rounds = []
    def sometimes_wrong(**kwargs):
        out = original(**kwargs)
        rounds.append(len(rounds) + 1)
        if rounds[-1] in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12):
            out.logits[:, -runtime.spec.draft_k, :] = 0
            out.logits[:, -runtime.spec.draft_k, 0] = 1
        return out
    runtime.draft_forward = sometimes_wrong
    actual = speculative.FusedPardRuntime._generate_spec(runtime, torch.tensor([[3, 4, 5, 6]]), 31)
    assert actual.output_ids == list(range(7, 38))
    assert actual.accept_length_by_step[:7] == [1, 1, 1, 4, 1, 1, 1]
    assert actual.ar_fallback_tokens == 0


@pytest.mark.parametrize('mode,threshold', [('ar', 4), ('pard2-td', 4), ('pard2-ti', -1), ('pard2-ti', 1.5)])
def test_invalid_policy_fails_before_loading_a_checkpoint(mode, threshold):
    with pytest.raises(ValueError, match='zero-accept fallback'):
        speculative.load_runtime(mode=mode, target_checkpoint='does-not-exist',
            draft_snapshot='does-not-exist', tokenizer_path='does-not-exist',
            ti_zero_accept_fallback=threshold)


def test_guard_rejects_unsupported_batches():
    runtime = runtime_with_bad_draft()
    with pytest.raises(ValueError, match='batch size one'):
        speculative.FusedPardRuntime.generate(runtime, torch.tensor([[3], [4]]), 10)
