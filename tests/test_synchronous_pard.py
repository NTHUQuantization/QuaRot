"""Independent AR oracle, divergent acceptance, and TD alignment contracts."""
import unittest
from types import SimpleNamespace, MethodType
from unittest.mock import patch

import torch

from e2e import speculative
from e2e.synchronous_pard import accepted_prefix_lengths, generate_synchronous
from e2e.benchmark_pard2_phases import UntimedStages
from test_benchmark_pard2_phases import ToyModel, toy_runtime


class FeatureTarget(ToyModel):
    def __call__(self, input_ids, past_key_values, **kwargs):
        output = super().__call__(input_ids, past_key_values, **kwargs)
        self.features = input_ids.float()[..., None].expand(-1, -1, 4)
        return output


class CheckedDraft(ToyModel):
    def __init__(self, fail_pattern, td):
        super().__init__(draft=True)
        self.fail_pattern, self.td = fail_pattern, td
        self.prefill_logits = []

    def project_features(self, features):
        return features

    def __call__(self, input_ids, past_key_values, cache_position, **kwargs):
        if self.td:
            features = kwargs.get('target_feat', kwargs.get('projected_target_feat'))
            assert features is not None
            real = input_ids.shape[1] if kwargs.get('logits_to_keep') == 1 else input_ids.shape[1] - 2
            start = int(cache_position[0])
            previous = (past_key_values.tokens[:, start - 1] if start else
                        torch.zeros(input_ids.shape[0], dtype=torch.long))
            expected = torch.cat((previous[:, None], input_ids[:, :real-1]), dim=1).float()
            torch.testing.assert_close(features[:, :real, 0], expected)
            if real < input_ids.shape[1]:
                torch.testing.assert_close(features[:, real:, 0], expected[:, -1:].expand(-1, 2))
        past_key_values.tokens[:, cache_position] = input_ids
        output = super().__call__(input_ids, past_key_values, **kwargs)
        if kwargs.get('logits_to_keep') == 1:
            self.prefill_logits.append(output.logits.shape[1])
        else:
            for row in range(input_ids.shape[0]):
                fail = self.fail_pattern[row % len(self.fail_pattern)]
                if fail < 3:
                    output.logits[row, -3 + fail].zero_()
                    output.logits[row, -3 + fail, 100] = 1
        return output


def runtime_for(mode, batch, pattern, lazy=False, projected=False):
    runtime = toy_runtime(mode)
    runtime.target = FeatureTarget()
    runtime.draft = CheckedDraft(pattern, mode == 'pard2-td')
    runtime.draft_forward = runtime.draft
    runtime.max_cache_len = 384
    def cache(size):
        return SimpleNamespace(length=0, batch_size=size, tokens=torch.zeros(size, 384, dtype=torch.long))
    runtime._target_cache = lambda size=1: cache(size)
    runtime._draft_cache = lambda size=1: cache(size)
    def target_call(ids, cache, positions, materialize_features=True):
        out = runtime.target(ids, cache)
        return out, runtime.target.features if materialize_features and mode == 'pard2-td' else None
    runtime._target_call = target_call
    if mode == 'pard2-td':
        runtime.collector = SimpleNamespace(reset=lambda: None,
            features=lambda rows=None: runtime.target.features if rows is None else runtime.target.features[:, rows])
    runtime.td_lazy_features, runtime.td_unique_projection = lazy, projected
    runtime._accepted_td_features = MethodType(speculative.FusedPardRuntime._accepted_td_features, runtime)
    runtime._project_td_features = MethodType(speculative.FusedPardRuntime._project_td_features, runtime)
    return runtime


def finish(state):
    assert next(state) == 'prefill_complete'
    try:
        next(state)
    except StopIteration as stopped:
        return stopped.value
    raise AssertionError('extra phase')


class SynchronousContracts(unittest.TestCase):
    def test_target_cache_batch_switch_releases_old_graphs(self):
        def build(batch, page, length, **kwargs):
            return SimpleNamespace(batch_size=batch, length=0, n_layers=1,
                _needs_init=[True], _persistent_metadata_enabled=True)
        runtime = SimpleNamespace(verification_graph=True, _verification_cache=None,
            _verification_graphs={}, target=SimpleNamespace(build_cache=build),
            page_size=128, max_cache_len=384, native_gqa=True, fused_decode_append=True)
        for batch in (1, 2, 4, 1):
            cache = speculative.FusedPardRuntime._target_cache(runtime, batch)
            self.assertEqual(cache.batch_size, batch)
            self.assertEqual(runtime._verification_graphs, {})
            self.assertEqual(cache._persistent_metadata_enabled, batch == 1)
            runtime._verification_graphs['sentinel'] = object()
            cache.length = 19
            reused = speculative.FusedPardRuntime._target_cache(runtime, batch)
            self.assertIs(reused, cache)
            self.assertEqual(reused.length, 0)
            self.assertIn('sentinel', runtime._verification_graphs)

    def test_per_row_acceptance(self):
        candidates = torch.tensor([[1, 2, 3]] * 4)
        predictions = torch.tensor([[0, 2, 3, 4], [1, 0, 3, 4], [1, 2, 0, 4], [1, 2, 3, 4]])
        self.assertEqual(accepted_prefix_lengths(candidates, predictions).tolist(), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            accepted_prefix_lengths(candidates, predictions[:, :3])

    def test_outputs_and_td_alignment_across_partial_rounds_and_pages(self):
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            for batch in (2, 4, 8, 16):
                for mode in ('pard2-ti', 'pard2-td'):
                    for tokens, length, pattern in ((1, 1, [3]), (7, 4, [1, 3]),
                                                  (31, 127, [0, 1, 2, 3]), (31, 129, [3])):
                        for lazy, projected in ((False, False), (True, True)):
                            with self.subTest(batch=batch, mode=mode, tokens=tokens, length=length,
                                              pattern=pattern, lazy=lazy):
                                prompt = torch.arange(batch * length).reshape(batch, length).remainder(50) + 3
                                expected = [speculative.FusedPardRuntime._generate_ar(
                                    toy_runtime('ar'), row[None], tokens).output_ids for row in prompt]
                                runtime = runtime_for(mode, batch, pattern, lazy, projected)
                                result = finish(generate_synchronous(runtime, prompt, tokens, UntimedStages()))
                                self.assertEqual(result.output_ids, expected)
                                self.assertEqual(sum(result.emitted_tokens_per_sequence_by_round), tokens)
                                self.assertTrue(all(c == min(a) for c, a in zip(
                                    result.common_accepted_by_round, result.accepted_lengths_by_round)))
                                if length > 1:
                                    self.assertEqual(runtime.draft.prefill_logits, [1])

    def test_public_runtime_guard_and_result(self):
        runtime = runtime_for('pard2-ti', 2, [1, 3])
        prompt = torch.tensor([[3, 4], [7, 8]])
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            result = speculative.FusedPardRuntime.generate(runtime, prompt, 7)
            self.assertEqual(result.output_ids, [list(range(5, 12)), list(range(9, 16))])
            runtime.ignore_eos = False
            with self.assertRaisesRegex(ValueError, 'ignore_eos'):
                speculative.FusedPardRuntime.generate(runtime, prompt, 7)


if __name__ == '__main__':
    unittest.main()
