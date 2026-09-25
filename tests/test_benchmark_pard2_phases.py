"""CPU contracts for benchmark-only phase boundaries and allocator ownership."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from e2e import speculative
from e2e import benchmark_real_llama_runtime as harness
from e2e.benchmark_pard2_phases import PhasedRuntime, UntimedStages
from e2e.benchmark_batched_ar import BatchedAR


class ToyModel:
    def __init__(self, draft=False):
        self.draft = draft
        self.features = None
        self.calls = []

    def __call__(self, input_ids, past_key_values, **kwargs):
        self.calls.append(input_ids.clone())
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 128)
        for row, values in enumerate(input_ids.tolist()):
            last = 0
            for i, token in enumerate(values):
                last = last + 1 if self.draft and token == 99 else token
                logits[row, i, (last + 1) % 128] = 1
        self.features = torch.ones(batch, length, 4)
        past_key_values.length += length
        if kwargs.get('logits_to_keep') == 1:
            logits = logits[:, -1:]
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def toy_runtime(mode):
    target, draft = ToyModel(), ToyModel(draft=True)
    runtime = SimpleNamespace(mode=mode, target=target, draft=draft,
        ignore_eos=True, max_cache_len=128, spec=SimpleNamespace(draft_k=3, pard_token=99),
        adaptive_k=None, verification_graph=False, td_unique_projection=False,
        td_lazy_features=False, collector=None)
    if mode == 'pard2-td':
        runtime.collector = SimpleNamespace(reset=lambda: None, features=lambda: target.features)
    def target_call(ids, cache, positions, materialize_features=True):
        out = target(ids, cache)
        return out, target.features if runtime.collector else None
    runtime._target_call = target_call
    runtime._target_cache = lambda: SimpleNamespace(length=0)
    runtime._draft_cache = lambda: SimpleNamespace(length=0)
    runtime.draft_forward = draft
    runtime._stop = lambda tokens: False
    return runtime


class PhaseContracts(unittest.TestCase):
    def test_batched_cache_uses_correct_interleaved_page_indices(self):
        from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit
        with patch.dict(os.environ, QUAROT_PERSISTENT_KV_METADATA='1',
                        QUAROT_STATIC_KV_METADATA='1'):
            for batch in (2, 4, 8, 16):
                runtime = toy_runtime('ar')
                runtime.page_size, runtime.max_cache_len = 128, 384
                runtime.native_gqa = runtime.fused_decode_append = True
                runtime.target.build_cache = lambda b, page, length, **kw: MultiLayerPagedKVCache4Bit(
                    batch_size=b, page_size=page, max_seq_len=length,
                    device=torch.device('cpu'), n_layers=1, num_heads=4, num_kv_heads=2,
                    head_dim=64, disable_quant=False, hadamard_dtype=torch.float16, **kw)
                adapter = BatchedAR(runtime, torch.zeros(batch, 4, dtype=torch.long), 1)
                cache = adapter._cache()
                self.assertFalse(cache._persistent_metadata_enabled)
                self.assertTrue(cache._static_metadata_enabled)
                for length in (1, 128, 129, 256, 257):
                    cache.length = length
                    pages = (length + 127) // 128
                    specs = cache.get_cache_specs_for_flash_infer(None)
                    expected = [page * batch + row for row in range(batch) for page in range(pages)]
                    self.assertEqual(specs['kv_indices'].tolist(), expected)
                    self.assertEqual(specs['kv_indptr'].tolist(), [i * pages for i in range(batch + 1)])

    def test_true_batched_ar_matches_each_independent_sequence(self):
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            for batch in (2, 4, 8, 16):
                for tokens in (1, 7, 31):
                    with self.subTest(batch=batch, tokens=tokens):
                        prompt = torch.arange(3, 3 + batch * 4).reshape(batch, 4)
                        expected = [speculative.FusedPardRuntime._generate_ar(
                            toy_runtime('ar'), row[None], tokens).output_ids for row in prompt]
                        runtime = toy_runtime('ar')
                        runtime.page_size, runtime.native_gqa, runtime.fused_decode_append = 128, True, True
                        runtime.target.build_cache = lambda *a, **kw: SimpleNamespace(length=0, n_layers=1)
                        adapter = BatchedAR(runtime, prompt, tokens)
                        adapter.expected_ids = expected
                        for _ in range(2):
                            state = adapter.begin()
                            self.assertEqual(runtime.target.calls[-1].shape, (batch, 4))
                            actual = adapter.finish(state)
                            adapter.validate(actual)
                            self.assertEqual(actual.output_ids, expected)
                            self.assertEqual(actual.target_forwards, tokens)

    def test_split_matches_original_for_all_modes_and_partial_final_round(self):
        prompt = torch.tensor([[3, 4, 5, 6]])
        with patch.object(speculative, '_StageTimer', UntimedStages), patch.multiple(
                torch.cuda, synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0):
            for mode in ('ar', 'pard2-ti', 'pard2-td'):
                for tokens in (1, 7, 31):
                    with self.subTest(mode=mode, tokens=tokens):
                        control = toy_runtime(mode)
                        method = (speculative.FusedPardRuntime._generate_ar if mode == 'ar'
                                  else speculative.FusedPardRuntime._generate_spec)
                        expected = method(control, prompt, tokens)
                        runtime = toy_runtime(mode)
                        adapter = PhasedRuntime(runtime, prompt, tokens)
                        state = adapter.begin()
                        self.assertEqual(len(runtime.target.calls), 1)
                        self.assertEqual(runtime.target.calls[0].shape[1], 4)
                        if mode != 'ar':
                            self.assertEqual(len(runtime.draft.calls), 1)
                            self.assertEqual(runtime.draft.calls[0].shape[1], 3)
                        actual = adapter.finish(state)
                        self.assertEqual(actual.output_ids, expected.output_ids)
                        self.assertEqual(actual.output_ids, list(range(7, 7 + tokens)))
                        self.assertEqual(actual.accepted_draft_tokens, expected.accepted_draft_tokens)
                        self.assertNotIn('torch.cuda.reset_peak_memory_stats', adapter.generated_source)
                        self.assertNotIn('torch.cuda.synchronize', adapter.generated_source)
                        self.assertNotIn('time.perf_counter', adapter.generated_source)

    def test_decode_peak_excludes_preparation_allocations(self):
        allocator = {'live': 400, 'peak': 0}
        calls = []
        def prepare():
            calls.append('prepare')
            allocator['peak'] = 1000
        def work():
            calls.append('work')
            allocator['peak'] = max(allocator['peak'], 500)
        def reset():
            allocator['peak'] = allocator['live']
        with patch.multiple(torch.cuda, synchronize=lambda: None,
                memory_allocated=lambda: allocator['live'],
                max_memory_allocated=lambda: allocator['peak'],
                reset_peak_memory_stats=reset):
            result = harness.measure(work, warmup=1, repeats=2, prepare=prepare)
        self.assertEqual(calls, ['prepare', 'work'] * 3)
        self.assertEqual(result['peak_memory_bytes'], 500)
        self.assertEqual(result['baseline_memory_bytes'], 400)
        self.assertEqual(result['incremental_peak_memory_bytes'], 100)
        self.assertEqual(result['peak_memory_bytes_per_sample'], [500, 500])


if __name__ == '__main__':
    unittest.main()
