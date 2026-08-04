from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is only available inside the ROCm container")
class PardGpuSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("ROCm/CUDA device is unavailable")

    def test_tiny_llama_pard_matches_ar(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from pard_benchmark.engine import PardRuntime

        cfg = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        torch.manual_seed(7)
        target = LlamaForCausalLM(cfg).eval().to("cuda", dtype=torch.bfloat16)
        torch.manual_seed(9)
        draft = LlamaForCausalLM(cfg).eval().to("cuda", dtype=torch.bfloat16)
        common = {
            "torch": torch,
            "tokenizer": None,
            "target": target,
            "target_config": cfg,
            "draft_config": SimpleNamespace(pard_token=127),
            "draft_k": 4,
            "max_cache_len": 64,
            "compile_mode": "eager",
            "load_memory": [],
        }
        ar = PardRuntime(mode="ar", draft=None, **common)
        pard = PardRuntime(mode="pard", draft=draft, **common)
        inputs = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
        ar_result = ar.generate(inputs, 12, capture_memory=False)
        pard_result = pard.generate(inputs, 12, capture_memory=False)
        self.assertEqual(ar_result.output_ids, pard_result.output_ids)


if __name__ == "__main__":
    unittest.main()
