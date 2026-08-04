from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pard_benchmark.compat import validate_configs
from pard_benchmark.config import OFFICIAL_TD_TARGET, load_hf_env
from pard_benchmark.report import first_mismatch, summarize


def config(**kwargs):
    return SimpleNamespace(**kwargs)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.target = config(model_type="llama", vocab_size=128256, hidden_size=4096, num_hidden_layers=32)
        self.pard = config(model_type="llama", vocab_size=128256, pard_token=128020, spd_type="pard")
        self.pard2 = config(
            model_type="llama",
            vocab_size=128256,
            pard_token=128020,
            spd_type="pard2",
            pard2=True,
            pard2_target_layers=[-1, -8, -16, -24],
            pard2_target_dim=16384,
        )

    def test_pard_is_compatible_with_llama31_vocab(self):
        self.assertTrue(validate_configs("pard", self.target, self.pard).compatible)

    def test_vocab_mismatch_fails(self):
        bad = config(model_type="llama", vocab_size=100, pard_token=20, spd_type="pard")
        self.assertFalse(validate_configs("pard", self.target, bad).compatible)

    def test_pard2_td_rejects_base_target(self):
        result = validate_configs("pard2-td", self.target, self.pard2, "meta-llama/Llama-3.1-8B")
        self.assertFalse(result.compatible)

    def test_pard2_td_accepts_official_target_shape(self):
        result = validate_configs("pard2-td", self.target, self.pard2, OFFICIAL_TD_TARGET)
        self.assertTrue(result.compatible)


class EnvironmentTests(unittest.TestCase):
    def test_env_file_does_not_override_existing_secret(self):
        old = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "existing"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ".hf_env"
                path.write_text("HF_TOKEN=file-secret\nUNKNOWN=value\n")
                loaded = load_hf_env(path)
                self.assertEqual(os.environ["HF_TOKEN"], "existing")
                self.assertNotIn("HF_TOKEN", loaded)
                self.assertNotIn("UNKNOWN", os.environ)
        finally:
            if old is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old

    def test_env_loader_redacts_return_value(self):
        old = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ".hf_env"
                path.write_text("HUGGING_FACE_HUB_TOKEN=secret-value\n")
                loaded = load_hf_env(path)
                self.assertEqual(loaded["HUGGING_FACE_HUB_TOKEN"], "<loaded>")
                self.assertNotIn("secret-value", repr(loaded))
        finally:
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            if old is not None:
                os.environ["HUGGING_FACE_HUB_TOKEN"] = old


class ReportTests(unittest.TestCase):
    def test_first_mismatch(self):
        self.assertEqual(first_mismatch([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(first_mismatch([1, 2], [1, 2]), "")
        self.assertEqual(first_mismatch([1], [1, 2]), 1)

    def test_summary_computes_parity_and_speedup(self):
        def payload(mode, speed, output_ids):
            return {
                "schema_version": 1,
                "upstream_commit": "test",
                "phase": "smoke",
                "mode": mode,
                "target_key": "base",
                "target": {"model_id": "target", "revision": "rev"},
                "draft": None if mode == "ar" else {"model_id": "draft", "revision": "rev"},
                "draft_k": 1 if mode == "ar" else 4,
                "context_len": 128,
                "max_new_tokens": 2,
                "compile_mode": "eager",
                "environment": {"device_total_memory": 32 * 2**30},
                "parameter_bytes": {"target": 10, "draft": 0 if mode == "ar" else 2},
                "load_memory": [],
                "repeats": [
                    {
                        "repeat": 0,
                        "steady_tokens_per_s": speed,
                        "end_to_end_tokens_per_s": speed,
                        "output_ids": output_ids,
                        "memory": [
                            {
                                "phase": "decode_peak",
                                "peak_reserved_bytes": 20 * 2**30,
                                "gpu_free_bytes": 12 * 2**30,
                                "allocated_bytes": 1,
                                "reserved_bytes": 1,
                                "peak_allocated_bytes": 1,
                                "gpu_total_bytes": 32 * 2**30,
                                "gpu_used_global_bytes": 20 * 2**30,
                                "host_rss_bytes": 1,
                            }
                        ],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "ar.json").write_text(json.dumps(payload("ar", 10.0, [1, 2])))
            (directory / "pard.json").write_text(json.dumps(payload("pard", 20.0, [1, 2])))
            result = summarize(directory)
            self.assertEqual(result["results"], 2)
            summary_text = (directory / "latency_summary.csv").read_text()
            parity_text = (directory / "generation_check.csv").read_text()
            self.assertIn("2.0", summary_text)
            self.assertIn("True", parity_text)


if __name__ == "__main__":
    unittest.main()
