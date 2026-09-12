"""Run the stable real-INT4 benchmark on the current multi-runtime implementation.

The workload and measurement implementation lives in
``benchmark_real_llama_runtime.py`` and is shared verbatim with the portable Llama
benchmark. This file contains only the current revision's model loader.
"""
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e import benchmark_real_llama_runtime as harness
from e2e.model_registry import runtime_types


def load_current_int4_model(checkpoint):
    config_class, model_class, _ = runtime_types(
        checkpoint, local_files_only=True)
    config = config_class.from_pretrained(
        checkpoint,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    return model_class.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
    )


def load_current_fp16_model(checkpoint):
    _, _, model_class = runtime_types(checkpoint)
    return model_class.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )


if __name__ == "__main__":
    harness.main(
        int4_loader=load_current_int4_model,
        fp16_loader=load_current_fp16_model,
        description=__doc__,
        default_output="benchmark_real_multi_runtime.json",
    )
