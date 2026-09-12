"""Shared helpers for loading and validating real QuaRot INT4 checkpoints."""
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quarot
from e2e.model_registry import runtime_types


def deterministic_tokens(batch, length, vocab_size, device):
    values = torch.arange(batch * length, device=device, dtype=torch.long)
    return (values.reshape(batch, length) % max(vocab_size - 3, 1)) + 3


def validate_int4_checkpoint(model, checkpoint):
    modules = [module for module in model.modules()
               if isinstance(module, quarot.nn.Linear4bit)]
    if not modules:
        raise RuntimeError(f"{checkpoint} contains no Linear4bit modules")
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, quarot.nn.Linear4bit):
            continue
        if module.weight.dtype != torch.uint8 or module.weight.numel() == 0:
            failures.append(f"{name}: invalid packed weight")
        if not torch.isfinite(module.weight_scales).all():
            failures.append(f"{name}: nonfinite weight scales")
        if torch.count_nonzero(module.weight_scales) != module.weight_scales.numel():
            failures.append(f"{name}: zero weight scales")
    if failures:
        raise RuntimeError("invalid INT4 checkpoint:\n  " +
                           "\n  ".join(failures[:20]))
    return {
        "linear4bit_modules": len(modules),
        "scale_min": min(float(m.weight_scales.abs().min()) for m in modules),
        "scale_max": max(float(m.weight_scales.abs().max()) for m in modules),
        "packed_min": min(int(m.weight.min()) for m in modules),
        "packed_max": max(int(m.weight.max()) for m in modules),
    }


def load_int4(path):
    config_class, model_class, _ = runtime_types(path, local_files_only=True)
    config = config_class.from_pretrained(
        path, attn_implementation="flash_attention_2", local_files_only=True)
    return model_class.from_pretrained(
        path, config=config, torch_dtype=torch.float16, local_files_only=True)
