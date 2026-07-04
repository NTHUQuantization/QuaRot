#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import torch


MODEL_ID = "meta-llama/Llama-3.1-8B"
DEFAULT_PROMPTS = [
    "The future of efficient large language model inference depends on",
    "In a short paragraph, explain why kernel fusion matters for GPU inference.",
    "List three practical constraints when deploying quantized transformer models.",
]


@dataclass
class RuntimeConfig:
    model_id: str = MODEL_ID
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: str = "eager"
    seed: int = 123
    local_files_only: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[name]


def require_hf_token(model_id: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return
    if model_id.startswith("meta-llama/"):
        raise RuntimeError(
            "HF_TOKEN is required for gated Meta Llama repositories. "
            "Set HF_TOKEN after your HuggingFace account is granted access."
        )


def load_tokenizer_and_model(config: RuntimeConfig):
    if not config.local_files_only:
        require_hf_token(config.model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(config.seed)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=token,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=dtype_from_name(config.dtype),
        device_map=None,
        attn_implementation=config.attn_implementation,
        token=token,
        local_files_only=config.local_files_only,
    )
    model.eval().to(config.device)
    return tokenizer, model


def load_prompts(path: str | None) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return [str(x) for x in data]
        return [str(x["prompt"]) for x in data["prompts"]]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: List[float]) -> dict:
    values = list(values)
    if not values:
        return {}
    ordered = sorted(values)
    def pct(q):
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * q
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": min(values),
        "max": max(values),
    }


def cuda_event_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def wall_time_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def model_shape_summary(model) -> dict:
    cfg = model.config
    return {
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "vocab_size": cfg.vocab_size,
    }


def assert_llama31_kernel_compat(model) -> tuple[bool, str]:
    s = model_shape_summary(model)
    if s["hidden_size"] != 4096 or s["head_dim"] != 128:
        return False, f"expected hidden_size=4096/head_dim=128, got {s}"
    if s["num_attention_heads"] % s["num_key_value_heads"] != 0:
        return False, f"q_heads must be divisible by kv_heads, got {s}"
    return (
        True,
        "GQA-aware K1/K2 kernels and the formal token-by-token QuaRot wrapper are enabled. "
        "The wrapper keeps HF weights/modules for prefill and projections, then routes decode_one "
        "through the QuaRot paged-cache K1/K2/K3/FFN path. This is not a monkey-patch of "
        "transformers.generate internals.",
    )


def error_metrics(ref: torch.Tensor, out: torch.Tensor) -> dict:
    diff = (out.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-6)
    return {
        "max_error": diff.max().item(),
        "mean_error": diff.mean().item(),
        "mean_relative_error": rel.mean().item(),
    }


def logits_metrics(ref: torch.Tensor, out: torch.Tensor, topk: int = 10) -> dict:
    metrics = error_metrics(ref, out)
    ref_top1 = ref.argmax(dim=-1)
    out_top1 = out.argmax(dim=-1)
    metrics["top1_match"] = (ref_top1 == out_top1).float().mean().item()
    ref_topk = ref.topk(topk, dim=-1).indices
    out_topk = out.topk(topk, dim=-1).indices
    overlaps = []
    for a, b in zip(ref_topk.reshape(-1, topk), out_topk.reshape(-1, topk)):
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / topk)
    metrics[f"top{topk}_overlap"] = sum(overlaps) / len(overlaps)
    log_p = torch.log_softmax(ref.float(), dim=-1)
    log_q = torch.log_softmax(out.float(), dim=-1)
    p = log_p.exp()
    metrics["kl_ref_to_out"] = torch.nn.functional.kl_div(log_q, p, reduction="batchmean").item()
    return metrics
