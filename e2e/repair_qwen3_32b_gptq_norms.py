#!/usr/bin/env python3
"""Restore source BF16 Q/K norm tensors in a streamed GPTQ checkpoint."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_COUNT = 64
NORM_SUFFIXES = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def _fail(message):
    raise ValueError(message)


def _atomic_save(tensors, path, metadata):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, temporary, metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair(checkpoint):
    root = Path(checkpoint).resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    conversion = config.get("quarot_conversion")
    if not isinstance(conversion, dict):
        _fail("quarot_conversion is not an object")
    if conversion.get("method") != "gptq" or conversion.get("version") != 5:
        _fail("repair requires the exact streamed GPTQ v5 checkpoint")
    signature = json.dumps(
        conversion, sort_keys=True, separators=(",", ":"))

    source = Path(conversion.get("model", "")).resolve()
    source_index_path = source / "model.safetensors.index.json"
    if not source_index_path.is_file():
        _fail(f"missing immutable source index: {source_index_path}")
    source_weight_map = json.loads(
        source_index_path.read_text()).get("weight_map")
    if not isinstance(source_weight_map, dict):
        _fail("source weight_map is not an object")

    repaired_shards = 0
    already_correct_shards = 0
    replaced_tensors = 0
    for layer in range(LAYER_COUNT):
        prefix = f"model.layers.{layer}."
        shard = root / f"model-layer-{layer:05d}.safetensors"
        if not shard.is_file():
            _fail(f"missing layer shard: {shard.name}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata != {
                    "format": "pt", "quarot_signature": signature}:
                _fail(f"{shard.name} metadata/signature mismatch")
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}

        changed = False
        for suffix in NORM_SUFFIXES:
            key = prefix + suffix
            current = tensors.get(key)
            source_shard_name = source_weight_map.get(key)
            if current is None or source_shard_name is None:
                _fail(f"missing norm mapping: {key}")
            with safe_open(
                    source / source_shard_name,
                    framework="pt", device="cpu") as source_handle:
                source_tensor = source_handle.get_tensor(key)
            if source_tensor.dtype != torch.bfloat16:
                _fail(f"{key} source dtype is {source_tensor.dtype}, not BF16")
            if current.shape != source_tensor.shape:
                _fail(
                    f"{key} shape mismatch: {tuple(current.shape)} != "
                    f"{tuple(source_tensor.shape)}")
            if current.dtype == torch.bfloat16:
                if not torch.equal(current, source_tensor):
                    _fail(f"{key} BF16 value differs from immutable source")
                continue
            if current.dtype != torch.float16:
                _fail(f"{key} unexpected checkpoint dtype: {current.dtype}")
            if not torch.equal(current.to(torch.bfloat16), source_tensor):
                _fail(
                    f"{key} cannot be losslessly restored from immutable source")
            tensors[key] = source_tensor
            replaced_tensors += 1
            changed = True

        if changed:
            _atomic_save(tensors, shard, metadata)
            repaired_shards += 1
        else:
            already_correct_shards += 1
        del tensors
        gc.collect()

    index = json.loads(index_path.read_text())
    expected_shards = {
        "model-global.safetensors",
        *(f"model-layer-{layer:05d}.safetensors"
          for layer in range(LAYER_COUNT)),
    }
    actual_shards = {
        path.name for path in root.glob("model-*.safetensors")}
    if actual_shards != expected_shards:
        _fail("checkpoint shard set changed during repair")
    total_size = sum((root / name).stat().st_size for name in expected_shards)
    index["metadata"] = {"total_size": total_size}
    temporary_index = index_path.with_name(
        f".{index_path.name}.{os.getpid()}.tmp")
    try:
        temporary_index.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_index, index_path)
    finally:
        temporary_index.unlink(missing_ok=True)

    return {
        "status": "passed",
        "checkpoint": str(root),
        "source": str(source),
        "repaired_shards": repaired_shards,
        "already_correct_shards": already_correct_shards,
        "replaced_tensors": replaced_tensors,
        "source_dtype": "BF16",
        "exact_source_match": True,
        "safetensors_file_bytes": total_size,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = repair(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
