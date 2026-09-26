#!/usr/bin/env python3
"""Strict, CPU-only audit for the streamed Qwen3-32B QuaRot checkpoint."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

from e2e.benchmark_pard2 import target_profile_preflight


EXPECTED_AUTO_MAP = {
    "AutoConfig": "quarot.QuarotQwen3Config",
    "AutoModelForCausalLM": "quarot.QuarotQwen3ForCausalLM",
}
REQUIRED_SUPPORT = {
    "config.json",
    "model.safetensors.index.json",
    "quarot.py",
    "quantized_common.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
DTYPE_BYTES = {"BF16": 2, "U8": 1, "F32": 4}
EXPECTED_PAYLOAD_BYTES = 18_732_843_008


def _fail(message):
    raise ValueError(message)


def _layer_tensors(layer):
    prefix = f"model.layers.{layer}"
    return {
        f"{prefix}.self_attn.q_norm.weight": ((128,), "BF16"),
        f"{prefix}.self_attn.k_norm.weight": ((128,), "BF16"),
        f"{prefix}.self_attn.q_proj.weight": ((8192, 2560), "U8"),
        f"{prefix}.self_attn.q_proj.weight_scales": ((8192, 1), "F32"),
        f"{prefix}.self_attn.k_proj.weight": ((1024, 2560), "U8"),
        f"{prefix}.self_attn.k_proj.weight_scales": ((1024, 1), "F32"),
        f"{prefix}.self_attn.v_proj.weight": ((1024, 2560), "U8"),
        f"{prefix}.self_attn.v_proj.weight_scales": ((1024, 1), "F32"),
        f"{prefix}.self_attn.o_proj.1.weight": ((5120, 4096), "U8"),
        f"{prefix}.self_attn.o_proj.1.weight_scales": ((5120, 1), "F32"),
        f"{prefix}.mlp.gate_proj.weight": ((25600, 2560), "U8"),
        f"{prefix}.mlp.gate_proj.weight_scales": ((25600, 1), "F32"),
        f"{prefix}.mlp.up_proj.weight": ((25600, 2560), "U8"),
        f"{prefix}.mlp.up_proj.weight_scales": ((25600, 1), "F32"),
        f"{prefix}.mlp.down_proj.weight": ((5120, 12800), "U8"),
        f"{prefix}.mlp.down_proj.weight_scales": ((5120, 1), "F32"),
    }


def _expected_shards():
    shards = {
        "model-global.safetensors": {
            "model.embed_tokens.weight": ((151936, 5120), "BF16"),
            "lm_head.weight": ((151936, 5120), "BF16"),
        },
    }
    for layer in range(64):
        shards[f"model-layer-{layer:05d}.safetensors"] = _layer_tensors(layer)
    return shards


def _snapshot(root):
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink() and not path.exists():
            _fail(f"broken symlink: {path.name}")
        if not path.is_file():
            _fail(f"unexpected non-file entry: {path.name}")
        stat = path.stat()
        if stat.st_size <= 0:
            _fail(f"empty artifact file: {path.name}")
        result[path.name] = (stat.st_size, stat.st_mtime_ns)
    return result


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _payload_bytes(spec):
    total = 0
    for shape, dtype in spec.values():
        count = math.prod(shape)
        total += count * DTYPE_BYTES[dtype]
    return total


def _validate_numbers(config):
    signs = config.get("quarot_rotation_signs")
    if (not isinstance(signs, list) or len(signs) != 5120
            or any(type(value) is not int or value not in (-1, 1)
                   for value in signs)):
        _fail("rotation signs must be exactly 5120 JSON integers in {-1,+1}")
    final_norm = config.get("quarot_final_norm_weight")
    if (not isinstance(final_norm, list) or len(final_norm) != 5120
            or any(type(value) not in (int, float)
                   or not math.isfinite(float(value))
                   for value in final_norm)):
        _fail("final norm must be exactly 5120 finite JSON numbers")
    return signs, final_norm


def _validate_source_norm(config, final_norm):
    conversion = config["quarot_conversion"]
    source = Path(conversion["model"])
    index_path = source / "model.safetensors.index.json"
    source_index = json.loads(index_path.read_text())
    key = "model.norm.weight"
    shard_name = source_index["weight_map"].get(key)
    if shard_name is None:
        _fail("source index has no model.norm.weight")
    with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
        source_norm = handle.get_tensor(key)
    recorded = torch.tensor(final_norm, dtype=source_norm.dtype)
    if source_norm.shape != (5120,) or not torch.equal(source_norm, recorded):
        _fail("quarot_final_norm_weight differs from immutable source tensor")
    return {
        "source_shard": shard_name,
        "shape": list(source_norm.shape),
        "dtype": str(source_norm.dtype),
        "exact": True,
    }


def _expected_conversion_signature(conversion, method):
    common = {
        "model": conversion.get("model"),
        "seed": 0,
        "rotation_device": "cuda",
        "rotation_dtype": "float32",
        "w_bits": 4,
        "w_groupsize": -1,
        "w_asym": False,
        "w_clip": True,
    }
    if method == "rtn":
        return {"version": 2, "method": "rtn", **common}
    if method == "gptq":
        return {
            "version": 5, "method": "gptq", **common,
            "percdamp": 0.01,
            "act_order": False,
            "dataset": "wikitext2",
            "nsamples": 128,
            "seqlen": 2048,
        }
    _fail(f"unsupported expected conversion method: {method!r}")


def audit(checkpoint, expected_method="rtn"):
    root = Path(checkpoint).resolve()
    if not root.is_dir():
        _fail(f"checkpoint is not a directory: {root}")
    before = _snapshot(root)
    bad_names = [
        name for name in before
        if any(marker in name for marker in (".tmp", ".partial", ".incomplete"))
    ]
    if bad_names:
        _fail(f"temporary/incomplete files remain: {bad_names}")

    missing_support = sorted(REQUIRED_SUPPORT - set(before))
    if missing_support:
        _fail(f"missing support files: {missing_support}")

    expected_shards = _expected_shards()
    actual_shards = {name for name in before if name.endswith(".safetensors")}
    if actual_shards != set(expected_shards):
        _fail(
            "safetensors shard set mismatch: "
            f"missing={sorted(set(expected_shards) - actual_shards)}, "
            f"extra={sorted(actual_shards - set(expected_shards))}")

    preflight = target_profile_preflight(root, "qwen3_32b")
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if config.get("auto_map") != EXPECTED_AUTO_MAP:
        _fail(f"auto_map mismatch: {config.get('auto_map')!r}")
    _, final_norm = _validate_numbers(config)

    conversion = config.get("quarot_conversion")
    if not isinstance(conversion, dict):
        _fail("quarot_conversion is not an object")
    expected_signature = _expected_conversion_signature(
        conversion, expected_method)
    if conversion != expected_signature:
        _fail(
            f"conversion signature mismatch: {conversion!r} "
            f"!= {expected_signature!r}")

    expected_weight_map = {}
    payload_bytes = 0
    signature_text = None
    shard_headers = {}
    for shard_name, tensor_spec in expected_shards.items():
        shard_path = root / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(tensor_spec):
                _fail(
                    f"{shard_name} key mismatch: "
                    f"missing={sorted(set(tensor_spec) - keys)}, "
                    f"extra={sorted(keys - set(tensor_spec))}")
            metadata = handle.metadata()
            if set(metadata or {}) != {"format", "quarot_signature"}:
                _fail(f"{shard_name} metadata keys mismatch: {metadata!r}")
            if metadata["format"] != "pt":
                _fail(f"{shard_name} metadata format is not pt")
            if signature_text is None:
                signature_text = metadata["quarot_signature"]
            elif metadata["quarot_signature"] != signature_text:
                _fail(f"{shard_name} quarot_signature differs")
            for key, (shape, dtype) in tensor_spec.items():
                tensor_slice = handle.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape())
                actual_dtype = str(tensor_slice.get_dtype())
                if actual_shape != shape or actual_dtype != dtype:
                    _fail(
                        f"{key} header mismatch: "
                        f"shape={actual_shape}, dtype={actual_dtype}; "
                        f"expected shape={shape}, dtype={dtype}")
                expected_weight_map[key] = shard_name
        shard_payload = _payload_bytes(tensor_spec)
        payload_bytes += shard_payload
        shard_headers[shard_name] = {
            "keys": len(tensor_spec),
            "payload_bytes": shard_payload,
            "file_bytes": shard_path.stat().st_size,
        }

    parsed_signature = json.loads(signature_text)
    if parsed_signature != expected_signature:
        _fail(
            f"shard signature mismatch: {parsed_signature!r} "
            f"!= {expected_signature!r}")
    if index.get("weight_map") != expected_weight_map:
        _fail("index weight_map differs from exact header-derived mapping")
    shard_file_bytes = sum(
        (root / name).stat().st_size for name in expected_shards)
    if index.get("metadata") != {"total_size": shard_file_bytes}:
        _fail(
            f"index total_size mismatch: {index.get('metadata')!r}, "
            f"expected {shard_file_bytes}")
    if payload_bytes != EXPECTED_PAYLOAD_BYTES:
        _fail(
            f"tensor payload bytes {payload_bytes} "
            f"!= {EXPECTED_PAYLOAD_BYTES}")

    source_norm = _validate_source_norm(config, final_norm)

    files = []
    manifest_digest = hashlib.sha256()
    for name in sorted(before):
        path = root / name
        digest = _sha256(path)
        size = path.stat().st_size
        entry = {"path": name, "size": size, "sha256": digest}
        files.append(entry)
        manifest_digest.update(json.dumps(
            [name, size, digest], separators=(",", ":")).encode())
        manifest_digest.update(b"\n")

    after = _snapshot(root)
    if before != after:
        _fail("checkpoint changed while it was being audited")

    return {
        "status": "passed",
        "checkpoint": str(root),
        "profile_preflight": preflight,
        "shard_count": len(expected_shards),
        "weight_map_entries": len(expected_weight_map),
        "tensor_payload_bytes": payload_bytes,
        "safetensors_file_bytes": shard_file_bytes,
        "checkpoint_file_bytes": sum(size for size, _ in before.values()),
        "quarot_signature": parsed_signature,
        "source_final_norm": source_norm,
        "shards": shard_headers,
        "files": files,
        "manifest_sha256": manifest_digest.hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-method", choices=("rtn", "gptq"), default="rtn",
        help="Require the exact conversion signature for this method.")
    args = parser.parse_args(argv)
    root = Path(args.checkpoint).resolve()
    output = args.output.resolve()
    if root == output.parent or root in output.parents:
        _fail("--output must be outside the audited checkpoint")
    result = audit(root, expected_method=args.expected_method)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "status": result["status"],
        "shard_count": result["shard_count"],
        "weight_map_entries": result["weight_map_entries"],
        "tensor_payload_bytes": result["tensor_payload_bytes"],
        "safetensors_file_bytes": result["safetensors_file_bytes"],
        "manifest_sha256": result["manifest_sha256"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
