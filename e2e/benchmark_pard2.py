"""Qualified single-process benchmark for one fused_v1 PARD2 mode/dataset."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.pard2 import (DEFAULT_DRAFT, DEFAULT_TARGET, DEFAULT_TOKENIZER,
                       verify_local_resources, verify_target_checkpoint)
from e2e.speculative import load_runtime


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_IDS_HASH_FORMAT = "sha256(canonical-json:{shape,values})"
DATA_ROOT = ROOT / "QuaRot/.benchmark_data/amd_pard_6f279bf"
DATASETS = {
    "humaneval": (80, "e16580cc87cac3168e59bde4ec1d0cd5cec7f31e1506c8ee1eba2ff914cf4368"),
    "gsm8k": (80, "56767ac321f1e80e3721f390c743a6a7253d4d5f50bd791339010ffcd8620371"),
    "math_500": (20, "4d8a669d746329ede429b1f4b037138ada9ae39ead6d0c34e1aa1be5349365a5"),
}
BENCHMARK_PROFILES = ("qwen3_8b", "qwen3_14b", "qwen3_32b")
TD_PROXY_PROFILES = ("qwen3-14b-on-qwen3-32b",)
DEFAULT_DRAFT_REVISION = "67a1516c8f6fc145cda99916799a0cbb3a4af135"
QWEN3_14B_SOURCE_MODEL_ID = "Qwen/Qwen3-14B"
QWEN3_14B_SOURCE_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
QWEN3_14B_SOURCE_CONFIG_SHA256 = (
    "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a")
QWEN3_14B_SOURCE_INDEX_SHA256 = (
    "62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc")
QWEN3_14B_ARCHITECTURE = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
QWEN3_32B_SOURCE_MODEL_ID = "Qwen/Qwen3-32B"
QWEN3_32B_SOURCE_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
QWEN3_32B_SOURCE_CONFIG_SHA256 = (
    "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb")
QWEN3_32B_SOURCE_INDEX_SHA256 = (
    "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0")
QWEN3_32B_ARCHITECTURE = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}

STRICT_PROFILE_ARCHITECTURES = {
    "qwen3_14b": QWEN3_14B_ARCHITECTURE,
    "qwen3_32b": QWEN3_32B_ARCHITECTURE,
}
STRICT_PROFILE_IDENTITIES = {
    "qwen3_14b": {
        "source_model_id": QWEN3_14B_SOURCE_MODEL_ID,
        "source_revision": QWEN3_14B_SOURCE_REVISION,
        "source_config_sha256": QWEN3_14B_SOURCE_CONFIG_SHA256,
        "source_index_sha256": QWEN3_14B_SOURCE_INDEX_SHA256,
    },
    "qwen3_32b": {
        "source_model_id": QWEN3_32B_SOURCE_MODEL_ID,
        "source_revision": QWEN3_32B_SOURCE_REVISION,
        "source_config_sha256": QWEN3_32B_SOURCE_CONFIG_SHA256,
        "source_index_sha256": QWEN3_32B_SOURCE_INDEX_SHA256,
    },
}


def cuda_memory_snapshot(stage):
    """Return allocator and device-global memory counters without resetting peaks."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "device_used_bytes": total - free,
        "global_used_bytes": total - free,
    }


def snapshot_revision(path):
    """Extract an immutable Hugging Face snapshot revision when one is present."""
    resolved = Path(path).resolve()
    return resolved.name if resolved.parent.name == "snapshots" else None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(input_ids):
    """Hash token IDs using a device- and dtype-independent JSON encoding."""
    values = input_ids.detach().to(device="cpu").tolist()
    canonical = json.dumps(
        {"shape": [int(size) for size in input_ids.shape], "values": values},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_output(*arguments):
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], check=True,
        capture_output=True, text=True)
    return completed.stdout.strip()


def _git_provenance():
    head = _git_output("rev-parse", "HEAD").lower()
    if (len(head) != 40
            or any(character not in "0123456789abcdef" for character in head)):
        raise RuntimeError(f"invalid benchmark git HEAD: {head!r}")
    return {
        "head": head,
        "dirty_tracked": bool(_git_output(
            "status", "--porcelain", "--untracked-files=no")),
    }


def _runtime_source_sha256():
    paths = {
        "benchmark_pard2.py": REPO_ROOT / "e2e/benchmark_pard2.py",
        "speculative.py": REPO_ROOT / "e2e/speculative.py",
        "quantized_common.py": REPO_ROOT / "e2e/quantized_common.py",
        "kv_cache.py": REPO_ROOT / "quarot/transformers/kv_cache.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing benchmark runtime sources: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def _hip_extension_provenance(expected_sha256=None):
    module = sys.modules.get("quarot._HIP")
    import_name = getattr(module, "__file__", None)
    if not import_name:
        raise RuntimeError("quarot._HIP is not loaded from a file")
    import_path = Path(import_name)
    try:
        resolved_path = import_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"loaded quarot._HIP path does not exist: {import_path}") from error
    actual_sha256 = sha256(resolved_path)
    expected_sha256 = _normalized_sha256(
        expected_sha256, "expected_hip_sha256")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "quarot._HIP SHA256 mismatch: "
            f"actual={actual_sha256}, expected={expected_sha256}")
    return {
        "module": "quarot._HIP",
        "import_path": str(import_path),
        "resolved_path": str(resolved_path),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "size_bytes": resolved_path.stat().st_size,
    }


def _uniform_module_flag(model, attribute):
    values = [bool(getattr(module, attribute)) for module in model.modules()
              if hasattr(module, attribute)]
    if not values:
        return {"effective": None, "module_count": 0}
    unique = set(values)
    if len(unique) != 1:
        raise RuntimeError(
            f"target modules disagree on {attribute}: {sorted(unique)}")
    return {"effective": values[0], "module_count": len(values)}


def capture_runtime_provenance(runtime, args, expected_hip_sha256=None):
    """Capture execution choices that can change AR/TI token parity."""
    fused_k1_env = os.getenv("QUAROT_FUSED_K1")
    fused_projections_env = os.getenv("QUAROT_FUSED_PROJECTIONS")
    return {
        "git": _git_provenance(),
        "source_sha256": _runtime_source_sha256(),
        "hip_extension": _hip_extension_provenance(expected_hip_sha256),
        "runtime_flags": {
            "compile_mode": args.compile_mode,
            "native_gqa": runtime.native_gqa,
            "fused_decode_append": runtime.fused_decode_append,
            "exact_row_norm": runtime.exact_row_norm,
            "rowwise_lm_head": runtime.rowwise_lm_head,
            "exact_small_chunk_effective": runtime.exact_small_chunk,
            "exact_small_chunk_requested": args.exact_small_chunk,
            "fused_norm_quant": runtime.fused_norm_quant,
            "adaptive_k": runtime.adaptive_k,
            "quarot_fused_k1_env": fused_k1_env,
            "cache_fused_k1_effective": (
                "1" if fused_k1_env is None else fused_k1_env) != "0",
            "target_attention_fused_k1": _uniform_module_flag(
                runtime.target, "_fused_k1_enabled"),
            "quarot_fused_projections_env": fused_projections_env,
            "target_attention_fused_projections": _uniform_module_flag(
                runtime.target, "_fused_projections_enabled"),
            "qwen3_32b_grouped_nwaves_env": os.getenv(
                "QUAROT_QWEN3_32B_GROUPED_NWAVES"),
            "qwen3_32b_multi_nwaves_env": os.getenv(
                "QUAROT_QWEN3_32B_MULTI_NWAVES"),
        },
    }


def _hf_snapshot_identity(reference):
    """Return stable Hub model/revision fields without retaining a local path."""
    if not reference:
        return None, None
    text = str(reference)
    parts = Path(text).parts
    model_id = None
    revision = None
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            components = part.removeprefix("models--").split("--")
            if len(components) >= 2:
                model_id = "/".join(components)
        if part == "snapshots" and index + 1 < len(parts):
            revision = parts[index + 1]
    if model_id is None and not Path(text).is_absolute() and text.count("/") == 1:
        model_id = text
    return model_id, revision


def _normalized_sha256(value, label):
    if value is None:
        return None
    value = str(value).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef"
                               for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _qwen3_32b_checkpoint_contract(config, label="qwen3_32b"):
    expected_config = {
        "model_type": "qwen3_quarot",
        "quarot_checkpoint_format_version": 2,
        "quarot_ffn_format": "grouped_h256_v1",
        "quarot_activation_clip_ratio": 0.9,
        "quarot_rotation_format": "hadk_v1",
        "quarot_rotation_width": 5120,
        "quarot_rotation_remainder": 40,
        "quarot_rotation_inner": 128,
        "quarot_rotation_seed": 0,
        "quarot_rotation_device": "cuda",
        "quarot_rotation_dtype": "float32",
    }
    mismatches = {
        name: (config.get(name), expected)
        for name, expected in expected_config.items()
        if config.get(name) != expected
    }
    conversion = config.get("quarot_conversion")
    expected_conversion = {
        "seed": 0,
        "rotation_device": "cuda",
        "rotation_dtype": "float32",
        "w_bits": 4,
        "w_groupsize": -1,
        "w_asym": False,
        "w_clip": True,
    }
    method = conversion.get("method") if isinstance(conversion, dict) else None
    method_contracts = {
        "rtn": {
            "version": 2,
        },
        "gptq": {
            "version": 5,
            "dataset": "wikitext2",
            "nsamples": 128,
            "seqlen": 2048,
            "percdamp": 0.01,
            "act_order": False,
        },
    }
    if method not in method_contracts:
        mismatches["quarot_conversion.method"] = (
            method, sorted(method_contracts))
    else:
        expected_conversion.update(method_contracts[method])
    if not isinstance(conversion, dict):
        mismatches["quarot_conversion"] = (
            conversion, expected_conversion)
    else:
        for name, expected in expected_conversion.items():
            actual = conversion.get(name)
            if actual != expected:
                mismatches[f"quarot_conversion.{name}"] = (actual, expected)

    signs = config.get("quarot_rotation_signs")
    if (not isinstance(signs, list) or len(signs) != 5120
            or any(type(value) is not int or value not in (-1, 1)
                   for value in signs)):
        mismatches["quarot_rotation_signs"] = (
            None if not isinstance(signs, list) else len(signs),
            "5120 values in {-1,+1}")
    final_norm = config.get("quarot_final_norm_weight")
    if (not isinstance(final_norm, list) or len(final_norm) != 5120
            or any(type(value) not in (int, float)
                   or not math.isfinite(float(value))
                   for value in (final_norm if isinstance(final_norm, list)
                                 else ()))):
        mismatches["quarot_final_norm_weight"] = (
            None if not isinstance(final_norm, list) else len(final_norm),
            "5120 finite values")
    if mismatches:
        raise ValueError(
            f"{label} target W4A4KV4 checkpoint contract mismatch: "
            f"{mismatches}")
    return {
        "checkpoint_format_version": 2,
        "ffn_format": "grouped_h256_v1",
        "activation_clip_ratio": 0.9,
        "quant_method": method,
        "weight_bits": 4,
        "activation_bits": 4,
        "kv_bits": 4,
        "weight_groupsize": -1,
        "weight_symmetric": True,
        "weight_clip": True,
        "rotation_format": "hadk_v1",
        "rotation_remainder": 40,
        "rotation_inner": 128,
        "rotation_sign_count": len(signs),
        "final_norm_count": len(final_norm),
    }


def _qwen3_32b_expected_weight_keys(layer_count=64):
    keys = {"model.embed_tokens.weight", "lm_head.weight"}
    for index in range(layer_count):
        prefix = f"model.layers.{index}."
        keys.update({
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
        })
        for name in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj.1", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj"):
            keys.add(prefix + name + ".weight")
            keys.add(prefix + name + ".weight_scales")
    return keys


def _qwen3_32b_packed_index_contract(
        target, layer_count=64, label="qwen3_32b"):
    index_path = Path(target) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing packed target index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("packed target index has no weight_map")
    expected = _qwen3_32b_expected_weight_keys(layer_count)
    expected_map = {}
    for key in expected:
        if key.startswith("model.layers."):
            layer = int(key.split(".")[2])
            filename = f"model-layer-{layer:05d}.safetensors"
        else:
            filename = "model-global.safetensors"
        expected_map[key] = filename
    missing = sorted(expected - weight_map.keys())
    unexpected = sorted(weight_map.keys() - expected)
    misplaced = sorted(
        key for key, filename in expected_map.items()
        if weight_map.get(key) != filename)
    if missing or unexpected or misplaced:
        raise ValueError(
            f"{label} packed target weight map mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"misplaced={misplaced[:8]}")
    filenames = set(weight_map.values())
    expected_filenames = set(expected_map.values())
    if filenames != expected_filenames:
        raise ValueError(
            f"{label} packed target shard names mismatch: "
            f"actual={sorted(filenames)}, expected={sorted(expected_filenames)}")
    invalid_files = []
    for filename in filenames:
        relative = Path(filename)
        shard = Path(target) / relative
        if (relative.is_absolute() or len(relative.parts) != 1
                or not shard.is_file() or shard.stat().st_size <= 0):
            invalid_files.append(filename)
    if invalid_files:
        raise ValueError(
            f"{label} packed target has missing/invalid shards: "
            f"{sorted(invalid_files)[:8]}")
    return {
        "target_index_sha256": sha256(index_path),
        "target_weight_map_entries": len(weight_map),
        "target_shard_count": len(filenames),
    }


def target_profile_preflight(target, profile):
    """Bind a benchmark profile to an exact CPU-readable target artifact."""
    config_path = Path(target) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing target config: {config_path}")
    config = json.loads(config_path.read_text())
    architecture = {}
    for name in QWEN3_32B_ARCHITECTURE:
        value = config.get(name)
        try:
            value = int(value)
        except (TypeError, ValueError):
            pass
        architecture[name] = value

    checkpoint_contract = None
    expected_architecture = STRICT_PROFILE_ARCHITECTURES.get(profile)
    if expected_architecture is not None:
        mismatches = {
            name: (architecture[name], expected)
            for name, expected in expected_architecture.items()
            if architecture[name] != expected
        }
        if mismatches:
            raise ValueError(
                f"{profile} benchmark target architecture mismatch: "
                f"{mismatches}")
        checkpoint_contract = _qwen3_32b_checkpoint_contract(
            config, label=profile)
        checkpoint_contract.update(
            _qwen3_32b_packed_index_contract(
                target, layer_count=expected_architecture["num_hidden_layers"],
                label=profile))

    source_reference = next((
        config.get(name) for name in (
            "quarot_source_snapshot",
            "tokenizer_name_or_path",
            "base_model_name_or_path",
            "_name_or_path",
        ) if config.get(name)), None)
    inferred_model_id, inferred_revision = _hf_snapshot_identity(
        source_reference)
    source_model_id = config.get("quarot_source_model_id")
    source_revision = config.get("quarot_source_revision")
    if source_model_id and inferred_model_id and source_model_id != inferred_model_id:
        raise ValueError("target source model ID conflicts with its snapshot path")
    if source_revision and inferred_revision and source_revision != inferred_revision:
        raise ValueError("target source revision conflicts with its snapshot path")
    source_model_id = source_model_id or inferred_model_id
    source_revision = source_revision or inferred_revision

    source_config_sha256 = _normalized_sha256(
        config.get("quarot_source_config_sha256"),
        "quarot_source_config_sha256")
    source_index_sha256 = _normalized_sha256(
        config.get("quarot_source_index_sha256"),
        "quarot_source_index_sha256")
    if source_reference:
        source_config_path = Path(source_reference) / "config.json"
        if source_config_path.is_file():
            observed = sha256(source_config_path)
            if source_config_sha256 is not None and source_config_sha256 != observed:
                raise ValueError(
                    "target source config SHA256 conflicts with its snapshot")
            source_config_sha256 = observed
        source_index_path = (
            Path(source_reference) / "model.safetensors.index.json")
        if source_index_path.is_file():
            observed = sha256(source_index_path)
            if source_index_sha256 is not None and source_index_sha256 != observed:
                raise ValueError(
                    "target source index SHA256 conflicts with its snapshot")
            source_index_sha256 = observed

    expected_identity = STRICT_PROFILE_IDENTITIES.get(profile)
    if expected_identity is not None:
        actual_identity = {
            "source_model_id": source_model_id,
            "source_revision": source_revision,
            "source_config_sha256": source_config_sha256,
            "source_index_sha256": source_index_sha256,
        }
        mismatches = {
            name: (actual_identity[name], expected)
            for name, expected in expected_identity.items()
            if actual_identity[name] != expected
        }
        if mismatches:
            raise ValueError(
                f"{profile} target source provenance mismatch: "
                f"{mismatches}")

    return {
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_config_sha256": source_config_sha256,
        "source_index_sha256": source_index_sha256,
        "source_config": architecture,
        "target_model_type": config.get("model_type"),
        "target_config_sha256": sha256(config_path),
        "checkpoint_contract": checkpoint_contract,
    }


def load_prompts(dataset, root=DATA_ROOT):
    expected_count, expected_hash = DATASETS[dataset]
    path = Path(root) / f"{dataset}.jsonl"
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"missing or modified pinned AMD dataset: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_count:
        raise RuntimeError(f"{dataset} must contain exactly {expected_count} prompts")
    return [row["data"] for row in rows]


def preflight_gpu(max_preexisting_bytes=1 << 30):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is unavailable")
    free, total = torch.cuda.mem_get_info()
    used = total - free
    if used > max_preexisting_bytes:
        raise RuntimeError(
            f"GPU preflight refused benchmark: {used / 2**30:.2f} GiB is already in use "
            f"(limit {max_preexisting_bytes / 2**30:.2f} GiB); no process was terminated")
    return {
        "free_bytes": free,
        "total_bytes": total,
        "preexisting_bytes": used,
        # Preserve the original fields while exposing the richer schema used
        # by all later benchmark snapshots.
        "memory_snapshot": cuda_memory_snapshot("preflight"),
    }


def tokenize(tokenizer, text, device="cuda"):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt",
                                            enable_thinking=False)
    else:
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids
    return ids.to(device)


@torch.inference_mode()
def precompile_proposal_shapes(runtime):
    """Materialize every static PARD2 proposal graph outside measured runs.

    After the first round, ``draft_input`` contains the accepted prefix plus
    the target correction/bonus token. Appending ``draft_k - 1`` PARD masks
    therefore gives M in [draft_k, 2 * draft_k], inclusive. Explicitly
    exercising all 16 shapes makes a cold compile cache reproducible instead
    of letting a scored prompt pay an input-dependent Dynamo/Inductor cost.
    """
    if runtime.draft is None:
        return []
    device = next(runtime.draft.parameters()).device
    target_dtype = next(runtime.target.parameters()).dtype
    compiled = []
    k = runtime.spec.draft_k
    for rows in range(k, 2 * k + 1):
        print(f"[precompile] proposal M={rows}", flush=True)
        real_rows = rows - (k - 1)
        input_ids = torch.ones((1, rows), device=device, dtype=torch.long)
        input_ids[:, real_rows:] = runtime.spec.pard_token
        positions = torch.arange(1, rows + 1, device=device)
        cache = runtime._draft_cache()
        kwargs = {}
        if runtime.mode == "pard2-td":
            if runtime.td_unique_projection:
                width = runtime.draft.target_proj.out_features
                kwargs["projected_target_feat"] = torch.zeros(
                    (1, rows, width), device=device,
                    dtype=next(runtime.draft.parameters()).dtype)
            else:
                kwargs["target_feat"] = torch.zeros(
                    (1, rows, runtime.spec.target_dim), device=device,
                    dtype=target_dtype)
        prefix_kwargs = {name: value[:, :1] for name, value in kwargs.items()}
        runtime.draft(
            input_ids=torch.ones((1, 1), device=device, dtype=torch.long),
            past_key_values=cache, cache_position=torch.zeros(
                1, device=device, dtype=torch.long), use_cache=True,
            attention_mask=None, return_dict=True, **prefix_kwargs)
        runtime.draft_forward(
            input_ids=input_ids, past_key_values=cache,
            cache_position=positions, use_cache=True, attention_mask=None,
            return_dict=True, **kwargs)
        torch.cuda.synchronize()
        compiled.append(rows)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    return compiled


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--mode", required=True, choices=("ar", "pard2-ti", "pard2-td"))
    result.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    result.add_argument("--target", default=str(DEFAULT_TARGET))
    result.add_argument("--draft", default=str(DEFAULT_DRAFT))
    result.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    result.add_argument("--data-root", default=str(DATA_ROOT))
    result.add_argument("--generated-tokens", type=int, default=256)
    result.add_argument("--max-cache-len", type=int, default=8192,
                        help="target/draft cache capacity; default preserves the formal protocol")
    result.add_argument("--warmups", type=int, default=8)
    result.add_argument("--sweeps", type=int, default=3)
    result.add_argument("--limit", type=int,
                        help="smoke-only prompt limit; marks output unqualified")
    result.add_argument("--offset", type=int, default=0,
                        help="smoke-only starting prompt index; requires --limit")
    result.add_argument("--compile-mode", default="max-autotune")
    result.add_argument("--precompile-draft-shapes",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="seed all static proposal graphs before timing")
    result.add_argument("--output", required=True)
    result.add_argument("--expected-hip-sha256",
                        help="fail if the loaded quarot._HIP binary differs")

    result.add_argument("--ignore-eos", action="store_true")
    result.add_argument("--benchmark-profile", choices=BENCHMARK_PROFILES,
                        default="qwen3_8b",
                        help="target model and result/qualification contract")
    result.add_argument("--td-proxy-profile", choices=TD_PROXY_PROFILES,
                        help="explicit experimental cross-target TD contract")
    result.add_argument("--calibration")
    result.add_argument("--quantized-draft")
    result.add_argument("--adaptive-k", action="store_true")
    result.add_argument("--expanded-mha", action="store_true")
    result.add_argument("--fused-decode-append",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="unified prefill/decode/chunk KV4 writer")
    result.add_argument("--exact-row-norm",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="use row-independent HIP RMSNorm for exact chunk parity")
    result.add_argument("--rowwise-lm-head",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="run small-chunk LM-head as independent M=1 launches")
    result.add_argument("--exact-small-chunk",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="legacy alias overriding both independent controls")
    result.add_argument("--td-cache-basis",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="cache TD rotation signs and final norm weight on the target GPU")
    result.add_argument("--td-lazy-features",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="restore only accepted verifier feature rows")
    result.add_argument("--td-unique-projection",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="project real TD rows before expanding PARD mask features")
    result.add_argument("--td-basis-fold",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="fold inverse Hadamard/sign restoration into TD projection")
    result.add_argument("--fused-norm-quant",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="fuse exact layer RMSNorm with activation INT4 packing")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_cache_len <= 0:
        raise ValueError("--max-cache-len must be positive")
    expected_hip_sha256 = _normalized_sha256(
        args.expected_hip_sha256, "expected_hip_sha256")
    if args.td_proxy_profile is not None:
        if args.mode != "pard2-td":
            raise ValueError("--td-proxy-profile is valid only for pard2-td")
        if args.benchmark_profile != "qwen3_32b":
            raise ValueError(
                "the TD proxy profile requires --benchmark-profile qwen3_32b")
    target_provenance = target_profile_preflight(
        args.target, args.benchmark_profile)
    gpu = preflight_gpu()
    memory_snapshots = [gpu["memory_snapshot"]]
    verify_target_checkpoint(args.target, args.benchmark_profile)
    if args.mode != "ar":
        verify_local_resources(args.draft, args.benchmark_profile, args.mode)
    prompts = load_prompts(args.dataset, args.data_root)
    if args.offset and args.limit is None:
        raise ValueError("--offset requires --limit")
    if args.limit is not None:
        if args.limit <= 0 or args.offset < 0:
            raise ValueError("--limit must be positive and --offset non-negative")
        prompts = prompts[args.offset:args.offset + args.limit]
    runtime = load_runtime(mode=args.mode, target_checkpoint=args.target,
        draft_snapshot=args.draft, tokenizer_path=args.tokenizer,
        max_cache_len=args.max_cache_len, compile_mode=args.compile_mode,
        ignore_eos=args.ignore_eos, calibration_path=args.calibration,
        quantized_draft=args.quantized_draft, adaptive_k=args.adaptive_k,
        native_gqa=not args.expanded_mha,
        fused_decode_append=args.fused_decode_append,
        exact_row_norm=args.exact_row_norm,
        rowwise_lm_head=args.rowwise_lm_head,
        exact_small_chunk=args.exact_small_chunk,
        td_cache_basis=args.td_cache_basis,
        td_lazy_features=args.td_lazy_features,
        td_unique_projection=args.td_unique_projection,
        td_basis_fold=args.td_basis_fold,
        fused_norm_quant=args.fused_norm_quant,
        benchmark_profile=args.benchmark_profile,
        td_proxy_profile=args.td_proxy_profile)
    run_provenance = capture_runtime_provenance(
        runtime, args, expected_hip_sha256)
    memory_snapshots.append(cuda_memory_snapshot("runtime_loaded"))

    compiled_shapes = []
    if (args.mode != "ar" and args.compile_mode != "eager"
            and args.precompile_draft_shapes):
        if args.compile_mode != "max-autotune-no-cudagraphs":
            raise ValueError("full-shape precompile requires "
                             "max-autotune-no-cudagraphs")
        compiled_shapes = precompile_proposal_shapes(runtime)
    memory_snapshots.append(cuda_memory_snapshot("precompile_complete"))

    # Warmups are deliberately not selected from the scored 80/80/20 prompts.
    warmup_texts = [f"Warmup {index}: briefly explain integer {index}."
                    for index in range(args.warmups)]
    for text in warmup_texts:
        runtime.generate(tokenize(runtime.tokenizer, text), min(args.generated_tokens, 32))
        gc.collect(); torch.cuda.empty_cache()
    memory_snapshots.append(cuda_memory_snapshot("warmups_complete"))

    runs = []
    order = list(range(len(prompts)))
    for sweep in range(args.sweeps):
        random.Random(0x50415244 + sweep).shuffle(order)
        for index in order:
            input_ids = tokenize(runtime.tokenizer, prompts[index])
            result = runtime.generate(input_ids, args.generated_tokens)
            memory = cuda_memory_snapshot("run_complete")
            row = {"sweep": sweep, "prompt_index": index,
                   "input_token_count": int(input_ids.numel()),
                   "input_ids_sha256": token_ids_sha256(input_ids),
                   "output_ids": result.output_ids, **result.metrics()}
            # peak_vram_bytes remains the historical allocated-memory key.
            row["peak_vram_allocated_bytes"] = row["peak_vram_bytes"]
            row["peak_vram_reserved_bytes"] = memory["max_reserved_bytes"]
            row["memory_snapshot"] = memory
            runs.append(row)
            gc.collect(); torch.cuda.empty_cache()
    memory_snapshots.append(cuda_memory_snapshot("benchmark_complete"))
    target_alignment = (
        "cross_target_proxy" if args.td_proxy_profile is not None else "strict")
    qualification_track = (
        "experimental" if args.td_proxy_profile is not None else "canonical")
    target_revision = target_provenance["source_revision"]
    draft_revision = snapshot_revision(args.draft)
    if (draft_revision is None
            and Path(args.draft).resolve() == Path(DEFAULT_DRAFT).resolve()):
        draft_revision = DEFAULT_DRAFT_REVISION
    payload = {
        "contract": {"mode": args.mode, "dataset": args.dataset,
            "formal_protocol": args.limit is None,
            "qualified": (
                args.limit is None and args.td_proxy_profile is None),
            "prompt_count": len(prompts),
            "prompt_offset": args.offset,
            "generated_tokens": args.generated_tokens, "batch_size": 1,
            "max_cache_len": args.max_cache_len,
            "warmups": args.warmups, "sweeps": args.sweeps,
            "greedy": True, "ignore_eos": args.ignore_eos,
            "benchmark_profile": args.benchmark_profile,
            "qualification_track": qualification_track,
            "target_alignment": target_alignment,
            "td_proxy_profile": args.td_proxy_profile,
            "target": args.target,
            "target_revision": target_revision,
            "target_model_id": target_provenance["source_model_id"],
            "target_provenance": target_provenance,
            "input_ids_hash_format": INPUT_IDS_HASH_FORMAT,
            "runtime_provenance": run_provenance,
            "draft": args.draft if args.mode != "ar" else None,
            "draft_revision": (
                draft_revision if args.mode != "ar" else None),
            "tokenizer": args.tokenizer,
            "compile_mode": args.compile_mode,
            "upstream_commit": "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038",
            "precompile_draft_shapes": args.precompile_draft_shapes,
            "compiled_proposal_shapes": compiled_shapes,
            "td_cache_basis": args.td_cache_basis,
            "td_lazy_features": args.td_lazy_features,
            "td_unique_projection": args.td_unique_projection,
            "td_basis_fold": args.td_basis_fold,
            "fused_norm_quant": args.fused_norm_quant},
        "gpu_preflight": gpu,
        "memory_snapshots": memory_snapshots,
        "runs": runs,
        "median_steady_tokens_per_s": statistics.median(
            row["steady_tokens_per_s"] for row in runs),
        "median_end_to_end_tokens_per_s": statistics.median(
            row["end_to_end_tokens_per_s"] for row in runs),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
