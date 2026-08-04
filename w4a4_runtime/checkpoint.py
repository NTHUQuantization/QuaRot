from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from .config import W4A4Config


_PACKED_SUFFIXES = (".qweight", ".scales")


def _tensor_nbytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def save_sharded_checkpoint(
    tensors: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    model_config: dict,
    quantization_config: W4A4Config,
    max_shard_bytes: int = 2 * 1024**3,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to save packed checkpoints") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        size = _tensor_nbytes(value)
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = value
        current_bytes += size
    if current:
        shards.append(current)

    weight_map: dict[str, str] = {}
    total = len(shards)
    for index, shard in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(root / filename))
        weight_map.update({name: filename for name in shard})
    index = {
        "metadata": {"total_size": sum(_tensor_nbytes(value) for value in tensors.values())},
        "weight_map": weight_map,
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    (root / "config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True))
    (root / "quantization_config.json").write_text(
        json.dumps(quantization_config.to_dict(), indent=2, sort_keys=True)
    )


def load_checkpoint_tensors(
    path: str | Path,
    names: Iterable[str] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict, W4A4Config]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load packed checkpoints") from exc
    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    wanted = set(index["weight_map"]) if names is None else set(names)
    unknown = wanted - set(index["weight_map"])
    if unknown:
        raise KeyError(f"checkpoint tensors not found: {sorted(unknown)}")
    by_shard: dict[str, list[str]] = {}
    for name in wanted:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for filename, shard_names in by_shard.items():
        with safe_open(root / filename, framework="pt", device=str(device)) as handle:
            for name in shard_names:
                tensors[name] = handle.get_tensor(name)
    model_config = json.loads((root / "config.json").read_text())
    quant_config = W4A4Config.from_dict(json.loads((root / "quantization_config.json").read_text()))
    return tensors, model_config, quant_config


def audit_checkpoint(
    path: str | Path,
    *,
    max_bytes: int = int(4.5 * 1024**3),
    strict_model: bool = False,
) -> dict:
    """Validate the release checkpoint without materializing its tensors."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to audit packed checkpoints") from exc
    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    model_config = json.loads((root / "config.json").read_text())
    quant_config = W4A4Config.from_dict(json.loads((root / "quantization_config.json").read_text()))
    total_size = int(index.get("metadata", {}).get("total_size", 0))
    errors: list[str] = []
    if total_size > max_bytes:
        errors.append(f"packed tensors occupy {total_size} bytes, limit is {max_bytes}")
    unexpected = [name for name in index["weight_map"] if not name.endswith(_PACKED_SUFFIXES)]
    if unexpected:
        errors.append(f"non-packed tensors are present: {unexpected[:8]}")
    if strict_model:
        expected = {
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "lm_head.qweight",
            "lm_head.scales",
        }
        for layer in range(int(model_config.get("num_hidden_layers", 0))):
            root_name = f"model.layers.{layer}"
            for module in (
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj",
            ):
                expected.add(f"{root_name}.{module}.qweight")
                expected.add(f"{root_name}.{module}.scales")
        actual = set(index["weight_map"])
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f"missing packed model tensors: {sorted(missing)[:8]}")
        if extra:
            errors.append(f"unexpected packed model tensors: {sorted(extra)[:8]}")
    by_shard: dict[str, list[str]] = {}
    for name, filename in index["weight_map"].items():
        by_shard.setdefault(filename, []).append(name)
    for filename, names in by_shard.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name in names:
                shape = handle.get_slice(name).get_shape()
                dtype = handle.get_slice(name).get_dtype()
                if name.endswith(".qweight") and dtype != "U8":
                    errors.append(f"{name}: qweight dtype is {dtype}, expected U8")
                if name.endswith(".scales") and dtype != "F16":
                    errors.append(f"{name}: scales dtype is {dtype}, expected F16")
                if name.endswith(".qweight") and len(shape) == 4:
                    if shape[-2:] != [16, 64]:
                        errors.append(f"{name}: invalid linear tile {shape}")
    if errors:
        raise ValueError("invalid W4A4 checkpoint:\n- " + "\n- ".join(errors))
    return {
        "schema_version": quant_config.schema_version,
        "tensor_bytes": total_size,
        "tensor_gib": total_size / 1024**3,
        "tensor_count": len(index["weight_map"]),
        "shard_count": len(by_shard),
        "weight_quant_method": quant_config.weight_quant_method,
    }
