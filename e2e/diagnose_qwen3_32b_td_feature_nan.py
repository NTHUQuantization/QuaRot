"""Diagnose S1 selected-feature NaNs on one frozen calibration sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch

from e2e.qwen3_32b_td_features import UNION_TAPS, gpu_snapshot, load_quantized_target


def tensor_stats(value):
    finite = torch.isfinite(value)
    clean = value[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": int(finite.sum()),
        "elements": value.numel(),
        "nan": int(torch.isnan(value).sum()),
        "posinf": int(torch.isposinf(value).sum()),
        "neginf": int(torch.isneginf(value).sum()),
        "finite_min": float(clean.min()) if clean.numel() else None,
        "finite_max": float(clean.max()) if clean.numel() else None,
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from e2e.speculative import SelectedHiddenCollector, load_td_target_basis

    row = json.loads(args.data.read_text().splitlines()[0])
    ids = torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long)
    target = load_quantized_target(args.target)
    signs, final_norm = load_td_target_basis(target, args.source)
    collector = SelectedHiddenCollector(
        target, UNION_TAPS, signs, final_norm, cache_basis=True
    )
    payload = {
        "sample_id": row["sample_id"],
        "tokens": ids.numel(),
        "union_taps": UNION_TAPS,
        "memory_loaded": gpu_snapshot("loaded"),
        "paths": {},
    }
    for name in ("base_model", "causal_lm"):
        collector.reset()
        with torch.inference_mode():
            if name == "base_model":
                output = target.model(
                    input_ids=ids,
                    use_cache=False,
                    attention_mask=None,
                    output_hidden_states=False,
                    return_dict=True,
                )
                output_stats = tensor_stats(output.last_hidden_state)
            else:
                output = target(
                    input_ids=ids,
                    use_cache=False,
                    attention_mask=None,
                    output_hidden_states=False,
                    return_dict=True,
                )
                output_stats = tensor_stats(output.logits)
        raw = {
            str(tap): tensor_stats(collector.values[index])
            for tap, index in zip(UNION_TAPS, collector.indices)
        }
        features = collector.features()
        restored = {
            str(tap): tensor_stats(value)
            for tap, value in zip(UNION_TAPS, features.split(5120, dim=-1))
        }
        payload["paths"][name] = {
            "output": output_stats,
            "raw_layers": raw,
            "restored_layers": restored,
        }
    payload["memory_complete"] = gpu_snapshot("complete")
    collector.close()
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
