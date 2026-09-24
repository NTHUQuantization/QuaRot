"""Capture the reproducible CPU-only S0 TD32 source and resource contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--hip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = json.loads(args.data_manifest.read_text())
    if data["formal_overlap"]["count"] != 0:
        raise RuntimeError("formal overlap gate failed")
    for split in data["splits"].values():
        if sha256(split["path"]) != split["sha256"]:
            raise RuntimeError(f"data split SHA mismatch: {split['path']}")
    if any(not source["license"] for source in data["sources"]):
        raise RuntimeError("a calibration source has no license")

    source_names = (
        "e2e/prepare_qwen3_32b_td_data.py",
        "e2e/capture_qwen3_32b_td_contract.py",
        "e2e/qwen3_32b_td_features.py",
        "e2e/benchmark_qwen3_32b_td_heldout.py",
        "e2e/summarize_qwen3_32b_td_monitor.py",
        "e2e/run_qwen3_32b_td_monitored.sh",
        "e2e/speculative.py",
        "e2e/model_registry.py",
        "e2e/quantized_common.py",
        "e2e/quantized_qwen3/modeling_qwen3.py",
        "quarot/transformers/kv_cache.py",
    )
    source_hashes = {}
    for name in source_names:
        path = args.repo / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[name] = sha256(path)

    reference_index = args.reference / "model.safetensors.index.json"
    target_index = args.target / "model.safetensors.index.json"
    draft_config = args.draft / "config.json"
    draft_warp = args.draft / "warp_model.bin"
    for path in (reference_index, target_index, draft_config, draft_warp, args.hip):
        if not path.is_file():
            raise FileNotFoundError(path)

    disk = shutil.disk_usage(args.repo)
    reserve = 25 * 2**30
    projected_reference_cache = 6 * 2**30
    if disk.free - projected_reference_cache < reserve:
        raise RuntimeError("disk reserve gate fails after projected S2 cache")
    status = git(args.repo, "status", "--porcelain")
    payload = {
        "schema_version": 1,
        "stage": "TD32-S0-CONTRACT",
        "gate_passed": True,
        "git": {
            "head": git(args.repo, "rev-parse", "HEAD"),
            "branch": git(args.repo, "branch", "--show-current"),
            "dirty": bool(status),
            "porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "porcelain_entries": len(status.splitlines()),
        },
        "source_sha256": source_hashes,
        "data_manifest": str(args.data_manifest.resolve()),
        "data_manifest_sha256": sha256(args.data_manifest),
        "formal_overlap_count": 0,
        "artifacts": {
            "target": str(args.target.resolve()),
            "target_index_sha256": sha256(target_index),
            "reference": str(args.reference.resolve()),
            "reference_revision": args.reference.name,
            "reference_index_sha256": sha256(reference_index),
            "draft": str(args.draft.resolve()),
            "draft_revision": args.draft.name,
            "draft_config_sha256": sha256(draft_config),
            "draft_warp_sha256": sha256(draft_warp),
            "hip": str(args.hip.resolve()),
            "hip_sha256": sha256(args.hip),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "projected_reference_cache_bytes": projected_reference_cache,
            "minimum_reserve_bytes": reserve,
            "projected_free_after_cache_bytes": disk.free - projected_reference_cache,
        },
        "next_stage": "TD32-S1-EXTRACTOR-PROBE",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
