"""Attach an audited AMD SMI CSV summary to a PARD2 benchmark JSON."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile


FIELDS = (
    "vram_used", "vram_percent", "hotspot_temperature",
    "memory_temperature", "power_usage", "gfx", "mem",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    result = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {name: float(raw[name]) for name in FIELDS}
                row["timestamp"] = int(float(raw["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in row.values()):
                result.append(row)
    if not result:
        raise ValueError(f"AMD SMI CSV has no valid rows: {path}")
    return result


def summarize(result_path: Path, csv_path: Path, gate_percent: float) -> dict:
    if not 0.0 < gate_percent <= 100.0:
        raise ValueError("gate_percent must be in (0, 100]")
    payload = json.loads(result_path.read_text())
    total = int(payload["gpu_preflight"]["total_bytes"])
    rows = load_rows(csv_path)
    peak_mib = max(row["vram_used"] for row in rows)
    peak_bytes = int(round(peak_mib * 2**20))
    peak_percent = max(row["vram_percent"] for row in rows)
    computed_percent = 100.0 * peak_bytes / total
    if abs(computed_percent - peak_percent) > 0.15:
        raise ValueError(
            "AMD SMI VRAM MiB/percent columns disagree: "
            f"computed={computed_percent:.3f}, reported={peak_percent:.3f}")
    summary = {
        "csv": str(csv_path),
        "csv_sha256": sha256(csv_path),
        "samples": len(rows),
        "first_timestamp": min(row["timestamp"] for row in rows),
        "last_timestamp": max(row["timestamp"] for row in rows),
        "peak_device_used_bytes": peak_bytes,
        "peak_vram_mib": peak_mib,
        "peak_vram_percent": peak_percent,
        "peak_hotspot_temperature_c": max(
            row["hotspot_temperature"] for row in rows),
        "peak_memory_temperature_c": max(
            row["memory_temperature"] for row in rows),
        "peak_power_usage_raw_w": max(row["power_usage"] for row in rows),
        "peak_gfx_percent": max(row["gfx"] for row in rows),
        "peak_mem_percent": max(row["mem"] for row in rows),
        "gate_percent": gate_percent,
        "gate_passed": peak_percent < gate_percent,
    }
    payload["external_vram_monitor"] = summary
    return payload


def atomic_write(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gate-percent", type=float, default=90.0)
    args = parser.parse_args(argv)
    payload = summarize(args.result, args.csv, args.gate_percent)
    atomic_write(args.result, payload)
    print(json.dumps(payload["external_vram_monitor"], indent=2))


if __name__ == "__main__":
    main()
