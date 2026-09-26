"""Attach an AMD-SMI summary to a TD32 stage JSON artifact."""
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gate-percent", type=float, default=95.0)
    args = parser.parse_args(argv)
    rows = []
    with args.csv.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {name: float(raw[name]) for name in FIELDS}
                row["timestamp"] = int(float(raw["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in row.values()):
                rows.append(row)
    if not rows:
        raise RuntimeError("AMD-SMI CSV contains no valid samples")
    payload = json.loads(args.result.read_text())
    summary = {
        "csv": str(args.csv),
        "csv_sha256": sha256(args.csv),
        "samples": len(rows),
        "first_timestamp": min(row["timestamp"] for row in rows),
        "last_timestamp": max(row["timestamp"] for row in rows),
        "peak_vram_mib": max(row["vram_used"] for row in rows),
        "peak_vram_percent": max(row["vram_percent"] for row in rows),
        "peak_hotspot_temperature_c": max(row["hotspot_temperature"] for row in rows),
        "peak_memory_temperature_c": max(row["memory_temperature"] for row in rows),
        "peak_power_usage_raw_w": max(row["power_usage"] for row in rows),
        "gate_percent": args.gate_percent,
    }
    summary["gate_passed"] = (
        summary["peak_vram_percent"] < args.gate_percent
        and summary["peak_hotspot_temperature_c"] < 110.0
        and summary["peak_memory_temperature_c"] < 108.0
    )
    if not summary["gate_passed"]:
        raise RuntimeError(f"external resource gate failed: {summary}")
    payload["external_monitor"] = summary
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.result.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
