import csv
import json

import pytest

from e2e.attach_amd_smi_monitor import atomic_write, load_rows, summarize


HEADER = [
    "timestamp", "gpu", "xcp", "power_usage", "max_power",
    "hotspot_temperature", "memory_temperature", "gfx_clk", "gfx",
    "mem", "mem_clock", "vram_used", "vram_free", "vram_total",
    "vram_percent",
]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def row(timestamp, used, percent, hotspot=80, memory=70):
    return {
        "timestamp": timestamp, "gpu": 0, "xcp": 0,
        "power_usage": 200, "max_power": 300,
        "hotspot_temperature": hotspot, "memory_temperature": memory,
        "gfx_clk": 2000, "gfx": 98, "mem": 40, "mem_clock": 900,
        "vram_used": used, "vram_free": 32624 - used,
        "vram_total": 32624, "vram_percent": percent,
    }


def test_summarize_and_atomic_write_external_monitor(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "gpu_preflight": {"total_bytes": 32624 * 2**20}}))
    monitor = tmp_path / "monitor.csv"
    write_csv(monitor, [
        row(100, 27000, 82.76, hotspot=90, memory=79),
        row(101, 28000, 85.82, hotspot=95, memory=82),
    ])

    payload = summarize(result, monitor, 95.0)
    summary = payload["external_vram_monitor"]
    assert summary["samples"] == 2
    assert summary["peak_vram_mib"] == 28000
    assert summary["peak_vram_percent"] == 85.82
    assert summary["peak_hotspot_temperature_c"] == 95
    assert summary["peak_memory_temperature_c"] == 82
    assert summary["gate_passed"] is True
    atomic_write(result, payload)
    assert json.loads(result.read_text()) == payload


def test_load_rows_skips_malformed_rows(tmp_path):
    monitor = tmp_path / "monitor.csv"
    write_csv(monitor, [row(100, 28000, 85.82), row("bad", 1, 1)])
    rows = load_rows(monitor)
    assert len(rows) == 1
    assert rows[0]["timestamp"] == 100


def test_summarize_rejects_disagreeing_vram_columns(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "gpu_preflight": {"total_bytes": 32624 * 2**20}}))
    monitor = tmp_path / "monitor.csv"
    write_csv(monitor, [row(100, 28000, 50.0)])
    with pytest.raises(ValueError, match="VRAM MiB/percent columns disagree"):
        summarize(result, monitor, 95.0)
