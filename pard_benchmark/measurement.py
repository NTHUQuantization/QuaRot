from __future__ import annotations

import platform
import resource
from pathlib import Path


MIB = 1024 * 1024


def current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)


def parameter_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def gpu_snapshot(torch, phase: str) -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "phase": phase,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "gpu_free_bytes": int(free),
        "gpu_total_bytes": int(total),
        "gpu_used_global_bytes": int(total - free),
        "host_rss_bytes": current_rss_bytes(),
    }


def environment_manifest(torch, transformers) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "transformers": transformers.__version__,
        "device": props.name or getattr(props, "gcnArchName", "unknown"),
        "device_total_memory": int(props.total_memory),
    }
