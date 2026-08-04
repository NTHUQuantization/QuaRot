from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_COMMIT = "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"
OFFICIAL_TD_TARGET = "unsloth/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


MODEL_SPECS = {
    "base": ModelSpec(
        "meta-llama/Llama-3.1-8B",
        "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    ),
    "instruct": ModelSpec(
        OFFICIAL_TD_TARGET,
        "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
    ),
    "pard": ModelSpec(
        "amd/PARD-Llama-3.2-1B",
        "f7fbf130e2569268b054ffa2234303070ae82f9b",
    ),
    "pard2": ModelSpec(
        "amd/PARD2-Llama-3.1-8B",
        "a509e315bfe79a0be8b832004bb3ee14acfdf86a",
    ),
}


def load_hf_env(path: str | Path = ".hf_env") -> dict[str, str]:
    """Load only the supported HF variables without overriding the environment."""
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    allowed = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME"}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        value = value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = "<loaded>"
    return loaded


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
