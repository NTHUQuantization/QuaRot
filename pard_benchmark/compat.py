from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import OFFICIAL_TD_TARGET


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    mode: str
    reason: str
    target_vocab_size: int
    draft_vocab_size: int | None
    pard_token: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _get(config, name, default=None):
    return getattr(config, name, default)


def validate_configs(mode: str, target_config, draft_config=None, target_id: str = "") -> CompatibilityResult:
    if mode not in {"ar", "pard", "pard2-ti", "pard2-td"}:
        raise ValueError(f"unsupported mode: {mode}")
    tv = int(_get(target_config, "vocab_size", -1))
    if _get(target_config, "model_type") != "llama":
        return CompatibilityResult(False, mode, "target must be a Llama model", tv, None, None)
    if mode == "ar":
        return CompatibilityResult(True, mode, "target-only autoregressive baseline", tv, None, None)
    if draft_config is None:
        raise ValueError("draft_config is required for speculative modes")
    dv = int(_get(draft_config, "vocab_size", -1))
    pard_token = _get(draft_config, "pard_token")
    if tv != dv:
        return CompatibilityResult(False, mode, f"vocab mismatch: target={tv}, draft={dv}", tv, dv, pard_token)
    if pard_token is None or not 0 <= int(pard_token) < dv:
        return CompatibilityResult(False, mode, f"invalid pard_token: {pard_token}", tv, dv, pard_token)
    if mode == "pard" and _get(draft_config, "spd_type") != "pard":
        return CompatibilityResult(False, mode, "draft config is not PARD", tv, dv, pard_token)
    if mode.startswith("pard2") and not (
        bool(_get(draft_config, "pard2", False)) or _get(draft_config, "spd_type") == "pard2"
    ):
        return CompatibilityResult(False, mode, "draft config is not PARD2", tv, dv, pard_token)
    if mode == "pard2-td":
        if target_id != OFFICIAL_TD_TARGET:
            return CompatibilityResult(
                False,
                mode,
                f"target-dependent evaluation is restricted to {OFFICIAL_TD_TARGET}",
                tv,
                dv,
                pard_token,
            )
        hidden = int(_get(target_config, "hidden_size", -1))
        layers = int(_get(target_config, "num_hidden_layers", -1))
        selected = list(_get(draft_config, "pard2_target_layers", []))
        target_dim = int(_get(draft_config, "pard2_target_dim", -1))
        if hidden != 4096 or layers != 32:
            return CompatibilityResult(False, mode, f"expected 32x4096 target, got {layers}x{hidden}", tv, dv, pard_token)
        if selected != [-1, -8, -16, -24] or len(selected) * hidden != target_dim:
            return CompatibilityResult(
                False,
                mode,
                f"PARD2 target feature mismatch: layers={selected}, target_dim={target_dim}",
                tv,
                dv,
                pard_token,
            )
    return CompatibilityResult(True, mode, "configuration compatibility checks passed", tv, dv, int(pard_token))
