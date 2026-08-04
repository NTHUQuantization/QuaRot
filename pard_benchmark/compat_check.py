from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compat import validate_configs
from .config import MODEL_SPECS, hf_token, load_hf_env


def main():
    parser = argparse.ArgumentParser(description="Download/read model configs and run PARD compatibility gates")
    parser.add_argument("--out", default="pard_decode_results/compatibility.json")
    parser.add_argument("--hf-env", default=".hf_env")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    load_hf_env(args.hf_env)
    from transformers import AutoConfig

    token = hf_token()
    configs = {}
    for key, spec in MODEL_SPECS.items():
        configs[key] = AutoConfig.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            token=token,
            local_files_only=args.local_files_only,
        )
    checks = []
    for target_key, modes in {
        "base": ["ar", "pard", "pard2-ti"],
        "instruct": ["ar", "pard", "pard2-ti", "pard2-td"],
    }.items():
        for mode in modes:
            draft_key = None if mode == "ar" else ("pard" if mode == "pard" else "pard2")
            result = validate_configs(
                mode,
                configs[target_key],
                configs[draft_key] if draft_key else None,
                MODEL_SPECS[target_key].model_id,
            )
            checks.append({"target": target_key, **result.to_dict()})
    payload = {
        "models": {
            key: {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "vocab_size": int(configs[key].vocab_size),
                "hidden_size": int(configs[key].hidden_size),
                "num_hidden_layers": int(configs[key].num_hidden_layers),
                "pard_token": getattr(configs[key], "pard_token", None),
                "spd_type": getattr(configs[key], "spd_type", None),
            }
            for key, spec in MODEL_SPECS.items()
        },
        "checks": checks,
        "all_compatible": all(x["compatible"] for x in checks),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "all_compatible": payload["all_compatible"]}))
    if not payload["all_compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
