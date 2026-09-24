"""Serial, explicit-switch A/B runner. Execute inside the existing ROCm image.

The default is a diagnostic smoke protocol, never a formal quality/speed claim.
Each subprocess starts from a clean runtime and records its exact environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SWITCHES = ("QUAROT_BATCHED_H128", "QUAROT_STATIC_KV_METADATA",
            "QUAROT_VERIFICATION_GRAPH", "QUAROT_CHUNK_PREPROCESS")
# Norm is the existing runtime option, because it chooses layer wrappers at
# construction. Graph has a real fixed-metadata dependency, stated explicitly.
STAGES = {
    "baseline": (0, 0, 0, 0, False),
    "had_only": (1, 0, 0, 0, False),
    "norm_only": (0, 0, 0, 0, True),
    "metadata_only": (0, 1, 0, 0, False),
    "graph_with_metadata": (0, 1, 1, 0, False),
    "chunk_only": (0, 0, 0, 1, False),
    "had_norm": (1, 0, 0, 0, True),
    "had_norm_metadata": (1, 1, 0, 0, True),
    "had_norm_metadata_graph": (1, 1, 1, 0, True),
    "all": (1, 1, 1, 1, True),
    "baseline_repeat": (0, 0, 0, 0, False),
}


def idle_snapshot(destination, timeout=60):
    """Require both zero utilization and no KFD processes before loading torch."""
    started = time.monotonic()
    snapshots, consecutive = [], 0
    while time.monotonic() - started < timeout:
        completed = subprocess.run(["rocm-smi", "--showuse", "--showpids", "--json"],
                                   capture_output=True, text=True)
        raw = completed.stdout
        record = {"unix_time": time.time(), "stdout": raw, "stderr": completed.stderr,
                  "exit_code": completed.returncode}
        snapshots.append(record)
        try:
            data = json.loads(raw[raw.index("{"):])
            cards = [value for key, value in data.items() if key.startswith("card")]
            busy_pids = [key for key in data.get("system", {}) if key.startswith("PID")]
            idle = (completed.returncode == 0 and cards and not busy_pids and
                    all(float(card["GPU use (%)"]) == 0 for card in cards))
        except (ValueError, KeyError, TypeError):
            idle = False
        record["idle"] = bool(idle)
        destination.write_text(json.dumps(snapshots, indent=2) + "\n")
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 3:
            return
        time.sleep(1)
    raise SystemExit(f"GPU was not idle for three checks; inspect {destination}. No process was terminated.")


def provenance():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, check=True).stdout.decode().strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO,
                          capture_output=True, check=True).stdout
    sources = ["e2e/speculative.py", "e2e/verification_graph.py", "e2e/benchmark_pard2.py",
               "e2e/quantized_common.py", "quarot/transformers/kv_cache.py",
               "quarot/kernels/bindings.cpp", "quarot/kernels/fused_hip.hip",
               "quarot/kernels/verification_preprocess.hip",
               "quarot/kernels/verification_metadata.hip", "setup.py"]
    return {"git_head": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                              for name in sources if (REPO / name).is_file()},
            "python": sys.version, "executable": sys.executable,
            "build_command": "MAX_JOBS=4 python setup.py build_ext --inplace"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=list(STAGES))
    parser.add_argument("--parts", nargs="+", choices=("target", "pard2"),
                        default=["target", "pard2"])
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--reference", type=Path,
                        help="baseline target artifacts; defaults to output/baseline/target/artifacts.pt")
    parser.add_argument("--model")
    parser.add_argument("--draft")
    parser.add_argument("--tokenizer")
    parser.add_argument("--contexts", nargs="+", type=int, default=[127, 128, 129])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dataset", default="humaneval", choices=("humaneval", "gsm8k", "math_500"))
    parser.add_argument("--modes", nargs="+", choices=("ar", "pard2-ti", "pard2-td"),
                        default=["pard2-ti", "pard2-td"])
    parser.add_argument("--compile-mode", default="eager")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = args.reference or args.output / "baseline/target/artifacts.pt"
    for stage in args.stages:
        values = STAGES[stage]
        env = os.environ.copy()
        env.update({key: str(value) for key, value in zip(SWITCHES, values[:4])})
        # Residual-add fusion is a separate primitive experiment; no runtime
        # flag currently enables it in the model or PARD-2 measurements.
        env.pop("QUAROT_VERIFY_RESIDUAL_NORM", None)
        env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(REPO / "third-party/hadacore"),
                                              env.get("PYTHONPATH", "")))
        destination = args.output / stage
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {"stage": stage, "flags": {key: env[key] for key in SWITCHES},
                    "fused_norm_quant": values[4], "commands": []}
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("flags") == manifest["flags"]:
                manifest["commands"] = previous.get("commands", [])
        manifest["last_invocation"] = {key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()}

        def run(command, name, result_file):
            if args.skip_complete and result_file.exists():
                prior = json.loads(result_file.read_text())
                if prior.get("passed") is True or ("runs" in prior and "contract" in prior):
                    return
            idle_snapshot(destination / f"{name}.idle_preflight.json")
            manifest["provenance"] = provenance()
            manifest["commands"].append(command)
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(stage, name, "starting", flush=True)
            before = time.time()
            with (destination / f"{name}.log").open("w") as log:
                completed = subprocess.run(command, cwd=REPO, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            print(stage, name, "exit", completed.returncode,
                  "seconds", round(time.time() - before, 2), flush=True)
            if completed.returncode:
                raise SystemExit(f"{stage}/{name} failed; inspect {destination / (name + '.log')}")

        if "target" in args.parts:
            command = [sys.executable, str(HERE / "target_probe.py"),
                       "--output", str(destination / "target"), "--samples", str(args.samples),
                       "--contexts", *map(str, args.contexts)]
            if args.model:
                command += ["--model", args.model]
            if args.tokenizer:
                command += ["--tokenizer", args.tokenizer]
            if values[4]:
                command += ["--fused-norm-quant"]
            if stage != "baseline":
                if not reference.exists():
                    raise SystemExit(f"baseline artifact does not exist: {reference}")
                command += ["--reference", str(reference)]
            run(command, "target", destination / "target/target.json")
        if "pard2" in args.parts:
            for mode in args.modes:
                result_file = destination / f"{mode}_{args.dataset}.json"
                command = [sys.executable, str(REPO / "e2e/benchmark_pard2.py"),
                           "--mode", mode, "--dataset", args.dataset,
                           "--output", str(result_file), "--limit", str(args.limit),
                           "--generated-tokens", str(args.generated_tokens),
                           "--warmups", str(args.warmups), "--sweeps", str(args.sweeps),
                           "--compile-mode", args.compile_mode, "--ignore-eos",
                           "--fused-norm-quant" if values[4] else "--no-fused-norm-quant"]
                for key in ("model", "draft", "tokenizer"):
                    if getattr(args, key):
                        command += ["--target" if key == "model" else "--" + key,
                                    getattr(args, key)]
                run(command, mode, result_file)


if __name__ == "__main__":
    main()
