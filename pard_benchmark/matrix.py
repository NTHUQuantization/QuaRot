from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


MODES = {
    "base": ["ar", "pard", "pard2-ti"],
    "instruct": ["ar", "pard", "pard2-ti", "pard2-td"],
}


def gpu_preflight(max_used_gib: float) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device is unavailable")
    free, total = torch.cuda.mem_get_info()
    used = total - free
    result = {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(used),
        "limit_bytes": int(max_used_gib * 2**30),
        "passed": used <= max_used_gib * 2**30,
    }
    return result


def cases_for_phase(phase: str, targets: list[str], out_dir: Path, compile_mode: str):
    if phase == "smoke":
        for target in targets:
            for mode in MODES[target]:
                draft_k = 1 if mode == "ar" else (12 if mode == "pard" else 15)
                yield target, mode, draft_k, 128, 16, 0, 1, "eager"
    elif phase == "tune":
        for target in targets:
            yield target, "ar", 1, 128, 128, 2, 3, compile_mode
            for mode in MODES[target]:
                if mode == "ar":
                    continue
                for draft_k in (4, 8, 12, 15):
                    yield target, mode, draft_k, 128, 128, 2, 3, compile_mode
    elif phase == "formal":
        selected = select_best_k(out_dir)
        for target in targets:
            for context in (128, 1024, 4096):
                yield target, "ar", 1, context, 128, 2, 5, compile_mode
                for mode in MODES[target]:
                    if mode == "ar":
                        continue
                    default_k = 12 if mode == "pard" else 15
                    yield target, mode, selected.get((target, mode, compile_mode), default_k), context, 128, 2, 5, compile_mode
    else:
        raise ValueError(phase)


def select_best_k(out_dir: Path) -> dict:
    groups = {}
    for path in out_dir.glob("tune_*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("mode") == "ar" or not data.get("repeats"):
            continue
        key = (data["target_key"], data["mode"], data["compile_mode"])
        score = statistics.median(x["steady_tokens_per_s"] for x in data["repeats"])
        if key not in groups or score > groups[key][0]:
            groups[key] = (score, int(data["draft_k"]))
    return {key: value[1] for key, value in groups.items()}


def run_phase(args, phase: str, compile_mode: str) -> list[dict]:
    failures = []
    for target, mode, k, context, tokens, warmups, repeats, case_compile in cases_for_phase(
        phase, args.targets, args.out_dir, compile_mode
    ):
        name = f"{phase}_{target}_{mode.replace('-', '_')}_k{k}_l{context}_{case_compile.replace('-', '_')}.json"
        output = args.out_dir / name
        if output.exists() and not args.force:
            continue
        cmd = [
            sys.executable,
            "-m",
            "pard_benchmark.benchmark",
            "--phase",
            phase,
            "--mode",
            mode,
            "--target",
            target,
            "--draft-k",
            str(k),
            "--context-len",
            str(context),
            "--max-new-tokens",
            str(tokens),
            "--warmups",
            str(warmups),
            "--repeats",
            str(repeats),
            "--compile-mode",
            case_compile,
            "--out-dir",
            str(args.out_dir),
            "--result-name",
            name,
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        log_path = args.out_dir / name.replace(".json", ".log")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode:
            failures.append({"case": name, "returncode": proc.returncode, "log": str(log_path)})
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run isolated PARD benchmark processes")
    parser.add_argument("--phase", choices=["smoke", "tune", "formal", "all"], default="smoke")
    parser.add_argument("--targets", default="base,instruct")
    parser.add_argument("--out-dir", type=Path, default=Path("pard_decode_results"))
    parser.add_argument("--compile-mode", choices=["eager", "reduce-overhead", "max-autotune"], default="max-autotune")
    parser.add_argument("--max-preexisting-vram-gib", type=float, default=1.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    invalid = set(args.targets) - set(MODES)
    if invalid:
        raise ValueError(f"invalid targets: {sorted(invalid)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_busy_gpu:
        preflight = gpu_preflight(args.max_preexisting_vram_gib)
        (args.out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
        if not preflight["passed"]:
            raise RuntimeError(
                f"GPU preflight failed: {preflight['used_bytes'] / 2**30:.2f} GiB is already used; "
                f"limit is {args.max_preexisting_vram_gib:.2f} GiB. External processes are not stopped."
            )
    phases = ["smoke", "tune", "formal"] if args.phase == "all" else [args.phase]
    all_failures = []
    compile_fallbacks = []
    for phase in phases:
        failures = run_phase(args, phase, "eager" if phase == "smoke" else args.compile_mode)
        if failures and phase != "smoke" and args.compile_mode != "eager":
            compile_fallbacks.extend(failures)
            all_failures.extend(run_phase(args, phase, "eager"))
        else:
            all_failures.extend(failures)
    if compile_fallbacks:
        (args.out_dir / "compile_fallbacks.json").write_text(json.dumps(compile_fallbacks, indent=2))
    from .report import summarize

    report = summarize(args.out_dir)
    result = {"failures": all_failures, "compile_fallbacks": compile_fallbacks, **report}
    print(json.dumps(result, indent=2))
    if all_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
