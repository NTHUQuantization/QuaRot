"""Serial AR/TI/TD/current-AR-repeat comparison. Wait for an idle GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from verification_optimization_20260924.run_ab import provenance, SWITCHES


def snapshot():
    process = subprocess.run(["rocm-smi", "--showuse", "--showpids", "--json"],
                             capture_output=True, text=True)
    record = dict(time=time.time(), stdout=process.stdout, stderr=process.stderr,
                  returncode=process.returncode, idle=False, pids=None)
    try:
        data = json.loads(process.stdout[process.stdout.index("{"):])
        cards = [value for key, value in data.items() if key.startswith("card")]
        pids = [key for key in data.get("system", {}) if key.startswith("PID")]
        record["pids"] = pids
        record["idle"] = (process.returncode == 0 and bool(cards) and not pids
                          and all(float(card["GPU use (%)"]) == 0 for card in cards))
    except (ValueError, KeyError, TypeError):
        pass
    return record


def wait_idle(path, timeout):
    started, consecutive, records = time.monotonic(), 0, []
    last_notice = 0
    while time.monotonic() - started < timeout:
        record = snapshot()
        records.append(record)
        path.write_text(json.dumps(records, indent=2) + "\n")
        consecutive = consecutive + 1 if record["idle"] else 0
        if consecutive == 3:
            return
        if time.monotonic() - last_notice > 55:
            print("waiting for three idle checks", path.name, record["pids"], flush=True)
            last_notice = time.monotonic()
        time.sleep(1 if record["idle"] else 10)
    raise RuntimeError("GPU stayed busy; no external process was stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--idle-timeout", type=int, default=3600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({name: "1" for name in SWITCHES})
    env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(REPO / "third-party/hadacore")))
    extension = list((REPO / "quarot").glob("_HIP*.so"))
    if len(extension) != 1:
        raise RuntimeError("expected exactly one built HIP extension")
    extension_sha = hashlib.sha256(extension[0].read_bytes()).hexdigest()
    manifest = dict(provenance=provenance(), flags={name: env[name] for name in SWITCHES},
                    extension_sha256=extension_sha, commands=[], protocol={
                        "limit": args.limit, "generated_tokens": args.generated_tokens,
                        "warmups": args.warmups, "sweeps": args.sweeps,
                        "dataset": "humaneval", "compile_mode": "eager",
                        "fused_norm_quant": True, "ignore_eos": True})
    for name, mode in (("ar", "ar"), ("pard2-ti", "pard2-ti"),
                       ("pard2-td", "pard2-td"), ("ar_repeat", "ar")):
        output = args.output / f"{name}.json"
        if output.exists():
            raise FileExistsError(f"use a fresh output directory: {output}")
        wait_idle(args.output / f"{name}.idle_preflight.json", args.idle_timeout)
        command = [sys.executable, str(Path(__file__).parent / "benchmark_phases.py"),
                   "--mode", mode, "--dataset", "humaneval", "--limit", str(args.limit),
                   "--generated-tokens", str(args.generated_tokens),
                   "--warmups", str(args.warmups), "--sweeps", str(args.sweeps),
                   "--compile-mode", "eager", "--ignore-eos", "--fused-norm-quant",
                   "--expected-hip-sha256", extension_sha, "--output", str(output)]
        manifest["commands"].append(command)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(name, "starting", flush=True)
        records, started = [], time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            while child.poll() is None:
                record = snapshot()
                records.append(record)
                (args.output / f"{name}.gpu_during.json").write_text(
                    json.dumps(records, indent=2) + "\n")
                if record["pids"] is not None and len(record["pids"]) > 1:
                    child.terminate()  # Only this runner's own benchmark.
                    child.wait()
                    raise RuntimeError("concurrent GPU process appeared; stopped own benchmark")
                time.sleep(5)
        print(name, "exit", child.returncode, "seconds", round(time.monotonic()-started, 2), flush=True)
        if child.returncode:
            raise SystemExit(f"{name} failed; inspect its log")
    print("all measurements complete", flush=True)


if __name__ == "__main__":
    main()
