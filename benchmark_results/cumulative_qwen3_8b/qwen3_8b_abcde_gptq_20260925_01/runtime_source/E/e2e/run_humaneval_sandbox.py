"""Execute prepared HumanEval programs in isolated, resource-bounded containers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid


FILENAME_RE = re.compile(r"s\d+_p\d{3}\.py")


def docker_output(args: list[str]) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True,
        timeout=30).stdout.strip()


def execute_one(path: Path, image: str, timeout_seconds: float) -> tuple[str, dict]:
    name = f"qwen32b-he-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    stat = path.stat()
    command = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "32",
        "--memory", "128m", "--memory-swap", "128m", "--cpus", "1",
        "--user", f"{stat.st_uid}:{stat.st_gid}",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--mount", f"type=bind,src={path.resolve()},dst=/candidate.py,readonly",
        image, "python", "-I", "-B", "/candidate.py",
    ]
    started = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds)
        exit_code = process.returncode
        stderr = process.stderr[-2000:]
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stderr = ((error.stderr or "")[-2000:] if isinstance(error.stderr, str)
                  else "")
        # The daemon can create the container shortly after the timed-out
        # docker client is killed. Retry by the exact unique name to close
        # that race without touching unrelated containers.
        for _ in range(50):
            removed = subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, text=True,
                timeout=10, check=False)
            if removed.returncode == 0:
                break
            time.sleep(0.1)
    return path.name, {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "stderr_tail": stderr,
    }


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="python:3.11-alpine")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or not 1 <= args.workers <= 8:
        raise ValueError("timeout must be positive and workers must be in [1, 8]")
    candidates = sorted(args.candidates_dir.glob("*.py"))
    if not candidates or any(not FILENAME_RE.fullmatch(path.name)
                             for path in candidates):
        raise ValueError("candidate directory is empty or has invalid filenames")
    image_id = docker_output(["image", "inspect", args.image,
                              "--format", "{{.Id}}"])
    statuses = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_one, path, args.image,
                               args.timeout_seconds) for path in candidates]
        for future in as_completed(futures):
            filename, status = future.result()
            statuses[filename] = status
            atomic_write(args.output, dict(sorted(statuses.items())))
    print(json.dumps({
        "image": args.image, "image_id": image_id,
        "candidates": len(statuses),
        "passed": sum(status["exit_code"] == 0 and
                      not status["timed_out"] for status in statuses.values()),
        "timed_out": sum(status["timed_out"] for status in statuses.values()),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
