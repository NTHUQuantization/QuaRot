"""Sweep real Qwen3 QuaRot INT4 prefill/decode latency for one FHT build."""
import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.real_checkpoint import (
    deterministic_tokens, load_int4, validate_int4_checkpoint)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": ordered,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def csv_ints(value):
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4-model", required=True)
    parser.add_argument("--backend", required=True,
                        choices=("hadacore", "fast_op", "naive"))
    parser.add_argument("--batch-sizes", type=csv_ints, default=[1, 2, 4])
    parser.add_argument("--sequence-lengths", type=csv_ints,
                        default=[1, 16, 128, 512, 1024, 2048, 4096])
    parser.add_argument("--decode-steps", type=csv_ints, default=[1, 16, 64])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be nonnegative and --repeats positive")
    import quarot
    compiled_backend = quarot._HIP.fht_backend()
    if compiled_backend != args.backend:
        raise RuntimeError(
            f"requested backend {args.backend!r}, but extension was built "
            f"for {compiled_backend!r}")
    torch.manual_seed(0)
    model = load_int4(args.int4_model).cuda().eval()
    validation = validate_int4_checkpoint(model, args.int4_model)
    device = torch.device("cuda")
    results = {
        "environment": {
            "torch": torch.__version__, "hip": torch.version.hip,
            "gpu": (torch.cuda.get_device_name(0) or
                    torch.cuda.get_device_properties(0).gcnArchName),
        },
        "backend": args.backend,
        "checkpoint": str(Path(args.int4_model).resolve()),
        "checkpoint_validation": validation,
        "configuration": vars(args),
        "cases": [],
    }

    for batch in args.batch_sizes:
        for sequence in args.sequence_lengths:
            tokens = deterministic_tokens(batch, sequence, model.config.vocab_size, device)
            case = {"batch_size": batch, "sequence_length": sequence}
            try:
                def prefill():
                    model._expected_max_length = sequence + max(args.decode_steps)
                    return model(tokens, use_cache=True)
                case["prefill"] = measure(prefill, args.warmup, args.repeats)
                for steps in args.decode_steps:
                    next_token = torch.full((batch, 1), 100, device=device,
                                            dtype=torch.long)
                    model._expected_max_length = sequence + steps
                    seed = model(tokens, use_cache=True)
                    cache = seed.past_key_values
                    del seed
                    def decode(cache=cache, steps=steps):
                        cache.length = sequence
                        for _ in range(steps):
                            model(next_token, past_key_values=cache, use_cache=True)
                    timing = measure(decode, args.warmup, args.repeats)
                    timing["per_token_ms"] = timing["median_ms"] / steps
                    case.setdefault("decode", {})[str(steps)] = timing
                    del cache
                results["cases"].append(case)
                print(f"{args.backend}: B={batch} S={sequence} complete", flush=True)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                case["error"] = str(error)
                results["cases"].append(case)
                print(f"{args.backend}: B={batch} S={sequence} skipped: {error}", flush=True)
            cleanup()
            output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
