"""Evaluate real QuaRot INT4 and FP16 models on paper downstream tasks."""
import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import datasets
import torch
import transformers

if not hasattr(datasets, "load_metric"):
    from evaluate import load as _load_metric
    datasets.load_metric = _load_metric

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm_eval
from e2e.evaluation import QuaRotHarnessLM
from e2e.model_registry import runtime_types, tokenizer_source

PAPER_TASKS = (
    "piqa", "winogrande", "hellaswag", "arc_easy",
    "arc_challenge", "lambada_openai",
)
PAPER_METRICS = {
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "lambada_openai": "acc,none",
}
PAPER_RESULTS = {
    "reference": {
        "piqa": 0.7911, "winogrande": 0.6906, "hellaswag": 0.7599,
        "arc_easy": 0.7458, "arc_challenge": 0.4625,
        "lambada_openai": 0.7390,
    },
    "int4": {
        "piqa": 0.7677, "winogrande": 0.6377, "hellaswag": 0.7216,
        "arc_easy": 0.6987, "arc_challenge": 0.4087,
        "lambada_openai": 0.7039,
    },
}


def load_int4(path):
    config_cls, int4_cls, _ = runtime_types(path, local_files_only=True)
    config = config_cls.from_pretrained(
        path, attn_implementation="flash_attention_2",
        local_files_only=True)
    return int4_cls.from_pretrained(
        path, config=config, torch_dtype=torch.float16,
        local_files_only=True).cuda().eval()


def load_reference(path, local_files_only=False):
    _, _, fp16_cls = runtime_types(
        path, local_files_only=local_files_only)
    return fp16_cls.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        local_files_only=local_files_only).cuda().eval()


def load_tokenizer(path, local_files_only=False):
    source = tokenizer_source(path, local_files_only=local_files_only)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source, use_fast=False, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer, source


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def metric_value(task, task_result):
    metric = PAPER_METRICS[task]
    if metric not in task_result:
        available = ", ".join(sorted(task_result))
        raise KeyError(
            f"{task} did not return {metric!r}; available: {available}")
    return float(task_result[metric]), metric


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        result, indent=2, default=json_ready) + "\n")


def evaluate_model(kind, model_path, args, result):
    tokenizer, tokenizer_path = load_tokenizer(
        model_path, args.local_files_only)
    model = (
        load_int4(model_path) if kind == "int4" else
        load_reference(model_path, args.local_files_only))
    adapter = QuaRotHarnessLM(
        model, tokenizer, batch_size=args.batch_size,
        max_length=args.max_length,
        kv_cache_dtype=(args.kv_cache_dtype if kind == "int4" else None))
    model_result = result["models"].setdefault(kind, {
        "model": str(model_path), "tokenizer": str(tokenizer_path),
        "tasks": {}, "summary": {},
    })
    for task in args.tasks:
        if args.resume and task in model_result["tasks"]:
            print(f"{kind}: {task} already complete", flush=True)
            continue
        print(f"{kind}: evaluating {task}", flush=True)
        raw = lm_eval.simple_evaluate(
            model=adapter, tasks=[task], num_fewshot=0,
            batch_size=args.batch_size, limit=args.limit,
            bootstrap_iters=args.bootstrap_iters, log_samples=False)
        task_result = raw["results"][task]
        value, metric = metric_value(task, task_result)
        model_result["tasks"][task] = {
            "metric": metric, "value": value,
            "stderr": task_result.get(metric.replace(",none", "_stderr,none")),
            "raw": task_result,
        }
        model_result["runtime"] = adapter.runtime_stats()
        write_result(args.output, result)
        print(f"{kind}: {task} {metric}={value:.6f}", flush=True)

    values = [model_result["tasks"][t]["value"] for t in args.tasks
              if t in model_result["tasks"]]
    model_result["summary"] = {
        "completed_tasks": len(values),
        "mean_accuracy": sum(values) / len(values) if values else None,
    }
    del adapter, model, tokenizer
    cleanup()
    write_result(args.output, result)


def parse_tasks(value):
    tasks = tuple(x.strip() for x in value.split(",") if x.strip())
    unknown = set(tasks) - set(PAPER_TASKS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported paper tasks: {sorted(unknown)}")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4-model", required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--int4-only", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype", choices=("int4", "float16"), default="int4",
        help="KV-cache storage used by the packed model (default: int4).")
    parser.add_argument(
        "--tasks", type=parse_tasks, default=PAPER_TASKS,
        help="Comma-separated subset of the six QuaRot paper tasks.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results/downstream/downstream_results.json"))
    args = parser.parse_args()
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    lm_eval.tasks.initialize_tasks()
    if not args.int4_only and not args.reference_model:
        parser.error("--reference-model is required unless --int4-only")
    if args.batch_size < 1 or args.max_length < 2:
        parser.error("batch size must be positive and max length at least 2")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    result = {
        "configuration": vars(args) | {"output": str(args.output)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "evaluate": importlib.metadata.version("evaluate"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "lm_eval_commit": "9b0b15b1ccace3534ffbd13298c569869ce8eaf3",
            "gpu": torch.cuda.get_device_name(0),
        },
        "protocol": {
            "num_fewshot": 0, "tasks": list(args.tasks),
            "metric_mapping": PAPER_METRICS,
            "paper_results": PAPER_RESULTS,
            "paper_int4_precision": "W4A4KV4",
            "effective_int4_precision": (
                "W4A4 plus {} KV token-by-token cached decode".format(
                    args.kv_cache_dtype)),
            "paper_equivalent_int4": False,
            "limited": args.limit is not None,
        },
        "models": {},
    }
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text())
        previous["configuration"] = result["configuration"]
        previous["environment"] = result["environment"]
        previous["protocol"] = result["protocol"]
        result = previous
    write_result(args.output, result)

    if not args.int4_only:
        evaluate_model(
            "reference", args.reference_model, args, result)
    evaluate_model("int4", args.int4_model, args, result)
    print(json.dumps(result, indent=2, default=json_ready))


if __name__ == "__main__":
    main()
