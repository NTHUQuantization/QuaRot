"""Score Qwen3-32B formal AR outputs against pinned task ground truth.

GSM8K uses numeric exact match, MATH-500 uses conservative normalized exact
match, and HumanEval prepares executable candidates whose sandbox exit status
is merged with --humaneval-status.  Benchmark timing is never mixed with
scoring time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DATASET_FILES = {
    "humaneval": "humaneval.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "math_500": "math_500.jsonl",
}
SOURCE_FILES = {
    "humaneval": "openai_humaneval_test.parquet",
    "gsm8k_test": "gsm8k_main_test.parquet",
    "gsm8k_train": "gsm8k_main_train.parquet",
}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*)?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_boxed(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.replace("<think>", "").replace("</think>", "")


def extract_numeric_answer(text: str) -> str | None:
    text = strip_thinking(text)
    boxed = last_boxed(text)
    if boxed is not None:
        values = NUMBER_RE.findall(boxed)
        if values:
            return values[-1]
    marker = re.findall(
        r"(?:final\s+answer|answer\s+is|####)\s*[:=]?\s*([^\n]+)",
        text, flags=re.I)
    if marker:
        values = NUMBER_RE.findall(marker[-1])
        if values:
            return values[-1]
    values = NUMBER_RE.findall(text)
    return values[-1] if values else None


def numeric_value(value: str | None) -> Fraction | None:
    if value is None:
        return None
    value = value.replace(",", "").strip()
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    value = strip_thinking(value).strip().strip("$.")
    boxed = last_boxed(value)
    if boxed is not None:
        value = boxed
    value = re.sub(r"\\(?:left|right)", "", value)
    value = re.sub(r"\\(?:,|!|;|quad|qquad)", "", value)
    value = value.replace(" ", "").replace("\n", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("^{\\circ}", "^\\circ").replace("^\u00b0", "^\\circ")
    return value.lower()


def extract_math_answer(text: str) -> str | None:
    text = strip_thinking(text)
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    marker = re.findall(
        r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*([^\n]+)",
        text, flags=re.I)
    if marker:
        return marker[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def math_equal(predicted: str | None, expected: str) -> bool:
    left, right = normalize_math(predicted), normalize_math(expected)
    if left is None:
        return False
    if left == right:
        return True
    left_number = numeric_value(extract_numeric_answer(left))
    right_number = numeric_value(extract_numeric_answer(right))
    return (left_number is not None and right_number is not None
            and left_number == right_number
            and NUMBER_RE.fullmatch(left) is not None
            and NUMBER_RE.fullmatch(right) is not None)


def entry_point(prompt: str) -> str:
    names = re.findall(r"(?m)^def\s+([A-Za-z_]\w*)\s*\(", prompt)
    if not names:
        raise ValueError("HumanEval prompt has no top-level function")
    return names[-1]


def python_prompt(prompt: str) -> str:
    match = re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+|def\s+|class\s+)", prompt)
    if not match:
        raise ValueError("HumanEval prompt has no Python source start")
    return prompt[match.start():]


def code_candidate(prompt: str, completion: str, name: str) -> str:
    completion = strip_thinking(completion)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion,
                        flags=re.S | re.I)
    preferred = [block for block in blocks
                 if re.search(rf"(?m)^def\s+{re.escape(name)}\s*\(", block)]
    if preferred:
        return preferred[-1].strip() + "\n"
    if blocks:
        block = max(blocks, key=len).strip()
        if re.search(r"(?m)^def\s+", block):
            return block + "\n"
    function = re.search(
        rf"(?ms)^def\s+{re.escape(name)}\s*\(.*", completion)
    if function:
        return function.group(0).strip() + "\n"
    return python_prompt(prompt).rstrip() + "\n" + completion.rstrip() + "\n"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def benchmark_rows(payload: dict) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for row in payload["runs"]:
        grouped[int(row["prompt_index"])].append(row)
    sweeps = int(payload["contract"]["sweeps"])
    expected = set(range(sweeps))
    for prompt, rows in grouped.items():
        actual = {int(row["sweep"]) for row in rows}
        if actual != expected or len(rows) != sweeps:
            raise ValueError(
                f"prompt {prompt} sweep rows mismatch: {sorted(actual)}")
        rows.sort(key=lambda row: int(row["sweep"]))
    return dict(grouped)


def summary(records: list[dict], sweeps: int) -> dict:
    by_sweep = {}
    for sweep in range(sweeps):
        rows = [row for row in records if row["sweep"] == sweep]
        scored = [row for row in rows if row.get("correct") is not None]
        by_sweep[str(sweep)] = {
            "correct": sum(bool(row["correct"]) for row in scored),
            "total": len(scored),
            "accuracy": (sum(bool(row["correct"]) for row in scored) / len(scored)
                         if scored else None),
        }
    scored = [row for row in records if row.get("correct") is not None]
    return {
        "correct": sum(bool(row["correct"]) for row in scored),
        "total": len(scored),
        "accuracy": (sum(bool(row["correct"]) for row in scored) / len(scored)
                     if scored else None),
        "by_sweep": by_sweep,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--humaneval-candidates-dir", type=Path)
    parser.add_argument("--humaneval-status", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.benchmark.read_text())
    contract = payload["contract"]
    dataset = contract["dataset"]
    if dataset not in DATASET_FILES:
        raise ValueError(f"unsupported dataset: {dataset}")
    if contract["mode"] != "ar" or not contract["formal_protocol"]:
        raise ValueError("task scoring requires a formal AR benchmark payload")
    prompts = load_jsonl(args.data_root / DATASET_FILES[dataset])
    grouped = benchmark_rows(payload)
    if set(grouped) != set(range(len(prompts))):
        raise ValueError("benchmark prompt indices do not cover pinned dataset")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    statuses = (json.loads(args.humaneval_status.read_text())
                if args.humaneval_status else {})
    records = []
    alignment = {}

    if dataset == "gsm8k":
        source_rows = []
        for key in ("gsm8k_test", "gsm8k_train"):
            path = args.ground_truth_dir / SOURCE_FILES[key]
            split = key.rsplit("_", 1)[-1]
            source_rows.extend((row["question"], row["answer"], split)
                               for row in pq.read_table(path).to_pylist())
        ground_truth = {question: (answer, split)
                        for question, answer, split in source_rows}
        for index, prompt in enumerate(prompts):
            question = prompt["data"]
            if question not in ground_truth:
                raise ValueError(f"GSM8K prompt {index} has no exact source match")
            answer, split = ground_truth[question]
            expected = answer.rsplit("####", 1)[-1].strip()
            alignment[str(index)] = {"split": split, "expected": expected}
            for row in grouped[index]:
                decoded = tokenizer.decode(row["output_ids"], skip_special_tokens=True)
                predicted = extract_numeric_answer(decoded)
                records.append({
                    "prompt_index": index, "sweep": int(row["sweep"]),
                    "prediction": predicted, "expected": expected,
                    "correct": (numeric_value(predicted)
                                == numeric_value(expected)),
                    "decoded_output": decoded,
                })
        metric = "numeric_exact_match"

    elif dataset == "math_500":
        for index, prompt in enumerate(prompts):
            expected = prompt["answer"]
            alignment[str(index)] = {
                "unique_id": prompt["unique_id"], "expected": expected}
            for row in grouped[index]:
                decoded = tokenizer.decode(row["output_ids"], skip_special_tokens=True)
                predicted = extract_math_answer(decoded)
                records.append({
                    "prompt_index": index, "sweep": int(row["sweep"]),
                    "prediction": predicted, "expected": expected,
                    "correct": math_equal(predicted, expected),
                    "decoded_output": decoded,
                })
        metric = "conservative_normalized_exact_match"

    else:
        source = pq.read_table(
            args.ground_truth_dir / SOURCE_FILES["humaneval"]).to_pylist()
        ground_truth = {row["entry_point"]: row for row in source}
        if args.humaneval_candidates_dir is None:
            raise ValueError("HumanEval requires --humaneval-candidates-dir")
        args.humaneval_candidates_dir.mkdir(parents=True, exist_ok=True)
        for index, prompt in enumerate(prompts):
            raw_prompt = prompt["data"]
            name = entry_point(raw_prompt)
            if name not in ground_truth:
                raise ValueError(
                    f"HumanEval prompt {index} entry point {name!r} is unmatched")
            truth = ground_truth[name]
            alignment[str(index)] = {
                "task_id": truth["task_id"], "entry_point": name}
            for row in grouped[index]:
                decoded = tokenizer.decode(row["output_ids"], skip_special_tokens=True)
                candidate = code_candidate(raw_prompt, decoded, name)
                program = (candidate.rstrip() + "\n" + truth["test"].rstrip()
                           + f"\ncheck({name})\n")
                filename = f"s{int(row['sweep'])}_p{index:03d}.py"
                path = args.humaneval_candidates_dir / filename
                path.write_text(program)
                status = statuses.get(filename)
                correct = None if status is None else (
                    not status.get("timed_out", False)
                    and int(status.get("exit_code", 1)) == 0)
                records.append({
                    "prompt_index": index, "sweep": int(row["sweep"]),
                    "task_id": truth["task_id"], "entry_point": name,
                    "candidate_file": filename,
                    "candidate_sha256": sha256(path),
                    "correct": correct, "sandbox_status": status,
                    "decoded_output": decoded,
                })
        metric = "execution_pass_at_1"

    cross_sweep_exact = all(
        len({tuple(row["output_ids"]) for row in rows}) == 1
        for rows in grouped.values())
    result = {
        "status": ("pending_humaneval_sandbox"
                   if dataset == "humaneval" and not statuses else "complete"),
        "dataset": dataset,
        "metric": metric,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark),
        "ground_truth_sources": {
            name: {"path": str(args.ground_truth_dir / filename),
                   "sha256": sha256(args.ground_truth_dir / filename)}
            for name, filename in SOURCE_FILES.items()
            if (args.ground_truth_dir / filename).is_file()
        },
        "prompt_count": len(prompts),
        "sweeps": int(contract["sweeps"]),
        "cross_sweep_output_exact": cross_sweep_exact,
        "alignment": alignment,
        "summary": summary(records, int(contract["sweeps"])),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "dataset", "metric", "prompt_count",
        "cross_sweep_output_exact", "summary")}, indent=2))


if __name__ == "__main__":
    main()
