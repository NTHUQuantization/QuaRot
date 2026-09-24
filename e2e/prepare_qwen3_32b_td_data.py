"""Build leakage-free S0 manifests for Qwen3-32B PARD-2 TD calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen


SOURCES = (
    {
        "stratum": "general",
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "mit",
        "config": "default",
        "split": "train_sft",
        "rows": 207865,
        "id_field": "prompt_id",
        "text_field": "prompt",
    },
    {
        "stratum": "code",
        "dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "revision": "b0079beaa0361d82412520b873715bee59cc7dd4",
        "license": "apache-2.0",
        "config": "default",
        "split": "train",
        "rows": 110000,
        "id_field": None,
        "text_field": "instruction",
    },
    {
        "stratum": "math",
        "dataset": "open-r1/OpenR1-Math-220k",
        "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
        "license": "apache-2.0",
        "config": "default",
        "split": "train",
        "rows": 93733,
        "id_field": "uuid",
        "text_field": "problem",
    },
)
FORMAL_HASHES = {
    "humaneval": "e16580cc87cac3168e59bde4ec1d0cd5cec7f31e1506c8ee1eba2ff914cf4368",
    "gsm8k": "56767ac321f1e80e3721f390c743a6a7253d4d5f50bd791339010ffcd8620371",
    "math_500": "4d8a669d746329ede429b1f4b037138ada9ae39ead6d0c34e1aa1be5349365a5",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def api_json(url: str):
    with urlopen(url, timeout=120) as response:
        return json.load(response)


def verify_source(source):
    metadata = api_json(f"https://huggingface.co/api/datasets/{source['dataset']}")
    actual = metadata.get("sha")
    if actual != source["revision"]:
        raise RuntimeError(
            f"dataset revision moved for {source['dataset']}: {actual}"
        )
    tags = set(metadata.get("tags") or ())
    card_license = (metadata.get("cardData") or {}).get("license")
    if source["license"] != card_license and f"license:{source['license']}" not in tags:
        raise RuntimeError(
            f"dataset license mismatch for {source['dataset']}: {card_license!r}"
        )


def fetch_rows(source, offset, length=100):
    query = urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "offset": offset,
            "length": length,
        }
    )
    payload = api_json(f"https://datasets-server.huggingface.co/rows?{query}")
    rows = payload.get("rows") or []
    if not rows:
        raise RuntimeError(f"Dataset Viewer returned no rows at offset {offset}")
    return rows


def qwen_prompt_ids(tokenizer, prompt: str, max_seq_length: int):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    return [int(token) for token in ids[:max_seq_length]]


def formal_contract(tokenizer):
    from e2e.benchmark_pard2 import DATASETS, load_prompts, tokenize

    if {name: digest for name, (_, digest) in DATASETS.items()} != FORMAL_HASHES:
        raise RuntimeError("formal dataset hashes changed from the frozen TD32 plan")
    text_hashes, token_hashes, normalized = set(), set(), []
    for dataset in DATASETS:
        for prompt in load_prompts(dataset):
            norm = normalized_text(prompt)
            normalized.append(norm)
            text_hashes.add(sha256_bytes(norm.encode("utf-8")))
            ids = tokenize(tokenizer, prompt).cpu().tolist()
            token_hashes.add(sha256_bytes(canonical_json(ids)))
    return text_hashes, token_hashes, normalized


def atomic_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-tokens", type=int, default=128000)
    parser.add_argument("--heldout-per-stratum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args(argv)
    if args.tune_tokens < 16000:
        raise ValueError("tune token budget must cover the 16k S1 prefix")
    if args.heldout_per_stratum < 1:
        raise ValueError("heldout-per-stratum must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    formal_text_hashes, formal_token_hashes, formal_normalized = formal_contract(
        tokenizer
    )
    per_stratum = math.ceil(args.tune_tokens / len(SOURCES))
    tune, heldout, source_manifests = [], [], []
    seen_sample_ids, seen_token_hashes = set(), set()
    overlap = {"text_hash": [], "token_hash": [], "substring": []}

    for source_index, source in enumerate(SOURCES):
        verify_source(source)
        local_seed = int(
            sha256_bytes(f"{args.seed}:{source['dataset']}".encode())[:16], 16
        )
        rng = random.Random(local_seed)
        start = rng.randrange(0, source["rows"] - 2000)
        selected_tune, selected_heldout = [], []
        selected_tokens, raw_hashes = 0, []
        offset = start
        while selected_tokens < per_stratum or len(selected_heldout) < args.heldout_per_stratum:
            for wrapped in fetch_rows(source, offset):
                row = wrapped["row"]
                prompt = str(row.get(source["text_field"]) or "").strip()
                if not prompt:
                    continue
                row_index = int(wrapped["row_idx"])
                raw_hashes.append(sha256_bytes(canonical_json(row)))
                sample_id = (
                    str(row.get(source["id_field"]))
                    if source["id_field"] and row.get(source["id_field"])
                    else f"row-{row_index}"
                )
                global_id = f"{source['dataset']}@{source['revision']}:{sample_id}"
                if global_id in seen_sample_ids:
                    continue
                ids = qwen_prompt_ids(tokenizer, prompt, args.max_seq_length)
                if len(ids) < 8:
                    continue
                norm = normalized_text(prompt)
                text_hash = sha256_bytes(norm.encode("utf-8"))
                token_hash = sha256_bytes(canonical_json(ids))
                if text_hash in formal_text_hashes:
                    overlap["text_hash"].append(global_id)
                if token_hash in formal_token_hashes:
                    overlap["token_hash"].append(global_id)
                for formal in formal_normalized:
                    shorter = min(len(norm), len(formal))
                    if shorter >= 80 and (norm in formal or formal in norm):
                        overlap["substring"].append(global_id)
                        break
                if token_hash in seen_token_hashes:
                    continue
                item = {
                    "sample_id": global_id,
                    "dataset": source["dataset"],
                    "source_revision": source["revision"],
                    "license": source["license"],
                    "config": source["config"],
                    "source_split": source["split"],
                    "row_index": row_index,
                    "stratum": source["stratum"],
                    "prompt": prompt,
                    "prompt_sha256": text_hash,
                    "input_ids": ids,
                    "token_ids_sha256": token_hash,
                    "token_count": len(ids),
                    "seed": args.seed,
                }
                seen_sample_ids.add(global_id)
                seen_token_hashes.add(token_hash)
                if selected_tokens < per_stratum:
                    item["split"] = "calibration_tune"
                    selected_tune.append(item)
                    selected_tokens += len(ids)
                elif len(selected_heldout) < args.heldout_per_stratum:
                    item["split"] = "heldout_validation"
                    selected_heldout.append(item)
                if selected_tokens >= per_stratum and len(selected_heldout) >= args.heldout_per_stratum:
                    break
            offset += 100
            if offset >= source["rows"] or offset - start > 2000:
                raise RuntimeError(f"insufficient rows for {source['dataset']}")
        tune.extend(selected_tune)
        heldout.extend(selected_heldout)
        source_manifests.append(
            {
                **{key: source[key] for key in (
                    "stratum", "dataset", "revision", "license", "config", "split"
                )},
                "start_offset": start,
                "last_offset_exclusive": offset,
                "selected_tune_rows": len(selected_tune),
                "selected_tune_tokens": selected_tokens,
                "selected_heldout_rows": len(selected_heldout),
                "viewer_row_payload_sha256": sha256_bytes(canonical_json(raw_hashes)),
            }
        )

    if any(overlap.values()):
        raise RuntimeError(f"formal overlap detected: {overlap}")
    tune.sort(key=lambda row: (row["stratum"], row["sample_id"]))
    heldout.sort(key=lambda row: (row["stratum"], row["sample_id"]))
    output_dir = Path(args.output_dir)
    tune_path = output_dir / "calibration_tune.jsonl"
    heldout_path = output_dir / "heldout_validation.jsonl"
    atomic_jsonl(tune_path, tune)
    atomic_jsonl(heldout_path, heldout)
    manifest = {
        "schema_version": 1,
        "stage": "TD32-S0-CONTRACT",
        "seed": args.seed,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_seq_length": args.max_seq_length,
        "formal_dataset_hashes": FORMAL_HASHES,
        "formal_overlap": {**overlap, "count": 0},
        "sources": source_manifests,
        "splits": {
            "calibration_tune": {
                "path": str(tune_path),
                "sha256": sha256_file(tune_path),
                "samples": len(tune),
                "tokens": sum(row["token_count"] for row in tune),
            },
            "heldout_validation": {
                "path": str(heldout_path),
                "sha256": sha256_file(heldout_path),
                "samples": len(heldout),
                "tokens": sum(row["token_count"] for row in heldout),
            },
        },
    }
    atomic_json(output_dir / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
