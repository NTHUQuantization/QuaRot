"""Build the cumulative, leakage-free 1M-token S3--S5 training split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

from e2e.prepare_qwen3_32b_td_data import (
    SOURCES,
    canonical_json,
    fetch_rows,
    formal_contract,
    normalized_text,
    sha256_bytes,
    sha256_file,
    verify_source,
)


TRAIN_SEED = 20260829


def response_messages(source, row):
    if source["stratum"] == "general":
        messages = row.get("messages") or []
        pair = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                pair.append({"role": role, "content": content})
            if len(pair) >= 2 and pair[-2]["role"] == "user" and pair[-1]["role"] == "assistant":
                return pair[-2:]
        return None
    prompt = str(row.get(source["text_field"]) or "").strip()
    if source["stratum"] == "code":
        answer = str(row.get("response") or "").strip()
    else:
        generations = row.get("generations") or []
        answer = str(generations[0] if generations else row.get("solution") or "").strip()
    if not prompt or not answer:
        return None
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def tokenize_example(tokenizer, messages, max_seq_length):
    full = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
    kwargs = dict(tokenize=True, add_generation_prompt=False, enable_thinking=False)
    try:
        ids = tokenizer.apply_chat_template(full, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        ids = tokenizer.apply_chat_template(full, **kwargs)
    ids = [int(x) for x in ids[:max_seq_length]]
    starts = [i for i, token in enumerate(ids) if token == 151644]
    if len(ids) < 16 or len(starts) < 3 or starts[-1] >= len(ids) - 3:
        return None
    return ids, starts[-1]


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_exclusions(paths):
    sample_ids, prompt_hashes, token_hashes = set(), set(), set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_ids.add(row["sample_id"])
                prompt_hashes.add(row["prompt_sha256"])
                token_hashes.add(row["token_ids_sha256"])
    return sample_ids, prompt_hashes, token_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    formal_text, formal_tokens, formal_normalized = formal_contract(tokenizer)
    excluded_ids, excluded_prompt, excluded_tokens = load_exclusions(args.exclude)
    seen_ids, seen_prompt, seen_tokens = set(excluded_ids), set(excluded_prompt), set(excluded_tokens)
    per_stratum = math.ceil(args.tokens / len(SOURCES))
    output, source_manifest = [], []
    overlap = {"formal_text": [], "formal_token": [], "formal_substring": [], "s0": []}

    for source in SOURCES:
        verify_source(source)
        local_seed = int(sha256_bytes(f"train:{args.seed}:{source['dataset']}".encode())[:16], 16)
        rng = random.Random(local_seed)
        selected, selected_tokens, payload_hashes = [], 0, []
        # Deliberately choose a different deterministic region than S0 and reject by hash/ID too.
        offset = rng.randrange(0, source["rows"] - 10_000)
        start = offset
        while selected_tokens < per_stratum:
            wrapped_rows = fetch_rows(source, offset, length=100)
            for wrapped in wrapped_rows:
                row = wrapped["row"]
                payload_hashes.append(sha256_bytes(canonical_json(row)))
                messages = response_messages(source, row)
                if not messages:
                    continue
                prompt = messages[0]["content"]
                norm = normalized_text(prompt)
                prompt_hash = sha256_bytes(norm.encode())
                tokenized = tokenize_example(tokenizer, messages, args.max_seq_length)
                if tokenized is None:
                    continue
                ids, loss_start = tokenized
                token_hash = sha256_bytes(canonical_json(ids))
                row_index = int(wrapped["row_idx"])
                sample_local = (
                    str(row.get(source["id_field"]))
                    if source["id_field"] and row.get(source["id_field"])
                    else f"row-{row_index}"
                )
                sample_id = f"{source['dataset']}@{source['revision']}:{sample_local}"
                if sample_id in seen_ids or prompt_hash in seen_prompt or token_hash in seen_tokens:
                    if sample_id in excluded_ids or prompt_hash in excluded_prompt or token_hash in excluded_tokens:
                        overlap["s0"].append(sample_id)
                    continue
                if prompt_hash in formal_text:
                    overlap["formal_text"].append(sample_id)
                    continue
                if token_hash in formal_tokens:
                    overlap["formal_token"].append(sample_id)
                    continue
                if any(min(len(norm), len(x)) >= 80 and (norm in x or x in norm) for x in formal_normalized):
                    overlap["formal_substring"].append(sample_id)
                    continue
                item = {
                    "sample_id": sample_id,
                    "dataset": source["dataset"],
                    "source_revision": source["revision"],
                    "license": source["license"],
                    "source_split": source["split"],
                    "row_index": row_index,
                    "stratum": source["stratum"],
                    "prompt_sha256": prompt_hash,
                    "token_ids_sha256": token_hash,
                    "input_ids": ids,
                    "token_count": len(ids),
                    "loss_start": loss_start,
                    "split": "train",
                    "seed": args.seed,
                }
                selected.append(item)
                selected_tokens += len(ids)
                seen_ids.add(sample_id); seen_prompt.add(prompt_hash); seen_tokens.add(token_hash)
                if selected_tokens >= per_stratum:
                    break
            offset += 100
            if offset >= source["rows"]:
                offset = 0
            if offset == start or len(payload_hashes) > 20_000:
                raise RuntimeError(f"unable to meet token budget for {source['dataset']}")
        output.extend(selected)
        source_manifest.append({
            "stratum": source["stratum"], "dataset": source["dataset"],
            "revision": source["revision"], "license": source["license"],
            "source_split": source["split"], "start_offset": start,
            "last_offset_exclusive": offset, "samples": len(selected),
            "tokens": selected_tokens,
            "viewer_payload_sha256": sha256_bytes(canonical_json(payload_hashes)),
        })

    # overlap.s0 records rejected rows, not leakage; formal lists must be empty.
    if overlap["formal_text"] or overlap["formal_token"] or overlap["formal_substring"]:
        raise RuntimeError(f"formal overlap: {overlap}")
    rng = random.Random(args.seed)
    rng.shuffle(output)
    path = args.output_dir / "train_1m.jsonl"
    atomic_jsonl(path, output)
    manifest = {
        "schema_version": 1, "stage": "TD32-S3-S5-DATA", "seed": args.seed,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_seq_length": args.max_seq_length,
        "formal_overlap_count": 0,
        "s0_rejected_overlap_count": len(overlap["s0"]),
        "exclusions": [{"path": str(x.resolve()), "sha256": sha256_file(x)} for x in args.exclude],
        "sources": source_manifest,
        "train": {"path": str(path.resolve()), "sha256": sha256_file(path),
                  "samples": len(output), "tokens": sum(x["token_count"] for x in output)},
        "cumulative_token_gates": [128_000, 500_000, 1_000_000],
    }
    atomic_json(args.output_dir / "train_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
