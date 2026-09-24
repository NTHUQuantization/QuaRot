"""Create a response-bearing training validation split disjoint from all TD32 data."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from e2e.prepare_qwen3_32b_td_data import (
    SOURCES, canonical_json, fetch_rows, formal_contract, normalized_text,
    sha256_bytes, sha256_file, verify_source, atomic_jsonl, atomic_json,
)
from e2e.prepare_qwen3_32b_td_train_data import response_messages, tokenize_example, load_exclusions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer",required=True); p.add_argument("--exclude",type=Path,action="append",required=True)
    p.add_argument("--train-manifest",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--per-stratum",type=int,default=4); p.add_argument("--max-seq-length",type=int,default=512)
    a=p.parse_args(argv)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    formal_text,formal_tokens,formal_norm=formal_contract(tok)
    ids,prompts,tokens=load_exclusions([*a.exclude, Path(json.loads(a.train_manifest.read_text())["train"]["path"])])
    train_manifest=json.loads(a.train_manifest.read_text()); source_by={x["stratum"]:x for x in train_manifest["sources"]}
    rows=[]; payload=[]
    for source in SOURCES:
        verify_source(source); selected=0; offset=int(source_by[source["stratum"]]["last_offset_exclusive"])+100
        while selected<a.per_stratum:
            for wrapped in fetch_rows(source,offset,100):
                raw=wrapped["row"]; payload.append(sha256_bytes(canonical_json(raw)))
                messages=response_messages(source,raw)
                if not messages: continue
                tokenized=tokenize_example(tok,messages,a.max_seq_length)
                if tokenized is None: continue
                input_ids,loss_start=tokenized; prompt=normalized_text(messages[0]["content"])
                ph=sha256_bytes(prompt.encode()); th=sha256_bytes(canonical_json(input_ids)); idx=int(wrapped["row_idx"])
                local=(str(raw.get(source["id_field"])) if source["id_field"] and raw.get(source["id_field"]) else f"row-{idx}")
                sid=f"{source['dataset']}@{source['revision']}:{local}"
                if sid in ids or ph in prompts or th in tokens or ph in formal_text or th in formal_tokens: continue
                if any(min(len(prompt),len(x))>=80 and (prompt in x or x in prompt) for x in formal_norm): continue
                rows.append({"sample_id":sid,"dataset":source["dataset"],"source_revision":source["revision"],
                    "license":source["license"],"source_split":source["split"],"row_index":idx,
                    "stratum":source["stratum"],"prompt_sha256":ph,"token_ids_sha256":th,
                    "input_ids":input_ids,"token_count":len(input_ids),"loss_start":loss_start,
                    "split":"training_validation","seed":20260831})
                ids.add(sid);prompts.add(ph);tokens.add(th);selected+=1
                if selected>=a.per_stratum: break
            offset+=100
    rows.sort(key=lambda x:(x["stratum"],x["sample_id"])); a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/"training_validation.jsonl"; atomic_jsonl(path,rows)
    manifest={"schema_version":1,"stage":"TD32-S3-S5-TRAIN-VALIDATION","formal_overlap_count":0,
        "train_overlap_count":0,"s0_overlap_count":0,"path":str(path.resolve()),"sha256":sha256_file(path),
        "samples":len(rows),"tokens":sum(x["token_count"] for x in rows),
        "per_stratum":{s:sum(x["stratum"]==s for x in rows) for s in ("general","code","math")},
        "payload_sha256":sha256_bytes(canonical_json(payload))}
    atomic_json(a.output_dir/"training_validation_manifest.json",manifest);print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
