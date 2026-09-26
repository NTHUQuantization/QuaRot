"""Offline, projection-only PARD-2 trainer with exact chunked CE+KD."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import torch
import torch.nn.functional as F

from e2e.qwen3_32b_td_features import atomic_json, gpu_snapshot, preflight_gpu, sha256_file


IGNORE = -100
PARA_NUM = 16
UNUSED_TOKEN = 151670
SCALE = 0.02


def load_target_head(source):
    from safetensors import safe_open
    index = json.loads((Path(source) / "model.safetensors.index.json").read_text())
    filename = index["weight_map"]["lm_head.weight"]
    with safe_open(Path(source) / filename, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("lm_head.weight")
    if tuple(weight.shape) != (151936, 5120):
        raise RuntimeError(f"unexpected target lm_head shape {weight.shape}")
    return weight.to("cuda", dtype=torch.bfloat16)


def load_draft(snapshot):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_initial_projection(draft, calibration):
    state = torch.load(Path(draft) / "warp_model.bin", map_location="cpu", weights_only=True)
    weight = state["target_proj.weight"]
    if tuple(weight.shape) != (1024, 20480):
        raise RuntimeError(f"unexpected projection shape {weight.shape}")
    cal = torch.load(calibration, map_location="cpu", weights_only=True)
    return weight, cal["raw_scale"], cal["raw_bias"]


def build_expanded(ids, feature, gold_prob, loss_start, *, seed, cod=True):
    """Reproduce the official PARD-2 Qwen3 collator for one unpadded row."""
    device = feature.device
    length = len(ids)
    input_ids = torch.tensor(ids, device=device, dtype=torch.long)
    labels = input_ids.clone(); labels[:loss_start] = IGNORE
    base_shifted = torch.cat((torch.zeros_like(feature[:1]), feature[:-1]), dim=0)
    blocks_ids, blocks_labels, blocks_feat, blocks_teacher, blocks_prob = [], [], [], [], []
    ones = torch.ones_like(gold_prob, dtype=torch.float32)
    for i in range(PARA_NUM):
        blocks_ids.append(input_ids if i == 0 else torch.full_like(input_ids, UNUSED_TOKEN))
        blocks_labels.append(torch.cat((labels[:i] * 0 + IGNORE, labels[i:])))
        blocks_feat.append(torch.cat((base_shifted[:i] * 0, base_shifted[: length - i]), dim=0))
        blocks_teacher.append(torch.arange(length, device=device))
        weight = ones.clone()
        for shift in range(1, i + 1):
            weight *= torch.cat((ones[:shift], gold_prob[:-shift]))
        blocks_prob.append(weight)
    new_ids = torch.cat(blocks_ids)
    new_labels = torch.cat(blocks_labels)
    new_feat = torch.cat(blocks_feat)
    teacher_index = torch.cat(blocks_teacher)
    prev_prob = torch.cat(blocks_prob).masked_fill(new_labels == IGNORE, 0.0)

    total = length * PARA_NUM
    cond = torch.arange(total, device=device)
    mask_bool = cond == cond.view(-1, 1)
    for i in range(PARA_NUM):
        mask_bool |= cond == (cond - length * i - i).view(-1, 1)
        mask_bool |= ((cond < (cond - i * length - (i - 1)).view(-1, 1))
                      & (cond < (i + 1) * length).view(-1, 1))
    attention = torch.full((total, total), torch.finfo(torch.float32).min, device=device)
    attention.masked_fill_(mask_bool, 0)
    position = torch.arange(length, device=device).repeat(PARA_NUM)

    if cod:
        generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
        index_mask = torch.zeros(PARA_NUM, length, dtype=torch.bool)
        index_mask[0] = True
        previous = torch.arange(length)
        for i in range(1, PARA_NUM):
            count = int(length * max(0.7 ** i, 0.1))
            count = min(count, len(previous))
            chosen = previous[torch.randperm(len(previous), generator=generator)[:count]]
            index_mask[i, chosen] = True
            previous = (chosen + 1) % length
        indices = index_mask.reshape(-1).nonzero(as_tuple=True)[0].to(device)
        # Official collator rolls labels/probabilities before gathering.
        rolled_labels = torch.roll(torch.roll(new_labels, -1, 0)[indices], 1, 0)
        rolled_prob = torch.roll(torch.roll(prev_prob, -1, 0)[indices], 1, 0)
        attention = attention[indices][:, indices]
        return {
            "input_ids": new_ids[indices], "labels": rolled_labels,
            "target_feat": new_feat[indices], "teacher_index": teacher_index[indices],
            "prev_prob": rolled_prob, "position_ids": position[indices],
            "attention_mask": attention[None, None],
        }
    return {"input_ids": new_ids, "labels": new_labels, "target_feat": new_feat,
            "teacher_index": teacher_index, "prev_prob": prev_prob,
            "position_ids": position, "attention_mask": attention[None, None]}


@torch.no_grad()
def exact_chunked_loss_and_hidden_grad(student_hidden, teacher_hidden, labels, weights,
                                       student_head, teacher_head, *, ce_alpha=0.1,
                                       kd_alpha=1.0, chunk_size=4096):
    """Return exact T=1 CE+KD and analytic d(loss)/d(student_hidden)."""
    valid = labels != IGNORE
    token_weight = weights.float() * valid.float()
    denom = token_weight.sum().clamp_min(1e-6)
    sh = student_hidden; th = teacher_hidden
    logz_s = torch.full((sh.shape[0],), -torch.inf, device=sh.device)
    logz_t = torch.full((th.shape[0],), -torch.inf, device=th.device)
    for start in range(0, student_head.shape[0], chunk_size):
        end = min(start + chunk_size, student_head.shape[0])
        sl = (sh @ student_head[start:end].to(sh.dtype).T).float()
        tl = (th @ teacher_head[start:end].to(th.dtype).T).float()
        logz_s = torch.logaddexp(logz_s, torch.logsumexp(sl, -1))
        logz_t = torch.logaddexp(logz_t, torch.logsumexp(tl, -1))
    ce_sum = torch.zeros((), device=sh.device); kd_sum = torch.zeros((), device=sh.device)
    grad_hidden = torch.zeros_like(sh)
    for start in range(0, student_head.shape[0], chunk_size):
        end = min(start + chunk_size, student_head.shape[0])
        sw = student_head[start:end].to(sh.dtype); tw = teacher_head[start:end].to(th.dtype)
        sl = (sh @ sw.T).float(); tl = (th @ tw.T).float()
        ps = torch.exp(sl - logz_s[:, None]); pt = torch.exp(tl - logz_t[:, None])
        kd_token = (pt * ((tl - logz_t[:, None]) - (sl - logz_s[:, None]))).sum(-1)
        kd_sum += (kd_token * token_weight).sum()
        local = valid & (labels >= start) & (labels < end)
        if local.any():
            rows = local.nonzero(as_tuple=True)[0]; cols = labels[rows] - start
            ce_sum += ((logz_s[rows] - sl[rows, cols]) * token_weight[rows]).sum()
        grad_logits = (ce_alpha + kd_alpha) * ps - kd_alpha * pt
        if local.any():
            grad_logits[rows, cols] -= ce_alpha
        grad_logits *= (token_weight / denom)[:, None]
        grad_hidden += (grad_logits.to(sw.dtype) @ sw).float()
    ce = ce_sum / denom; kd = kd_sum / denom
    return ce_alpha * ce + kd_alpha * kd, ce, kd, grad_hidden


def cosine_lr(step, total_steps, base_lr=3e-5, warmup_ratio=0.03, min_ratio=0.1):
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def load_cache(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"teacher shard SHA mismatch: {path}")
        yield torch.load(path, map_location="cpu", weights_only=True)


def package_deploy(output_dir, draft_snapshot, projection):
    deploy = output_dir / "deploy" / "snapshots" / Path(draft_snapshot).name; deploy.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "README.md"):
        source = Path(draft_snapshot) / name
        if not source.exists():
            continue
        destination = deploy / name
        destination.unlink(missing_ok=True)
        try: os.link(source.resolve(), destination)
        except OSError: os.symlink(source.resolve(), destination)
    warp = deploy / "warp_model.bin"
    torch.save({"target_proj.weight": projection.detach().cpu().to(torch.bfloat16)}, warp)
    return deploy, sha256_file(warp)


def save_checkpoint(output_dir, projection, optimizer, state, draft_snapshot):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "projection_optimizer.pt"
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as f: tmp = Path(f.name)
    torch.save({"projection": projection.detach().cpu(), "optimizer": optimizer.state_dict(),
                "state": state}, tmp); tmp.replace(checkpoint)
    deploy, warp_sha = package_deploy(output_dir, draft_snapshot, projection)
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "deploy": str(deploy.resolve()), "warp_sha256": warp_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-samples", type=int, default=2279)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--vocab-chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    preflight = preflight_gpu()
    draft = load_draft(args.draft)
    target_head = load_target_head(args.target_source)
    initial, raw_scale, raw_bias = load_initial_projection(args.draft, args.calibration)
    projection = torch.nn.Parameter(initial.to("cuda", dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([projection], lr=3e-5, betas=(0.9, 0.95), weight_decay=0.0)
    state = {"global_sample": 0, "optimizer_step": 0, "tokens_seen": 0}
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        projection.data.copy_(resume["projection"].to("cuda", dtype=torch.bfloat16))
        optimizer.load_state_dict(resume["optimizer"]); state.update(resume["state"])
    raw_scale = raw_scale.to("cuda", dtype=torch.float32)
    raw_bias = raw_bias.to("cuda", dtype=torch.float32)
    snapshots = [gpu_snapshot("runtime_loaded")]

    def one_row(ids, feature, prob, loss_start, row_seed, training):
        calibrated = feature.to("cuda", dtype=torch.float32) * raw_scale + raw_bias
        gold = prob.to("cuda", dtype=torch.float32)
        expanded = build_expanded(ids, calibrated, gold, loss_start, seed=row_seed, cod=True)
        keep = (random.Random(row_seed + 991).random() > 0.1) if training else True
        word = draft.get_input_embeddings()(expanded["input_ids"]).detach()
        projected = F.linear(expanded["target_feat"].to(torch.bfloat16), projection) * SCALE
        if not training: projected = projected.detach()
        if not keep: projected = projected * 0
        hidden = draft.model(
            input_ids=None, inputs_embeds=(word + projected)[None],
            attention_mask=expanded["attention_mask"],
            position_ids=expanded["position_ids"][None], use_cache=False,
            return_dict=True,
        ).last_hidden_state
        shifted_hidden = hidden[0, :-1]
        labels = expanded["labels"][1:]
        weights = expanded["prev_prob"][1:]
        teacher = feature.to("cuda", dtype=torch.bfloat16)[expanded["teacher_index"][:-1], :5120]
        loss, ce, kd, grad = exact_chunked_loss_and_hidden_grad(
            shifted_hidden.detach(), teacher, labels, weights,
            draft.lm_head.weight.detach(), target_head,
            chunk_size=args.vocab_chunk,
        )
        if training and keep:
            shifted_hidden.backward(grad.to(shifted_hidden.dtype))
        return float(loss), float(ce), float(kd), int(keep), int(len(ids)), int(shifted_hidden.shape[0])

    @torch.no_grad()
    def validate():
        draft.eval(); values=[]
        for payload in load_cache(args.validation_manifest):
            offset=0
            for row in payload["rows"]:
                n=int(row["tokens"]); ids_by_ordinal[row["ordinal"]] if False else None
                # Validation cache rows include input_ids explicitly.
                ids=row["input_ids"]
                result=one_row(ids, payload["features"][0,offset:offset+n],
                                   payload["teacher_gold_prob"][0,offset:offset+n],
                                   int(row["loss_start"]), args.seed+row["ordinal"], False)
                values.append(result[:3]); offset += n
        draft.train()
        return {"loss": sum(x[0] for x in values)/len(values),
                "ce": sum(x[1] for x in values)/len(values),
                "kd": sum(x[2] for x in values)/len(values), "samples": len(values)}

    # Training cache rows only store identity; recover IDs from the frozen train JSONL.
    data_path = Path(json.loads(args.train_manifest.read_text())["data"])
    ids_by_ordinal = {}
    wanted = {row["ordinal"] for payload in load_cache(args.train_manifest) for row in payload["rows"]}
    with data_path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if ordinal in wanted:
                row=json.loads(line); ids_by_ordinal[ordinal]=row["input_ids"]

    validation_before = validate()
    started=time.perf_counter(); history=[]; pending=0
    for payload in load_cache(args.train_manifest):
        offset=0
        for row in payload["rows"]:
            n=int(row["tokens"]); ids=ids_by_ordinal[row["ordinal"]]
            result=one_row(ids, payload["features"][0,offset:offset+n],
                           payload["teacher_gold_prob"][0,offset:offset+n],
                           int(row["loss_start"]), args.seed+row["ordinal"], True)
            pending += result[3]; state["global_sample"] += 1; state["tokens_seen"] += n
            history.append({"sample": state["global_sample"], "loss": result[0],
                            "ce": result[1], "kd": result[2], "keep": result[3],
                            "tokens": n, "expanded_tokens": result[5]})
            offset += n
            if pending >= args.gradient_accumulation:
                for parameter in [projection]:
                    if parameter.grad is not None: parameter.grad.div_(pending)
                lr=cosine_lr(state["global_sample"], args.total_samples)
                for group in optimizer.param_groups: group["lr"]=lr
                torch.nn.utils.clip_grad_norm_([projection], 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                state["optimizer_step"] += 1; pending=0
            if state["global_sample"] % 16 == 0:
                snapshots.append(gpu_snapshot(f"sample_{state['global_sample']}"))
        gc.collect(); torch.cuda.empty_cache()
    if pending:
        projection.grad.div_(pending); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state["optimizer_step"] += 1
    elapsed=time.perf_counter()-started
    validation_after=validate(); snapshots.append(gpu_snapshot("complete"))
    artifacts=save_checkpoint(args.output_dir, projection, optimizer, state, args.draft)
    result={"schema_version":1,"stage":args.stage,"trainable_parameters":projection.numel(),
            "fixed":{"para_num":16,"scale":0.02,"ce_alpha":0.1,"kd_alpha":1.0,
                     "kd_temperature":1.0,"target_feat_mask":0.1,"cod":[0.7,0.1],
                     "vocab_chunk":args.vocab_chunk},
            "train_manifest":str(args.train_manifest.resolve()),
            "train_manifest_sha256":sha256_file(args.train_manifest),
            "validation_manifest":str(args.validation_manifest.resolve()),
            "validation_before":validation_before,"validation_after":validation_after,
            "validation_improved":validation_after["loss"] < validation_before["loss"],
            "state":state,"elapsed_seconds":elapsed,"tokens_per_second":state["tokens_seen"]/elapsed,
            "history_tail":history[-32:],"artifacts":artifacts,"gpu_preflight":preflight,
            "memory_snapshots":snapshots}
    atomic_json(args.output_dir/"train_result.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"history_tail","memory_snapshots"}},indent=2))


if __name__ == "__main__": main()
