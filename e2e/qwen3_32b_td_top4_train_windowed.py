"""R9700-safe PARD-2 trainer for projection plus the top draft layers."""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from e2e.qwen3_32b_td_features import (
    atomic_json,
    gpu_snapshot,
    preflight_gpu,
    sha256_file,
)
from e2e.qwen3_32b_td_projection_train import (
    SCALE,
    build_expanded,
    cosine_lr,
    exact_chunked_loss_and_hidden_grad,
    load_cache,
    load_draft,
    load_initial_projection,
    load_target_head,
)
from e2e.qwen3_32b_td_projection_train_windowed import windows


def trainable_top_layers(draft, count):
    layers = draft.model.layers
    if not 0 < count <= len(layers):
        raise ValueError(f"invalid top-layer count {count} for {len(layers)} layers")
    selected = []
    start = len(layers) - count
    for layer_index in range(start, len(layers)):
        for name, parameter in layers[layer_index].named_parameters():
            parameter.requires_grad_(True)
            selected.append((f"model.layers.{layer_index}.{name}", parameter))
    return selected


def package_deploy(output_dir, draft, draft_snapshot, projection):
    revision = Path(draft_snapshot).name
    deploy = output_dir / "deploy" / "snapshots" / revision
    deploy.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in draft.state_dict().items()}
    draft.save_pretrained(
        deploy,
        state_dict=state,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    readme = Path(draft_snapshot) / "README.md"
    if readme.exists():
        shutil.copy2(readme, deploy / "README.md")
    warp = deploy / "warp_model.bin"
    torch.save({"target_proj.weight": projection.detach().cpu().to(torch.bfloat16)}, warp)
    return {
        "deploy": str(deploy.resolve()),
        "warp_sha256": sha256_file(warp),
        "model_sha256": sha256_file(deploy / "model.safetensors"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--total-units", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--vocab-chunk", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    preflight = preflight_gpu()
    draft = load_draft(args.draft)
    target_head = load_target_head(args.target_source)
    initial, raw_scale, raw_bias = load_initial_projection(args.draft, args.calibration)
    projection = torch.nn.Parameter(initial.to("cuda", dtype=torch.float32))
    named_model_parameters = trainable_top_layers(draft, args.top_layers)
    master_parameters = [
        torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
        for _, parameter in named_model_parameters
    ]
    optimizer = torch.optim.AdamW(
        [projection, *master_parameters],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    raw_scale = raw_scale.to("cuda", dtype=torch.float32)
    raw_bias = raw_bias.to("cuda", dtype=torch.float32)
    state = {"global_unit": 0, "optimizer_step": 0, "source_tokens_seen": 0,
             "trained_tokens_seen": 0}
    snapshots = [gpu_snapshot("runtime_loaded")]

    def clear_model_gradients():
        for _, parameter in named_model_parameters:
            parameter.grad = None

    def copy_model_gradients_to_master(divisor):
        for (_, parameter), master in zip(named_model_parameters, master_parameters):
            if parameter.grad is None:
                master.grad = None
            else:
                gradient = parameter.grad.detach().float().div(divisor)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("non-finite top-layer gradient")
                master.grad = gradient

    @torch.no_grad()
    def copy_master_to_model():
        for (_, parameter), master in zip(named_model_parameters, master_parameters):
            parameter.copy_(master.to(parameter.dtype))

    def optimizer_step(pending):
        if projection.grad is None:
            raise RuntimeError("projection gradient is absent")
        projection.grad.div_(pending)
        if not torch.isfinite(projection.grad).all():
            raise FloatingPointError("non-finite projection gradient")
        copy_model_gradients_to_master(pending)
        learning_rate = cosine_lr(
            state["global_unit"], args.total_units, base_lr=args.learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        torch.nn.utils.clip_grad_norm_([projection, *master_parameters], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        copy_master_to_model()
        clear_model_gradients()
        state["optimizer_step"] += 1

    def one(ids, feature, probability, loss_start, seed, training):
        calibrated = feature.to("cuda", dtype=torch.float32) * raw_scale + raw_bias
        gold = probability.to("cuda", dtype=torch.float32)
        expanded = build_expanded(ids, calibrated, gold, loss_start, seed=seed, cod=True)
        keep = random.Random(seed + 991).random() > 0.1 if training else True
        word = draft.get_input_embeddings()(expanded["input_ids"]).detach()
        projected = F.linear(
            expanded["target_feat"].to(torch.bfloat16), projection.to(torch.bfloat16)
        ) * SCALE
        if not training:
            projected = projected.detach()
        if not keep:
            projected = projected * 0
        with torch.set_grad_enabled(training):
            hidden = draft.model(
                input_ids=None,
                inputs_embeds=(word + projected)[None],
                attention_mask=expanded["attention_mask"],
                position_ids=expanded["position_ids"][None],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        shifted = hidden[0, :-1]
        labels = expanded["labels"][1:]
        weights = expanded["prev_prob"][1:]
        teacher = feature.to("cuda", dtype=torch.bfloat16)[
            expanded["teacher_index"][:-1], :5120
        ]
        loss, ce, kd, gradient = exact_chunked_loss_and_hidden_grad(
            shifted.detach(), teacher, labels, weights,
            draft.lm_head.weight.detach(), target_head,
            chunk_size=args.vocab_chunk,
        )
        hidden_finite = bool(torch.isfinite(shifted).all())
        loss_finite = bool(torch.isfinite(torch.stack((loss, ce, kd))).all())
        gradient_finite = bool(torch.isfinite(gradient).all())
        if not hidden_finite or not loss_finite or not gradient_finite:
            finite_hidden = shifted.detach().float()[torch.isfinite(shifted)]
            finite_gradient = gradient.detach().float()[torch.isfinite(gradient)]
            hidden_max = float(finite_hidden.abs().max()) if finite_hidden.numel() else None
            gradient_max = float(finite_gradient.abs().max()) if finite_gradient.numel() else None
            raise FloatingPointError(
                "non-finite top4 step "
                f"seed={seed} hidden_finite={hidden_finite} "
                f"loss_finite={loss_finite} gradient_finite={gradient_finite} "
                f"hidden_abs_max={hidden_max} gradient_abs_max={gradient_max}"
            )
        if training and keep:
            shifted.backward(gradient.to(shifted.dtype))
        return float(loss), float(ce), float(kd), int(keep), len(ids), int(shifted.shape[0])

    train_meta = json.loads(args.train_manifest.read_text())
    data_path = Path(train_meta["data"])
    wanted = {row["ordinal"] for payload in load_cache(args.train_manifest)
              for row in payload["rows"]}
    ids_by_ordinal = {}
    with data_path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if ordinal in wanted:
                ids_by_ordinal[ordinal] = json.loads(line)["input_ids"]

    def evaluate():
        draft.eval()
        values = []
        with torch.no_grad():
            for payload in load_cache(args.validation_manifest):
                offset = 0
                for row in payload["rows"]:
                    count = int(row["tokens"])
                    ids = row["input_ids"]
                    feature = payload["features"][0, offset:offset + count]
                    probability = payload["teacher_gold_prob"][0, offset:offset + count]
                    for window_index, item in enumerate(windows(
                        ids, feature, probability, int(row["loss_start"]), args.max_context
                    )):
                        x, f, g, local_loss_start, _ = item
                        values.append(one(
                            x, f, g, local_loss_start,
                            args.seed + row["ordinal"] * 100 + window_index,
                            False,
                        )[:3])
                    offset += count
        draft.train()
        return {
            "loss": sum(x[0] for x in values) / len(values),
            "ce": sum(x[1] for x in values) / len(values),
            "kd": sum(x[2] for x in values) / len(values),
            "units": len(values),
        }

    before = evaluate()
    started = time.perf_counter()
    history = []
    pending = 0
    stage_source = 0
    stage_trained = 0
    for payload in load_cache(args.train_manifest):
        offset = 0
        for row in payload["rows"]:
            count = int(row["tokens"])
            ids = ids_by_ordinal[row["ordinal"]]
            feature = payload["features"][0, offset:offset + count]
            probability = payload["teacher_gold_prob"][0, offset:offset + count]
            stage_source += count
            for window_index, item in enumerate(windows(
                ids, feature, probability, int(row["loss_start"]), args.max_context
            )):
                x, f, g, local_loss_start, _ = item
                metrics = one(
                    x, f, g, local_loss_start,
                    args.seed + row["ordinal"] * 100 + window_index,
                    True,
                )
                pending += 1
                state["global_unit"] += 1
                state["trained_tokens_seen"] += metrics[4]
                stage_trained += metrics[4]
                history.append({
                    "unit": state["global_unit"], "loss": metrics[0],
                    "ce": metrics[1], "kd": metrics[2], "keep": metrics[3],
                    "tokens": metrics[4], "expanded": metrics[5],
                })
                if pending >= args.gradient_accumulation:
                    optimizer_step(pending)
                    pending = 0
                if state["global_unit"] % 16 == 0:
                    snapshots.append(gpu_snapshot(f"unit_{state['global_unit']}"))
            offset += count
        gc.collect()
        torch.cuda.empty_cache()
    state["source_tokens_seen"] += stage_source
    if pending:
        optimizer_step(pending)
    elapsed = time.perf_counter() - started
    after = evaluate()
    snapshots.append(gpu_snapshot("complete"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_state = {
        name: parameter.detach().cpu()
        for name, parameter in named_model_parameters
    }
    checkpoint = args.output_dir / "top4_optimizer.pt"
    with tempfile.NamedTemporaryFile(dir=args.output_dir, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save({
        "schema_version": 1,
        "projection": projection.detach().cpu(),
        "top_layers": top_state,
        "master_parameters": [parameter.detach().cpu() for parameter in master_parameters],
        "optimizer": optimizer.state_dict(),
        "state": state,
    }, temporary)
    temporary.replace(checkpoint)
    artifacts = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        **package_deploy(args.output_dir, draft, args.draft, projection),
    }
    result = {
        "schema_version": 1,
        "stage": args.stage,
        "method": "windowed-128-exact-top-layers-fp32-master",
        "top_layers": args.top_layers,
        "top_layer_start": len(draft.model.layers) - args.top_layers,
        "trainable_projection_parameters": projection.numel(),
        "trainable_draft_parameters": sum(p.numel() for _, p in named_model_parameters),
        "optimizer_precision": "fp32_master",
        "learning_rate": args.learning_rate,
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "validation_before": before,
        "validation_after": after,
        "validation_improved": after["loss"] < before["loss"],
        "state": state,
        "stage_source_tokens": stage_source,
        "stage_trained_tokens": stage_trained,
        "elapsed_seconds": elapsed,
        "trained_tokens_per_second": stage_trained / elapsed,
        "history_tail": history[-32:],
        "artifacts": artifacts,
        "gpu_preflight": preflight,
        "memory_snapshots": snapshots,
    }
    atomic_json(args.output_dir / "train_result.json", result)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"history_tail", "memory_snapshots"}
    }, indent=2))


if __name__ == "__main__":
    main()
