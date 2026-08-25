"""Accuracy regression benchmark for dense QuaRot Llama/Qwen checkpoints."""
import argparse
import gc
import json
import math
import sys
from pathlib import Path
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers
from e2e.model_registry import runtime_types
from e2e.checkpoint_utils.data_utils import get_loaders

def tokens(batch, length, vocab, device):
    values = torch.arange(batch * length, device=device).reshape(batch, length)
    return values.remainder(max(vocab - 3, 1)).add(3).long()

@torch.inference_mode()
def snapshot(model, input_ids, decode_steps):
    model._expected_max_length = input_ids.shape[1] + decode_steps
    output = model(input_ids, use_cache=True)
    result = [output.logits[:, -1].float().cpu()]
    next_token = torch.full((input_ids.shape[0], 1), 100,
                            dtype=torch.long, device=input_ids.device)
    for _ in range(decode_steps):
        output = model(next_token, past_key_values=output.past_key_values,
                       use_cache=True)
        result.append(output.logits[:, -1].float().cpu())
    return result

def metrics(actual, expected):
    delta = actual - expected
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            actual, expected).mean()),
        "top_token_agreement": float(
            (actual.argmax(-1) == expected.argmax(-1)).float().mean()),
        "finite": bool(torch.isfinite(actual).all()),
    }

@torch.inference_mode()
def perplexity(model, input_ids, chunk):
    total_nll, total_tokens = 0.0, 0
    for start in range(0, input_ids.shape[1] - 1, chunk):
        stop = min(start + chunk, input_ids.shape[1] - 1)
        window = input_ids[:, start:stop + 1]
        logits = model(window, use_cache=False).logits[:, :-1].float()
        labels = window[:, 1:]
        total_nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            reduction="sum").item()
        total_tokens += labels.numel()
    return math.exp(total_nll / total_tokens)

@torch.inference_mode()
def dataset_metrics(actual_model, reference_model, input_ids, chunk, expected_chunks=None):
    totals = {
        "positions": 0, "logits": 0,
        "actual_nll": 0.0, "reference_nll": 0.0,
        "cosine": 0.0, "kl_reference_to_int4": 0.0,
        "mean_abs": 0.0, "top1": 0, "reference_top1_in_actual_top5": 0,
        "top5_overlap": 0.0, "max_abs": 0.0,
    }
    finite = True
    chunks = 0
    for start in range(0, input_ids.shape[1] - 1, chunk):
        stop = min(start + chunk, input_ids.shape[1] - 1)
        window = input_ids[:, start:stop + 1]
        labels = window[:, 1:]
        actual = actual_model(window, use_cache=False).logits[:, :-1].float()
        if expected_chunks is None:
            expected = reference_model(window, use_cache=False).logits[:, :-1].float()
        else:
            expected = expected_chunks[chunks].to(actual.device)
        flat_actual = actual.reshape(-1, actual.shape[-1])
        flat_expected = expected.reshape(-1, expected.shape[-1])
        flat_labels = labels.reshape(-1)
        positions = flat_labels.numel()
        delta = flat_actual - flat_expected

        totals["actual_nll"] += torch.nn.functional.cross_entropy(
            flat_actual, flat_labels, reduction="sum").item()
        totals["reference_nll"] += torch.nn.functional.cross_entropy(
            flat_expected, flat_labels, reduction="sum").item()
        totals["cosine"] += torch.nn.functional.cosine_similarity(
            flat_actual, flat_expected, dim=-1).sum().item()
        actual_logp = torch.nn.functional.log_softmax(flat_actual, dim=-1)
        expected_logp = torch.nn.functional.log_softmax(flat_expected, dim=-1)
        totals["kl_reference_to_int4"] += (
            expected_logp.exp() * (expected_logp - actual_logp)
        ).sum(-1).sum().item()
        totals["mean_abs"] += delta.abs().sum().item()
        totals["max_abs"] = max(totals["max_abs"], float(delta.abs().max()))

        actual_top5 = flat_actual.topk(5, dim=-1).indices
        expected_top5 = flat_expected.topk(5, dim=-1).indices
        totals["top1"] += int(
            (actual_top5[:, 0] == expected_top5[:, 0]).sum())
        totals["reference_top1_in_actual_top5"] += int(
            (actual_top5 == expected_top5[:, :1]).any(-1).sum())
        overlap = (
            actual_top5.unsqueeze(-1) == expected_top5.unsqueeze(-2)
        ).any(-1).sum(-1)
        totals["top5_overlap"] += overlap.float().sum().item()
        totals["positions"] += positions
        totals["logits"] += delta.numel()
        finite = finite and bool(torch.isfinite(actual).all())
        chunks += 1

    positions = totals["positions"]
    return {
        "positions": positions,
        "chunks": chunks,
        "int4_perplexity": math.exp(totals["actual_nll"] / positions),
        "reference_perplexity": math.exp(
            totals["reference_nll"] / positions),
        "mean_cosine": totals["cosine"] / positions,
        "mean_kl_reference_to_int4": (
            totals["kl_reference_to_int4"] / positions),
        "mean_abs_logit_error": totals["mean_abs"] / totals["logits"],
        "max_abs_logit_error": totals["max_abs"],
        "top1_agreement": totals["top1"] / positions,
        "reference_top1_in_int4_top5": (
            totals["reference_top1_in_actual_top5"] / positions),
        "mean_top5_overlap": totals["top5_overlap"] / (positions * 5),
        "finite": finite,
    }

def load_int4(path):
    config_cls, int4_cls, _ = runtime_types(path, local_files_only=True)
    config = config_cls.from_pretrained(
        path, attn_implementation="flash_attention_2", local_files_only=True)
    return int4_cls.from_pretrained(
        path, config=config, torch_dtype=torch.float16, local_files_only=True)

def load_reference(path, *, device_map=None, max_memory=None,
                   offload_folder=None, offload_state_dict=True):
    if device_map is not None:
        # The QuaRot FP16 wrapper always calls FlashAttention, which cannot
        # execute layers assigned to CPU. The standard dense model with SDPA
        # is numerically equivalent and supports mixed GPU/CPU/disk dispatch.
        return transformers.AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map=device_map, max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            low_cpu_mem_usage=True)
    _, _, fp16_cls = runtime_types(path)
    return fp16_cls.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def model_input_device(model):
    return model.get_input_embeddings().weight.device


@torch.inference_mode()
def snapshot_no_cache(model, input_ids, decode_steps):
    sequence = input_ids
    result = []
    for step in range(decode_steps + 1):
        output = model(sequence, use_cache=False)
        result.append(output.logits[:, -1].float().cpu())
        if step != decode_steps:
            next_token = torch.full(
                (sequence.shape[0], 1), 100, dtype=torch.long,
                device=sequence.device)
            sequence = torch.cat((sequence, next_token), dim=1)
        del output
    return result


@torch.inference_mode()
def collect_dataset_logits(model, input_ids, chunk):
    result = []
    device = model_input_device(model)
    for start in range(0, input_ids.shape[1] - 1, chunk):
        stop = min(start + chunk, input_ids.shape[1] - 1)
        window = input_ids[:, start:stop + 1].to(device)
        result.append(model(window, use_cache=False).logits[:, :-1].float().cpu())
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4-model", required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument(
        "--tokenizer-model",
        help="Optional tokenizer ID/path when the reference model tokenizer "
             "is gated or unavailable (defaults to --reference-model).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--dataset", choices=("wikitext2", "ptb", "c4"), default="wikitext2")
    parser.add_argument("--hf-token")
    parser.add_argument("--ppl-tokens", type=int, default=512)
    parser.add_argument("--ppl-chunk", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path("accuracy_results.json"))
    parser.add_argument(
        "--sequential-low-vram", action="store_true",
        help="Run an offloaded FP16 reference pass first, unload it, then run "
             "INT4 so both models are never resident on the GPU together.")
    parser.add_argument("--reference-max-gpu-memory", default="16GiB")
    parser.add_argument("--reference-max-cpu-memory", default="40GiB")
    parser.add_argument(
        "--offload-folder", type=Path,
        default=Path("/tmp/quarot-reference-offload"))
    parser.add_argument(
        "--reference-disk-offload", action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow reference weights/state to spill to --offload-folder "
             "(default: enabled). Disable only when GPU+CPU budgets cover "
             "the complete reference checkpoint.")
    args = parser.parse_args()
    if args.ppl_tokens < 2 or args.ppl_chunk < 1:
        parser.error("--ppl-tokens must be at least 2 and --ppl-chunk positive")
    device = torch.device("cuda")
    evaluation = get_loaders(
        args.dataset, seed=0,
        model=args.tokenizer_model or args.reference_model,
        seqlen=args.ppl_chunk, hf_token=args.hf_token, eval_mode=True)
    ppl_ids_cpu = evaluation.input_ids[:, :args.ppl_tokens].cpu()

    if args.sequential_low_vram:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        reference = load_reference(
            args.reference_model, device_map="auto",
            max_memory={
                0: args.reference_max_gpu_memory,
                "cpu": args.reference_max_cpu_memory,
            },
            offload_folder=(str(args.offload_folder)
                            if args.reference_disk_offload else None),
            offload_state_dict=args.reference_disk_offload).eval()
        reference_device = model_input_device(reference)
        reference_ids = tokens(
            args.batch_size, args.prefill,
            reference.config.vocab_size, reference_device)
        expected = snapshot_no_cache(
            reference, reference_ids, args.decode_steps)
        expected_chunks = collect_dataset_logits(
            reference, ppl_ids_cpu, args.ppl_chunk)
        del reference, reference_ids
        cleanup()

        int4 = load_int4(args.int4_model).cuda().eval()
        input_ids = tokens(
            args.batch_size, args.prefill, int4.config.vocab_size, device)
        actual = snapshot_no_cache(int4, input_ids, args.decode_steps)
        dataset_evaluation = dataset_metrics(
            int4, None, ppl_ids_cpu.to(device), args.ppl_chunk,
            expected_chunks=expected_chunks)
    else:
        int4 = load_int4(args.int4_model).cuda().eval()
        reference = load_reference(args.reference_model).cuda().eval()
        input_ids = tokens(args.batch_size, args.prefill,
                           int4.config.vocab_size, device)
        actual = snapshot(int4, input_ids, args.decode_steps)
        expected = snapshot(reference, input_ids, args.decode_steps)
        dataset_evaluation = dataset_metrics(
            int4, reference, ppl_ids_cpu.to(device), args.ppl_chunk)
    result = {
        "configuration": vars(args) | {
            "output": str(args.output),
            "offload_folder": str(args.offload_folder),
        },
        "model_type": int4.config.model_type,
        "logit_comparison": [
            {"phase": "prefill" if i == 0 else f"decode_{i}",
             **metrics(a, e)}
            for i, (a, e) in enumerate(zip(actual, expected))],
        "dataset_evaluation": dataset_evaluation,
        "perplexity": {
            "int4": dataset_evaluation["int4_perplexity"],
            "reference": dataset_evaluation["reference_perplexity"],
        },
    }
    result["perplexity"]["ratio"] = (
        result["perplexity"]["int4"] / result["perplexity"]["reference"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
