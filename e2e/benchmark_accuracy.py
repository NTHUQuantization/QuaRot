"""Full-corpus perplexity benchmark for dense QuaRot checkpoints."""
import argparse
import gc
import json
import sys
from pathlib import Path
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e2e.checkpoint_utils.data_utils import get_loaders
from e2e.model_registry import runtime_types


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_layers(model):
    return getattr(getattr(model, "model", None), "layers", ())


def set_synthetic_kv4(model, enabled):
    """Enable paper-style synthetic KV4 on every QuaRot attention layer."""
    for layer in model_layers(model):
        layer.self_attn._synthetic_kv4 = enabled


def synthetic_kv4_enabled(model):
    return any(
        getattr(layer.self_attn, "_synthetic_kv4", False)
        for layer in model_layers(model))


def load_int4(path):
    config_cls, int4_cls, _ = runtime_types(path, local_files_only=True)
    config = config_cls.from_pretrained(
        path, attn_implementation="flash_attention_2", local_files_only=True)
    return int4_cls.from_pretrained(
        path, config=config, torch_dtype=torch.float16, local_files_only=True)


def load_reference(path):
    _, _, fp16_cls = runtime_types(path)
    return fp16_cls.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2")


@torch.inference_mode()
def full_perplexity(model, input_ids, context_length, batch_size=1):
    """Evaluate complete blocks with full-sequence causal forwards."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("full perplexity expects a [1, tokens] tensor")
    if context_length < 2 or batch_size < 1:
        raise ValueError("context_length >= 2 and batch_size >= 1 required")
    dataset_tokens = input_ids.numel()
    block_count = dataset_tokens // context_length
    if not block_count:
        raise ValueError(
            f"dataset has {dataset_tokens} tokens, fewer than one "
            f"{context_length}-token block")
    used_tokens = block_count * context_length
    blocks = input_ids[:, :used_tokens].reshape(block_count, context_length)
    sequence_nlls = []
    total_nll, scored_tokens = 0.0, 0
    device = model_input_device(model)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    for start in range(0, block_count, batch_size):
        block = blocks[start:start + batch_size].to(device)
        logits = model(block, use_cache=False).logits[:, :-1].float()
        labels = block[:, 1:]
        losses = loss_function(logits.permute(0, 2, 1), labels)
        sequence_nlls.append(losses.float().mean(dim=1).cpu())
        total_nll += losses.double().sum().item()
        scored_tokens += labels.numel()
        del block, logits, labels, losses
    mean_nll_tensor = torch.cat(sequence_nlls).mean()
    return {
        "perplexity": torch.exp(mean_nll_tensor).item(),
        "mean_nll": mean_nll_tensor.item(), "total_nll": total_nll,
        "scored_tokens": scored_tokens, "block_count": block_count,
        "context_length": context_length, "batch_size": batch_size,
        "dataset_tokens": dataset_tokens, "used_input_tokens": used_tokens,
        "truncated_tail_tokens": dataset_tokens - used_tokens,
        "execution": "full_sequence_no_cache",
        "kv_precision": (
            "synthetic_int4" if synthetic_kv4_enabled(model)
            else "float16"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4-model", required=True,
                        help="converted packed INT4 checkpoint directory")
    parser.add_argument("--reference-model", required=True,
                        help="base Hugging Face model used during conversion")
    parser.add_argument("--dataset", choices=("wikitext2", "ptb", "c4"),
                        default="wikitext2")
    parser.add_argument("--hf-token")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--ppl-batch-size", type=int, default=1)
    parser.add_argument(
        "--kv-cache-dtype", choices=("int4", "float16"), default="int4",
        help=("KV precision for PPL: paper-style synthetic INT4 QDQ or "
              "unquantized FP16 K/V (default: int4)."))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark_results/accuracy/accuracy_results.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("packed INT4 kernels require a CUDA/HIP device")
    if args.context_length < 2 or args.ppl_batch_size < 1:
        parser.error("invalid context length or batch size")

    encoded = get_loaders(
        args.dataset, seed=0, model=args.reference_model,
        seqlen=args.context_length, hf_token=args.hf_token, eval_mode=True)
    input_ids = encoded.input_ids.cpu()
    reference_eval = None
    if not args.skip_reference:
        reference = load_reference(args.reference_model).cuda().eval()
        reference_eval = full_perplexity(
            reference, input_ids, args.context_length, args.ppl_batch_size)
        del reference
        cleanup()
    int4 = load_int4(args.int4_model)
    set_synthetic_kv4(int4, args.kv_cache_dtype == "int4")
    int4 = int4.cuda().eval()
    int4_eval = full_perplexity(
        int4, input_ids, args.context_length, args.ppl_batch_size)

    perplexity = {"int4": int4_eval["perplexity"]}
    full_result = {"int4": int4_eval}
    if reference_eval is not None:
        ratio = int4_eval["perplexity"] / reference_eval["perplexity"]
        perplexity.update(reference=reference_eval["perplexity"], ratio=ratio)
        full_result.update(reference=reference_eval, ratio=ratio)
    result = {
        "configuration": {
            "int4_model": args.int4_model, "reference_model": args.reference_model,
            "dataset": args.dataset, "context_length": args.context_length,
            "ppl_batch_size": args.ppl_batch_size,
            "kv_cache_dtype": args.kv_cache_dtype,
            "skip_reference": args.skip_reference, "output": str(args.output)},
        "model_type": int4.config.model_type,
        "evaluation_protocol": {
            "kind": "full_corpus_non_overlapping_fake_quant",
            "dataset": args.dataset,
            "tail_policy": "truncate_incomplete_block",
            "nll_aggregation": "mean_per_sequence_then_mean_sequences",
            "execution": "full_sequence_no_cache",
            "context_preserved_across_blocks": False,
            "kv_precision": (
                "synthetic_int4" if args.kv_cache_dtype == "int4"
                else "float16"),
            "kv_quantization": (
                "symmetric_per_token_qdq" if args.kv_cache_dtype == "int4"
                else "none"),
            "k_hadamard_after_rope": args.kv_cache_dtype == "int4"},
        "full_perplexity": full_result, "perplexity": perplexity}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
