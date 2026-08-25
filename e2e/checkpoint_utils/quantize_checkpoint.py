"""Convert dense Llama 2/3.1, CodeLlama, Qwen2.5, or Qwen3 to QuaRot INT4."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import transformers
from e2e.checkpoint_utils import streaming_gptq, streaming_rtn

QUAROT_CHECKPOINT_FORMAT_VERSION = 2
QUAROT_FFN_FORMAT = "grouped_h256_v1"
QUAROT_ACTIVATION_CLIP_RATIO = 0.9

FAMILIES = {
    "llama": ("e2e.quantized_llama.modeling_llama", "QuarotLlamaConfig",
              "QuarotLlamaForCausalLM", "llama_quarot"),
    "qwen2": ("e2e.quantized_qwen2.modeling_qwen2", "QuarotQwen2Config",
              "QuarotQwen2ForCausalLM", "qwen2_quarot"),
    "qwen3": ("e2e.quantized_qwen3.modeling_qwen3", "QuarotQwen3Config",
              "QuarotQwen3ForCausalLM", "qwen3_quarot"),
}

def runtime_types(model_type):
    if model_type not in FAMILIES:
        raise ValueError(f"unsupported dense model_type {model_type!r}")
    module_name, config_name, model_name, output_type = FAMILIES[model_type]
    module = __import__(module_name, fromlist=[config_name, model_name])
    return getattr(module, config_name), getattr(module, model_name), output_type, module

def main(args):
    # A resumed run must recreate the rotation used by its cached layers.
    torch.manual_seed(args.seed)
    config = transformers.AutoConfig.from_pretrained(args.model)
    if getattr(config, "num_experts", 0):
        raise ValueError("MoE checkpoints are intentionally not supported")
    config_cls, runtime_cls, output_type, runtime_module = runtime_types(config.model_type)
    if args.quant_method == "rtn":
        print(f"Streaming RtN: source={args.model}, output={args.output}, "
              f"rotation_device={args.rotation_device}, "
              f"rotation_dtype={args.rotation_dtype}", flush=True)
        streaming_rtn.convert(args.model, args.output, config, args)
    else:
        print(f"Streaming GPTQ: source={args.model}, output={args.output}, "
              f"rotation_device={args.rotation_device}, "
              f"rotation_dtype={args.rotation_dtype}", flush=True)
        streaming_gptq.convert(args.model, args.output, config, args)

    finalize_output(args.output, args.model, config_cls, runtime_cls,
                    output_type, runtime_module)


def finalize_output(output_path, model, config_cls, runtime_cls, output_type,
                    runtime_module):
    output = Path(output_path)
    runtime_config = config_cls.from_pretrained(
        model, attn_implementation="flash_attention_2")
    runtime_config.save_pretrained(output)
    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["auto_map"] = {
        "AutoConfig": f"quarot.{config_cls.__name__}",
        "AutoModelForCausalLM": f"quarot.{runtime_cls.__name__}"}
    saved_config["model_type"] = output_type
    saved_config["tokenizer_name_or_path"] = model
    saved_config["quarot_checkpoint_format_version"] = (
        QUAROT_CHECKPOINT_FORMAT_VERSION)
    saved_config["quarot_ffn_format"] = QUAROT_FFN_FORMAT
    saved_config["quarot_activation_clip_ratio"] = (
        QUAROT_ACTIVATION_CLIP_RATIO)
    config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
    transformers.AutoTokenizer.from_pretrained(model).save_pretrained(output)
    source = Path(runtime_module.__file__)
    shutil.copy(source, output / "quarot.py")
    shutil.copy(Path(__file__).parents[1] / "quantized_common.py",
                output / "quantized_common.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cal-dataset", default="wikitext2",
                        choices=("wikitext2", "ptb", "c4"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--tokenizer-model",
        help="Optional tokenizer ID/path for calibration when --model uses a "
             "gated or unavailable tokenizer.")
    parser.add_argument(
        "--quant-method", choices=("rtn", "gptq"), default="rtn",
        help="Weight quantization method (default: rtn). Both methods stream "
             "the checkpoint one layer at a time.")
    parser.add_argument("--w-groupsize", type=int, default=-1)
    parser.add_argument("--w-asym", action="store_true")
    parser.add_argument(
        "--w-clip", action=argparse.BooleanOptionalAction, default=True,
        help="Optimize per-channel weight clipping (default: enabled; use "
             "--no-w-clip only for diagnostic unclipped RTN/GPTQ runs).")
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument(
        "--rotation-device", choices=("cuda", "cpu"), default="cuda",
        help="Device for offline LayerNorm fusion and randomized rotations "
             "(default: cuda). Weights are copied back to CPU after each op.")
    parser.add_argument(
        "--rotation-dtype", choices=("float32", "float64"),
        default="float32",
        help="Compute dtype for offline fusion and rotation (default: float32). "
             "Use float64 for legacy conversion numerics.")
    args = parser.parse_args()
    args.w_bits = 4
    main(args)
