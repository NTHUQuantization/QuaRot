"""Convert a dense Llama, Qwen2.5, or Qwen3 checkpoint to QuaRot INT4."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import transformers
from e2e.checkpoint_utils import data_utils, gptq_utils, rotation_utils
from quarot.functional import pack_i4

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
    config = transformers.AutoConfig.from_pretrained(args.model)
    if getattr(config, "num_experts", 0):
        raise ValueError("MoE checkpoints are intentionally not supported")
    config_cls, runtime_cls, output_type, runtime_module = runtime_types(config.model_type)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16)
    model.seqlen = args.seqlen
    device = torch.device("cuda:0")
    rotation_device = device if args.rotation_device == "cuda" else torch.device("cpu")
    rotation_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.rotation_dtype]
    print(f"Offline fusion/rotation: device={rotation_device}, "
          f"dtype={rotation_dtype}", flush=True)
    rotation_utils.fuse_layer_norms(
        model, device=rotation_device, dtype=rotation_dtype)
    rotation_utils.rotate_model(
        model, device=rotation_device, dtype=rotation_dtype)
    if args.w_rtn:
        quantizers = gptq_utils.rtn_fwrd(model, device, args)
    else:
        loader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples, seed=args.seed,
            model=args.tokenizer_model or args.model,
            seqlen=model.seqlen, eval_mode=False)
        quantizers = gptq_utils.gptq_fwrd(model, loader, device, args)

    key_maps = {"mlp.down_proj": "mlp.down_proj.2",
                "self_attn.o_proj": "self_attn.o_proj.1"}
    def remap(key):
        for old, new in key_maps.items():
            key = key.replace(old, new)
        return key
    removed_norms = ("post_attention_layernorm.weight",
                     "input_layernorm.weight", "model.norm.weight")
    state = {remap(k): v for k, v in model.state_dict().items()
             if not any(name in k for name in removed_norms)}
    for key, quantizer in quantizers.items():
        key = remap(key)
        scale = quantizer.scale
        state[f"{key}.weight_scales"] = scale
        state[f"{key}.weight"] = pack_i4(
            (state[f"{key}.weight"] / scale).round().to(torch.int8))

    runtime_config = config_cls.from_pretrained(
        args.model, attn_implementation="flash_attention_2")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights():
        converted = runtime_cls(runtime_config)
    torch.set_default_dtype(old_dtype)
    result = converted.load_state_dict(state, strict=False)
    unexpected_missing = [k for k in result.missing_keys if "had_rem_dim" not in k]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mapping failed: {result}")
    converted.cpu().save_pretrained(args.output)

    output = Path(args.output)
    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["auto_map"] = {
        "AutoConfig": f"quarot.{config_cls.__name__}",
        "AutoModelForCausalLM": f"quarot.{runtime_cls.__name__}"}
    saved_config["model_type"] = output_type
    config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
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
    parser.add_argument("--w-rtn", action="store_true")
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
