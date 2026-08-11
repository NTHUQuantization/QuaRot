"""Dense model-family registry shared by e2e conversion and benchmarks."""
import json
from pathlib import Path
import transformers
from e2e.quantized_llama import modeling_llama
from e2e.quantized_qwen2 import modeling_qwen2
from e2e.quantized_qwen3 import modeling_qwen3

RUNTIMES = {
    "llama": (modeling_llama.QuarotLlamaConfig,
              modeling_llama.QuarotLlamaForCausalLM,
              modeling_llama.QuarotFP16LlamaForCausalLM),
    "llama_quarot": (modeling_llama.QuarotLlamaConfig,
                     modeling_llama.QuarotLlamaForCausalLM,
                     modeling_llama.QuarotFP16LlamaForCausalLM),
    "qwen2": (modeling_qwen2.QuarotQwen2Config,
              modeling_qwen2.QuarotQwen2ForCausalLM,
              modeling_qwen2.QuarotFP16Qwen2ForCausalLM),
    "qwen2_quarot": (modeling_qwen2.QuarotQwen2Config,
                     modeling_qwen2.QuarotQwen2ForCausalLM,
                     modeling_qwen2.QuarotFP16Qwen2ForCausalLM),
    "qwen3": (modeling_qwen3.QuarotQwen3Config,
              modeling_qwen3.QuarotQwen3ForCausalLM,
              modeling_qwen3.QuarotFP16Qwen3ForCausalLM),
    "qwen3_quarot": (modeling_qwen3.QuarotQwen3Config,
                     modeling_qwen3.QuarotQwen3ForCausalLM,
                     modeling_qwen3.QuarotFP16Qwen3ForCausalLM),
}

def runtime_types(path, local_files_only=False):
    config_path = Path(path) / "config.json"
    if config_path.is_file():
        model_type = json.loads(config_path.read_text())["model_type"]
    else:
        config = transformers.AutoConfig.from_pretrained(
            path, local_files_only=local_files_only, trust_remote_code=False)
        model_type = config.model_type
    try:
        return RUNTIMES[model_type]
    except KeyError as error:
        raise ValueError(f"unsupported dense model_type {model_type!r}") from error
