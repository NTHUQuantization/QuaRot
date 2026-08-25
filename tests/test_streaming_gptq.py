from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from e2e.checkpoint_utils import streaming_gptq
from e2e.checkpoint_utils import rotation_utils
from quarot.functional.hadamard import matmul_grouped_h256


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(256, 256, bias=False)
        with torch.no_grad():
            self.self_attn.o_proj.weight.copy_(torch.eye(4))
            self.mlp.down_proj.weight.copy_(torch.eye(256))


def test_calibration_layer_installs_runtime_hadamards_without_changing_keys(
        monkeypatch):
    layer = _ToyLayer()
    original_keys = tuple(layer.state_dict())
    config = SimpleNamespace(
        num_attention_heads=2, hidden_size=4, head_dim=2,
        intermediate_size=256)

    # A last-dimension reversal makes the attention head-axis transform easy
    # to distinguish from an incorrect full-hidden-dimension transform.
    monkeypatch.setattr(
        streaming_gptq, "_apply_hadamard",
        lambda x, had_rem_dim, rem_dim: x.flip(-1))
    streaming_gptq._install_online_hadamards(layer, config)

    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
    attention = layer.self_attn.o_proj(x)
    mlp_x = torch.arange(256, dtype=torch.float32).reshape(1, 1, 256)
    mlp = layer.mlp.down_proj(mlp_x)

    assert torch.equal(attention, torch.tensor([[[2., 3., 0., 1.]]]))
    assert torch.equal(mlp, matmul_grouped_h256(mlp_x))
    assert tuple(layer.state_dict()) == original_keys
    assert dict(layer.named_modules())["self_attn.o_proj"] is layer.self_attn.o_proj
    assert dict(layer.named_modules())["mlp.down_proj"] is layer.mlp.down_proj


import pytest
from transformers import LlamaConfig, Qwen2Config, Qwen3Config
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


@pytest.mark.parametrize(
    ("config", "decoder_cls"),
    [
        (LlamaConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2),
         LlamaDecoderLayer),
        (Qwen2Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2),
         Qwen2DecoderLayer),
        (Qwen3Config(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8),
         Qwen3DecoderLayer),
    ],
)
def test_supported_family_layer_preserves_state_and_installs_hooks(
        config, decoder_cls):
    source = decoder_cls(config, 0).to(dtype=torch.float16)
    rotation_utils.pad_mlp_modules(source)
    prefix = "model.layers.0."
    tensors = {prefix + key: value.clone()
               for key, value in source.state_dict().items()}

    actual = streaming_gptq._layer_from_tensors(
        config, 0, tensors, prefix)

    assert isinstance(actual, decoder_cls)
    assert actual.state_dict().keys() == source.state_dict().keys()
    assert len(actual.self_attn.o_proj._forward_pre_hooks) == 1
    assert len(actual.mlp.down_proj._forward_pre_hooks) == 1
    _, rotary_cls = streaming_gptq._calibration_types(config)
    rotary = rotary_cls(config=config)
    x = torch.zeros(1, 4, config.hidden_size, dtype=torch.float16)
    positions = torch.arange(4).unsqueeze(0)
    cos, sin = rotary(x, positions)
    expected_head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads)
    assert cos.shape[-1] == expected_head_dim
    assert sin.shape == cos.shape


def test_unsupported_gptq_family_is_rejected():
    config = SimpleNamespace(model_type="unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        streaming_gptq._calibration_types(config)

def test_fp32_embedding_produces_fp16_calibration_activations(tmp_path):
    global_path = tmp_path / "model-global.safetensors"
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    save_file({"model.embed_tokens.weight": embedding}, global_path)
    loader = [
        (torch.tensor([[0, 3, 7]], dtype=torch.long),),
        (torch.tensor([[2, 5, 1]], dtype=torch.long),),
    ]
    config = SimpleNamespace(hidden_size=4)

    inps, outs = streaming_gptq._initial_activations(
        global_path, loader, config, torch.device("cpu"))

    assert inps.dtype == torch.float16
    assert outs.dtype == torch.float16
    assert torch.equal(inps[0], embedding[loader[0][0][0]].half())
