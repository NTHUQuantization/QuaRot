import argparse
import os
import sys

import torch

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "attention_fusion"))

from quarot_attention_fusion import (
    hadamard_reference,
    quantize_grouped_reference,
    quantize_s4_reference,
)
from quarot_model_integration import QuaRotFusedDecodeLayer


def dequantize_s4(packed, params):
    lo = ((packed & 0x0F).to(torch.int16) - 8).float()
    hi = (((packed >> 4) & 0x0F).to(torch.int16) - 8).float()
    q = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    return (q * params[..., 0].float().unsqueeze(-1)).half()


def ffn_reference(gate, up, group_size):
    x = torch.nn.functional.silu(gate.float()) * up.float()
    grouped = x.reshape(-1, x.size(-1) // group_size, group_size)
    rotated = hadamard_reference(grouped).reshape_as(x)
    return quantize_grouped_reference(rotated, group_size)


def run(batch, heads, seq_len, page_size, ffn_hidden, rope):
    torch.manual_seed(1234 + int(rope))
    layer = QuaRotFusedDecodeLayer(
        batch_size=batch,
        num_heads=heads,
        seq_len=seq_len,
        page_size=page_size,
        ffn_group_size=256,
    )
    query = torch.randn((batch, heads, 128), device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn((batch, ffn_hidden), device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)

    output = layer.decode_step(query, key, value, gate, up, apply_rope_to_k=rope)
    torch.cuda.synchronize()

    assert output.attention.shape == (batch, heads, 128)
    assert output.attention_packed.shape == (batch, 2048)
    assert output.attention_scales.shape == (batch, 16)
    assert output.ffn_packed.shape == (batch, ffn_hidden // 2)
    assert output.ffn_scales.shape == (batch, ffn_hidden // 256)
    assert output.kv_data.dtype == torch.uint8
    assert output.kv_param.dtype == torch.float16

    entry = (seq_len - 1) % page_size
    page_ids = layer.metadata.indices[layer.metadata.indptr[:-1] + (seq_len - 1) // page_size]
    v_packed = output.kv_data[page_ids, 0, 1, :, entry, :]
    v_param = output.kv_param[page_ids, 0, 1, :, entry, :]
    v_dequant = dequantize_s4(v_packed, v_param)

    # With seq_len=1, softmax has one key, so decode output should be V after K1's
    # Hadamard+quant path. For seq_len>1, previous cache entries are not initialized
    # by this smoke test, so only shape/API checks are meaningful.
    if seq_len == 1:
        attn_err = (output.attention.float() - v_dequant.float()).abs().max().item()
        print(f"attention_vs_dequant_v_maxerr={attn_err:.6g}")
        assert attn_err <= 0.02

    attn_ref_packed, attn_ref_scales = quantize_grouped_reference(
        hadamard_reference(output.attention.reshape(batch, 16, 256)).reshape(batch, 4096),
        256,
    )
    attn_mismatch = (output.attention_packed != attn_ref_packed).sum().item()
    attn_scale_err = (output.attention_scales.float() - attn_ref_scales.float()).abs().max().item()
    print(f"attention_k3_mismatch={attn_mismatch} scale_err={attn_scale_err:.6g}")
    assert attn_mismatch / attn_ref_packed.numel() <= 0.01
    assert attn_scale_err <= 0.001

    ffn_ref_packed, ffn_ref_scales = ffn_reference(gate, up, 256)
    ffn_mismatch = (output.ffn_packed != ffn_ref_packed).sum().item()
    ffn_scale_err = (output.ffn_scales.float() - ffn_ref_scales.float()).abs().max().item()
    print(f"ffn_mismatch={ffn_mismatch} scale_err={ffn_scale_err:.6g}")
    assert ffn_mismatch / ffn_ref_packed.numel() <= 0.01
    assert ffn_scale_err <= 0.006
    print("quarot model integration: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--ffn-hidden", type=int, default=14336)
    parser.add_argument("--rope", action="store_true")
    args = parser.parse_args()
    run(args.batch, args.heads, args.seq_len, args.page_size, args.ffn_hidden, args.rope)


if __name__ == "__main__":
    main()

