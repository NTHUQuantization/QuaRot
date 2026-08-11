"""HIP-only correctness checks for the unified QuaRot fused extension."""

import math

import pytest
import torch


_HIP = pytest.importorskip("quarot._HIP")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def _hadamard(x, *, output_dtype=torch.float32):
    width = x.size(-1)
    y = x.float().clone()
    stride = 1
    while stride < width:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        left, right = y[..., :stride].clone(), y[..., stride:].clone()
        y[..., :stride], y[..., stride:] = left + right, left - right
        y = y.reshape_as(x)
        stride <<= 1
    return (y / math.sqrt(width)).to(output_dtype)


def _pack_s4(x, scale):
    q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
    return (q[..., 0::2].to(torch.uint8) & 0x0F) | ((q[..., 1::2].to(torch.uint8) & 0x0F) << 4)


def _reference_attention(x):
    # Same layout as OnlineHadamard(num_heads): rotate across heads for every
    # head-dimension coordinate, then quantize the flattened projection input.
    b, s, h, d = x.shape
    y = _hadamard(x.transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(b, s, h * d)
    scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    return _pack_s4(y, scale), scale


@pytest.mark.parametrize("batch,seq,heads,head_dim", [(1, 1, 2, 128), (2, 16, 32, 128), (1, 128, 64, 128)])
def test_fused_attention_matches_quarot_layout(batch, seq, heads, head_dim):
    torch.manual_seed(heads)
    x = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_attention_hadamard_quant(x.contiguous(), heads)
    expected_packed, expected_scale = _reference_attention(x)
    torch.cuda.synchronize()
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("batch,width", [(1, 128), (2, 256), (2, 1024), (4, 4096), (1, 8192)])
def test_fused_ffn_matches_silu_hadamard_quant(batch, width):
    torch.manual_seed(width)
    gate = torch.randn(batch, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant(gate.contiguous(), up.contiguous())
    # The fused reference promotes FP16 inputs before SiLU, multiplication,
    # and every Hadamard butterfly. FP16 intermediates change values at
    # quantization thresholds and are not the fused mathematical contract.
    gate_f = gate.float()
    y = _hadamard(torch.nn.functional.silu(gate_f) * up.float())
    expected_scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(y, expected_scale))
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size,positions", [(4, [1, 2, 4, 5]), (8, [1, 8, 9])])
def test_fused_kv_append_writes_target_asymmetric_cache(head_dim, page_size, positions):
    batch, heads, pages, layers = 2, 4, 2, 1
    data = torch.empty((batch * pages, layers, 2, heads, page_size, head_dim // 2), device="cuda", dtype=torch.uint8)
    params = torch.empty((batch * pages, layers, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    for position in positions:
        key = torch.randn(batch, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        current_pages = (position + page_size - 1) // page_size
        indptr = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * current_pages
        indices = ((torch.arange(current_pages, device="cuda", dtype=torch.int32) * batch).unsqueeze(0) + torch.arange(batch, device="cuda", dtype=torch.int32).unsqueeze(1)).flatten()
        offset = (position - 1) % page_size + 1
        last = torch.full((batch,), offset, device="cuda", dtype=torch.int32)
        _HIP.fused_append_kv_i4(data, params, indptr, indices, last, key, value, layers, 0, heads, page_size, batch)
        page = (position - 1) // page_size
        slot = (position - 1) % page_size
        page_ids = indices[indptr[:-1] + page]
        for source, plane in ((key, 0), (value, 1)):
            rotated = _hadamard(source)
            xmin, xmax = rotated.amin(dim=-1, keepdim=True), rotated.amax(dim=-1, keepdim=True)
            scale = ((xmax - xmin).clamp_min(1e-5) / 15).half()
            zero = (-xmin).half()
            expected = torch.round((rotated + zero) / scale).clamp(0, 15).to(torch.uint8)
            expected = expected[..., 0::2] | (expected[..., 1::2] << 4)
            assert torch.equal(data[page_ids, 0, plane, :, slot], expected)
            assert torch.equal(params[page_ids, 0, plane, :, slot, 0], scale.squeeze(-1))
            assert torch.equal(params[page_ids, 0, plane, :, slot, 1], zero.squeeze(-1))


@pytest.mark.parametrize("position", [1, 31, 32, 33, 63, 64, 65, 1024])
def test_fused_kv_append_page_boundaries(position):
    page_size, batch, heads, layers = 32, 1, 2, 1
    page_count = (position + page_size - 1) // page_size
    data = torch.full((page_count, layers, 2, heads, page_size, 64), 0xA5, device="cuda", dtype=torch.uint8)
    params = torch.full((page_count, layers, 2, heads, page_size, 2), 3.0, device="cuda", dtype=torch.float16)
    key = torch.zeros(batch, heads, 128, device="cuda", dtype=torch.float16)
    value = torch.full_like(key, 2.0)
    indptr = torch.tensor([0, page_count], device="cuda", dtype=torch.int32)
    indices = torch.arange(page_count, device="cuda", dtype=torch.int32)
    last = torch.tensor([(position - 1) % page_size + 1], device="cuda", dtype=torch.int32)
    _HIP.fused_append_kv_i4(data, params, indptr, indices, last, key, value, layers, 0, heads, page_size, batch)
    torch.cuda.synchronize()
    page, slot = (position - 1) // page_size, (position - 1) % page_size
    assert torch.all(data[page, 0, :, :, slot] != 0xA5)
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[page, 0, :, :, slot] = False
    assert torch.all(data[mask] == 0xA5), "fused append wrote outside the target cache slot"


def test_special_values_and_int4_endpoints():
    for fill in (0.0, 1.0, -1.0, 100.0):
        gate = torch.full((2, 256), fill, device="cuda", dtype=torch.float16)
        up = torch.ones_like(gate)
        packed, scale = _HIP.fused_ffn_silu_hadamard_quant(gate, up)
        gate_f = gate.float()
        y = _hadamard(torch.nn.functional.silu(gate_f) * up.float())
        expected_scale = (y.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
        torch.cuda.synchronize()
        assert torch.equal(packed, _pack_s4(y, expected_scale))
        assert torch.equal(scale, expected_scale)
    endpoints = torch.tensor([-8, -7, -1, 0, 1, 7], device="cuda", dtype=torch.int8)
    expected = torch.tensor([0x98, 0x0F, 0x71], device="cuda", dtype=torch.uint8)
    assert torch.equal(_pack_s4(endpoints, torch.tensor(1.0, device="cuda")), expected)


def test_unsupported_contracts_fail_before_launch():
    with pytest.raises(RuntimeError, match="float16"):
        _HIP.fused_ffn_silu_hadamard_quant(
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
            torch.ones(1, 128, device="cuda", dtype=torch.bfloat16),
        )
    base = torch.ones(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        _HIP.fused_ffn_silu_hadamard_quant(base[:, ::2], base[:, ::2])
    data = torch.empty((1, 1, 2, 1, 1, 64), device="cuda", dtype=torch.uint8)
    params = torch.empty((1, 1, 2, 1, 1, 2), device="cuda", dtype=torch.float16)
    key = torch.zeros(1, 1, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="metadata must be CUDA/HIP"):
        _HIP.fused_append_kv_i4(data, params, torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32), torch.tensor([1], dtype=torch.int32),
            key, key, 1, 0, 1, 1, 1)


def test_fused_ffn_generalized_llama2_7b_width():
    from quarot.functional.hadamard import get_hadK

    width, remainder = 11008, 43
    torch.manual_seed(width)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    hadamard, order = get_hadK(width)
    assert order == remainder
    hadamard = hadamard.to(device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_general(
        gate, up, hadamard)

    inner = width // remainder
    values = torch.nn.functional.silu(gate.float()) * up.float()
    values = values.view(-1, remainder, inner)
    stride = 1
    while stride < inner:
        view = values.view(-1, remainder, inner // (2 * stride), 2, stride)
        low, high = view[..., 0, :].clone(), view[..., 1, :].clone()
        view[..., 0, :], view[..., 1, :] = low + high, low - high
        stride <<= 1
    expected = torch.matmul(hadamard.float(), values).reshape_as(gate) / width**0.5
    expected_scale = (expected.abs().amax(dim=-1, keepdim=True) / 7).half()
    expected_scale.clamp_min_(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(expected, expected_scale))
    assert torch.equal(scale, expected_scale)



@pytest.mark.parametrize("width,remainder", [
    (13824, 27),    # Llama-2 13B: DCT27 x H512
    (22016, 43),    # CodeLlama 34B: DCT43 x H512
    (25600, 25),    # Qwen3-32B: DCT25 x H1024
    (27648, 27),    # Qwen2.5-32B: DCT27 x H1024
    (28672, 28),    # Llama-2 70B: H28 x H1024
])
def test_fused_ffn_generalized_benchmark_widths(width, remainder):
    from quarot.functional.hadamard import get_hadK

    torch.manual_seed(width)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    hadamard, order = get_hadK(width)
    assert order == remainder
    hadamard = hadamard.to(device="cuda", dtype=torch.float16)
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_general(
        gate, up, hadamard)

    inner = width // remainder
    values = torch.nn.functional.silu(gate.float()) * up.float()
    values = _hadamard(
        values.view(-1, remainder, inner), output_dtype=torch.float32)
    expected = torch.matmul(hadamard.float(), values).reshape_as(gate)
    expected /= math.sqrt(remainder)
    expected_scale = (expected.abs().amax(dim=-1, keepdim=True) / 7).half()
    expected_scale.clamp_min_(torch.finfo(torch.float16).tiny)
    torch.cuda.synchronize()
    assert torch.equal(packed, _pack_s4(expected, expected_scale))
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("width,remainder", [
    (11008, 43), (13824, 27), (22016, 43),
    (25600, 25), (27648, 27), (28672, 28),
])
def test_single_fp16lds_large_ffn_agrees_with_fp32_contract(width, remainder):
    from quarot.functional.hadamard import get_hadK, _dct_remainder

    torch.manual_seed(width + 17)
    gate = torch.randn(1, width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    matrix = get_hadK(width)[0]
    matrix = matrix.cuda().half()
    packed, scale = _HIP.fused_ffn_silu_hadamard_quant_single_fp16lds(
        gate, up, matrix)
    reference_packed, reference_scale = (
        _HIP.fused_ffn_silu_hadamard_quant_general(gate, up, matrix))
    torch.cuda.synchronize()
    assert torch.equal(scale, reference_scale)
    # FP16 LDS intentionally rounds at the inner/remainder boundary. Packed
    # decisions should remain stable except for values at INT4 thresholds.
    assert (packed == reference_packed).float().mean().item() >= 0.995


def test_gqa_prefill_quantizes_before_cache_head_expansion(monkeypatch):
    import quarot.transformers.kv_cache as kv_cache

    observed_heads = []
    original_quantize = kv_cache.asym_quantize_and_pack_i4

    def recording_quantize(x):
        observed_heads.append(x.shape[2])
        return original_quantize(x)

    monkeypatch.setattr(kv_cache, "asym_quantize_and_pack_i4", recording_quantize)
    cache = kv_cache.MultiLayerPagedKVCache4Bit(
        batch_size=2, page_size=16, max_seq_len=16, device="cuda",
        n_layers=1, num_heads=4, num_kv_heads=2, head_dim=128)
    torch.manual_seed(3408)
    key = torch.randn(2, 16, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    returned_key, returned_value = cache.update(key, value, 0, {})
    torch.cuda.synchronize()

    assert observed_heads == [2, 2]
    assert returned_key is key and returned_value is value
    for plane in (0, 1):
        assert torch.equal(cache.pages[:, 0, plane, 0], cache.pages[:, 0, plane, 1])
        assert torch.equal(cache.pages[:, 0, plane, 2], cache.pages[:, 0, plane, 3])
        assert torch.equal(cache.scales[:, 0, plane, 0], cache.scales[:, 0, plane, 1])
        assert torch.equal(cache.scales[:, 0, plane, 2], cache.scales[:, 0, plane, 3])
