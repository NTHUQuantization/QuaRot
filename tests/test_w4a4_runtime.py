import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from w4a4_runtime.checkpoint import audit_checkpoint, load_checkpoint_tensors, save_sharded_checkpoint
from w4a4_runtime.config import W4A4Config
from w4a4_runtime.hadamard import hadamard_transform, paley_hadamard_28
from w4a4_runtime.packing import (
    PackedActivation,
    PackedWeight,
    linear_reference,
    pack_activation_reference,
    pack_int4,
    pack_weight_reference,
    unpack_int4,
)
from w4a4_runtime.model import (
    CacheTransaction,
    QuaRotW4A4LlamaForCausalLM,
    W4A4PagedCache,
    _quantize_kv_biased,
    apply_rope,
)
from w4a4_runtime.ops import (
    append_kv_biased_native,
    append_kv_chunk_biased_native,
    cross_head_hadamard_native,
    hadamard_quantize,
    qkv_rope_hadamard,
    quantize_kv_biased_native,
    rmsnorm_quantize,
    silu_hadamard_quantize,
    virtual_verify_metadata_native,
)
from w4a4_runtime.rotation import _left_rotate_transpose, _right_rotate, _rotation_signs


def test_signed_int4_round_trip():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(16)
    assert torch.equal(unpack_int4(pack_int4(values)), values)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_integer_grouped_linear_matches_explicit_dequant(rows):
    torch.manual_seed(7)
    x = torch.randn(rows, 256, dtype=torch.float16)
    weight = torch.randn(32, 256, dtype=torch.float16)
    px = pack_activation_reference(x)
    pw = pack_weight_reference(weight)
    actual = linear_reference(px, pw)

    qx = unpack_int4(px.qdata).float().reshape(rows, 2, 128)
    qw_rows = pw.qweight.permute(0, 2, 1, 3).reshape(32, 128)
    qw = unpack_int4(qw_rows).float().reshape(32, 2, 128)
    dx = (qx * px.scales.float().unsqueeze(-1)).reshape(rows, 256)
    dw = (qw * pw.scales.float().unsqueeze(-1)).reshape(32, 256)
    expected = (dx @ dw.T).half()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)


def test_hadamard_14336_is_norm_preserving():
    torch.manual_seed(9)
    x = torch.randn(2, 14336)
    transformed = hadamard_transform(x)
    torch.testing.assert_close(transformed.norm(), x.norm(), rtol=2e-5, atol=2e-5)
    h28 = paley_hadamard_28()
    torch.testing.assert_close(h28 @ h28.T, torch.eye(28), rtol=1e-5, atol=1e-5)


def test_structured_global_rotation_matches_dense_definition():
    signs = _rotation_signs(0, "cpu")
    h = hadamard_transform(torch.eye(4096))
    q = h * signs.unsqueeze(0)
    torch.manual_seed(10)
    right_weight = torch.randn(2, 4096, dtype=torch.float16)
    left_weight = torch.randn(4096, 2, dtype=torch.float16)
    right_original = right_weight.clone()
    torch.testing.assert_close(
        _right_rotate(right_weight, signs),
        (right_original.float() @ q).half(),
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        _left_rotate_transpose(left_weight, signs),
        (q.T @ left_weight.float()).half(),
        rtol=2e-3,
        atol=2e-3,
    )


def test_sharded_checkpoint_round_trip(tmp_path):
    tensors = {
        "a": torch.arange(256, dtype=torch.uint8),
        "b": torch.arange(64, dtype=torch.float16),
    }
    config = replace(W4A4Config(), weight_quant_method="rtn")
    save_sharded_checkpoint(
        tensors,
        tmp_path,
        model_config={"model_type": "llama"},
        quantization_config=config,
        max_shard_bytes=300,
    )
    restored, model_config, restored_config = load_checkpoint_tensors(tmp_path)
    assert model_config["model_type"] == "llama"
    assert restored_config == config
    assert set(restored) == set(tensors)
    for name in tensors:
        assert torch.equal(restored[name], tensors[name])
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 2


def test_checkpoint_audit_rejects_shadow_tensor(tmp_path):
    tensors = {
        "model.embed_tokens.qweight": torch.zeros(4, 128, dtype=torch.uint8),
        "model.embed_tokens.scales": torch.ones(4, 1, dtype=torch.float16),
    }
    config = replace(W4A4Config(), weight_quant_method="rtn")
    save_sharded_checkpoint(tensors, tmp_path, model_config={"model_type": "llama"}, quantization_config=config)
    report = audit_checkpoint(tmp_path)
    assert report["tensor_count"] == 2

    tensors["model.layers.0.weight"] = torch.ones(2, dtype=torch.float16)
    save_sharded_checkpoint(tensors, tmp_path, model_config={"model_type": "llama"}, quantization_config=config)
    with pytest.raises(ValueError, match="non-packed"):
        audit_checkpoint(tmp_path)


def test_composed_quantization_apis_match_reference():
    torch.manual_seed(13)
    x = torch.randn(2, 128, dtype=torch.float16)
    rms = rmsnorm_quantize(x, 1e-5, require_native=False)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    expected = pack_activation_reference(normalized.half())
    assert torch.equal(rms.qdata, expected.qdata)
    torch.testing.assert_close(rms.scales, expected.scales, rtol=0, atol=0)
    rotated = hadamard_quantize(x, "h128", require_native=False)
    expected_rotated = pack_activation_reference(hadamard_transform(x).half())
    assert torch.equal(rotated.qdata, expected_rotated.qdata)


def test_cache_transaction_commit_and_rollback():
    cache = type("Cache", (), {"seq_len": 20})()
    tx = CacheTransaction(cache, 12, 20)
    assert cache.seq_len == 12
    tx.commit(5)
    assert cache.seq_len == 17
    with pytest.raises(RuntimeError):
        tx.rollback()

    cache.seq_len = 20
    tx = CacheTransaction(cache, 12, 20)
    tx.rollback()
    assert cache.seq_len == 12


def test_full_model_skeleton_has_only_packed_persistent_weights():
    config = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    model = QuaRotW4A4LlamaForCausalLM(config, W4A4Config(), device="meta")
    assert not any(isinstance(module, torch.nn.Linear) for module in model.modules())
    assert all(name.endswith(("qweight", "scales")) for name, _ in model.named_buffers())
    assert all(value.device.type == "meta" for _, value in model.named_buffers())


def test_virtual_verify_metadata_aliases_prefix_across_page_boundary():
    allocation = SimpleNamespace(
        pages_per_batch=3,
        total_pages=6,
        page_size=128,
    )
    cache = W4A4PagedCache(
        batch_size=2,
        seq_len=127,
        max_seq_len=384,
        page_size=128,
        allocation=allocation,
        kv_data=[torch.empty(0)],
        kv_params=[],
    )
    metadata = QuaRotW4A4LlamaForCausalLM._virtual_verify_metadata(cache, 127, 4)
    assert metadata.batch_size == 8
    assert metadata.indptr.tolist() == [0, 1, 3, 5, 7, 8, 10, 12, 14]
    assert metadata.last_page_offset.tolist() == [128, 1, 2, 3, 128, 1, 2, 3]
    assert metadata.indices.tolist() == [0, 0, 1, 0, 1, 0, 1, 3, 3, 4, 3, 4, 3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires R9700")
@pytest.mark.parametrize("requests", [2, 16])
def test_flashinfer_virtual_requests_may_alias_one_physical_page(requests):
    import flashinfer_test._HIP as flashinfer_hip

    torch.manual_seed(21)
    key = torch.randn(1, requests, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    kp, ks = _quantize_kv_biased(key, 0.95)
    vp, vs = _quantize_kv_biased(value, 0.95)
    data = torch.zeros(1, 1, 2, 8, 128, 64, device="cuda", dtype=torch.uint8)
    params = torch.zeros(1, 1, 2, 8, 128, 2, device="cuda", dtype=torch.float16)
    data[0, 0, 0, :, :requests] = kp[0].transpose(0, 1)
    data[0, 0, 1, :, :requests] = vp[0].transpose(0, 1)
    params[0, 0, 0, :, :requests] = ks[0].transpose(0, 1)
    params[0, 0, 1, :, :requests] = vs[0].transpose(0, 1)
    query = torch.randn(requests, 32, 128, device="cuda", dtype=torch.float16)

    def decode(q, indptr, indices, last):
        out = torch.empty_like(q)
        flashinfer_hip.batch_decode_i4_gqa(
            out, q, data, params, indptr, indices, last, 1, 0, 32, 8, 128, q.size(0)
        )
        return out

    virtual = decode(
        query,
        torch.arange(requests + 1, device="cuda", dtype=torch.int32),
        torch.zeros(requests, device="cuda", dtype=torch.int32),
        torch.arange(1, requests + 1, device="cuda", dtype=torch.int32),
    )
    separate = torch.cat(
        [
            decode(
                query[index : index + 1],
                torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                torch.tensor([0], device="cuda", dtype=torch.int32),
                torch.tensor([1 + index], device="cuda", dtype=torch.int32),
            )
            for index in range(requests)
        ]
    )
    torch.testing.assert_close(virtual, separate, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires R9700")
def test_native_kernel_matches_reference():
    from w4a4_runtime.ops import quantize_activation, w4a4_linear

    torch.manual_seed(11)
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    px_ref = pack_activation_reference(x)
    pw = pack_weight_reference(weight)
    px = quantize_activation(x)
    assert torch.equal(px.qdata, px_ref.qdata)
    torch.testing.assert_close(px.scales, px_ref.scales, rtol=0, atol=0)
    actual = w4a4_linear(px, pw)
    # ROCm does not implement integer einsum/baddbmm; keep the integer oracle
    # on CPU and compare only the native result on-device.
    expected = linear_reference(
        PackedActivation(px.qdata.cpu(), px.scales.cpu(), px.rows, px.cols, px.group_size),
        PackedWeight(
            pw.qweight.cpu(),
            pw.scales.cpu(),
            pw.out_features,
            pw.in_features,
            pw.group_size,
        ),
    ).to("cuda")
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires R9700")
@pytest.mark.parametrize("rows", [2, 4, 5, 8, 9, 16])
def test_wmma_parallel_buckets_match_integer_reference(rows):
    from w4a4_runtime.ops import quantize_activation, w4a4_linear

    torch.manual_seed(40 + rows)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    packed_x = quantize_activation(x)
    packed_weight = pack_weight_reference(weight)
    actual = w4a4_linear(packed_x, packed_weight)
    expected = linear_reference(
        PackedActivation(
            packed_x.qdata.cpu(), packed_x.scales.cpu(), rows, 256, packed_x.group_size
        ),
        PackedWeight(
            packed_weight.qweight.cpu(),
            packed_weight.scales.cpu(),
            packed_weight.out_features,
            packed_weight.in_features,
            packed_weight.group_size,
        ),
    ).to("cuda")
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires R9700")
def test_fused_attention_transforms_and_kv4_match_references():
    from w4a4_runtime.hadamard import cross_head_hadamard

    torch.manual_seed(31)
    qkv = torch.randn(2, 3, 6144, device="cuda", dtype=torch.float16)
    positions = torch.tensor([127, 128, 129], device="cuda", dtype=torch.int64)
    q, k, v = qkv_rope_hadamard(qkv, positions, 500000.0)
    q_ref, k_ref, v_ref = torch.split(qkv, (4096, 1024, 1024), dim=-1)
    q_ref = q_ref.view(2, 3, 32, 128).transpose(1, 2)
    k_ref = k_ref.view(2, 3, 8, 128).transpose(1, 2)
    v_ref = v_ref.view(2, 3, 8, 128).transpose(1, 2)
    q_ref = hadamard_transform(apply_rope(q_ref, positions, 500000.0))
    k_ref = hadamard_transform(apply_rope(k_ref, positions, 500000.0))
    torch.testing.assert_close(q, q_ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(k, k_ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)

    attention = torch.randn(2, 3, 32, 128, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        cross_head_hadamard_native(attention),
        cross_head_hadamard(attention),
        rtol=2e-3,
        atol=2e-3,
    )

    key = torch.randn(2, 3, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    kp, ks, vp, vs = quantize_kv_biased_native(key, value)
    kp_ref, ks_ref = _quantize_kv_biased(key, 0.95)
    vp_ref, vs_ref = _quantize_kv_biased(value, 0.95)
    assert torch.equal(kp, kp_ref)
    assert torch.equal(vp, vp_ref)
    torch.testing.assert_close(ks, ks_ref, rtol=0, atol=0)
    torch.testing.assert_close(vs, vs_ref, rtol=0, atol=0)

    cache_data = torch.zeros(6, 1, 2, 8, 128, 64, device="cuda", dtype=torch.uint8)
    cache_params = torch.zeros(6, 1, 2, 8, 128, 2, device="cuda", dtype=torch.float16)
    decode_key = key[:, 0]
    decode_value = value[:, 0]
    page_ids = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    append_kv_biased_native(
        decode_key, decode_value, cache_data, cache_params, page_ids, 7, 128
    )
    dkp, dks = _quantize_kv_biased(decode_key, 0.95)
    dvp, dvs = _quantize_kv_biased(decode_value, 0.95)
    assert torch.equal(cache_data[page_ids, 0, 0, :, 7], dkp)
    assert torch.equal(cache_data[page_ids, 0, 1, :, 7], dvp)
    torch.testing.assert_close(cache_params[page_ids, 0, 0, :, 7], dks, rtol=0, atol=0)
    torch.testing.assert_close(cache_params[page_ids, 0, 1, :, 7], dvs, rtol=0, atol=0)

    chunk_key = key[:, :3]
    chunk_value = value[:, :3]
    allocation_indptr = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    allocation_indices = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
    cache_data.zero_()
    cache_params.zero_()
    append_kv_chunk_biased_native(
        chunk_key,
        chunk_value,
        cache_data,
        cache_params,
        allocation_indptr,
        allocation_indices,
        127,
        128,
    )
    ckp, cks = _quantize_kv_biased(chunk_key, 0.95)
    cvp, cvs = _quantize_kv_biased(chunk_value, 0.95)
    for batch in range(2):
        for offset in range(3):
            position = 127 + offset
            page = batch * 2 + position // 128
            slot = position % 128
            assert torch.equal(cache_data[page, 0, 0, :, slot], ckp[batch, offset])
            assert torch.equal(cache_data[page, 0, 1, :, slot], cvp[batch, offset])
            torch.testing.assert_close(cache_params[page, 0, 0, :, slot], cks[batch, offset], rtol=0, atol=0)
            torch.testing.assert_close(cache_params[page, 0, 1, :, slot], cvs[batch, offset], rtol=0, atol=0)

    indptr, indices, last = virtual_verify_metadata_native(cache_data, 2, 127, 4, 128, 3)
    assert indptr.tolist() == [0, 1, 3, 5, 7, 8, 10, 12, 14]
    assert indices.tolist() == [0, 0, 1, 0, 1, 0, 1, 3, 3, 4, 3, 4, 3, 4]
    assert last.tolist() == [128, 1, 2, 3, 128, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires R9700")
@pytest.mark.parametrize("rows", [1, 2, 16])
def test_fused_silu_h14336_pack_matches_reference(rows):
    torch.manual_seed(31)
    gate = torch.randn(rows, 14336, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    actual = silu_hadamard_quantize(gate, up)
    expected = silu_hadamard_quantize(gate.cpu(), up.cpu(), require_native=False)
    mismatch = torch.count_nonzero(actual.qdata.cpu() != expected.qdata).item()
    assert mismatch / actual.qdata.numel() < 2e-3
    torch.testing.assert_close(actual.scales.cpu(), expected.scales, rtol=2e-3, atol=2e-3)
