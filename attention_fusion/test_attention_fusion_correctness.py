import argparse

import torch

from quarot_attention_fusion import (
    append_quantized_kv_decode,
    apply_rope_k_reference,
    hadamard_reference,
    make_uniform_paged_kv_metadata,
    quantize_attention_output,
    quantize_attention_output_hadacore256,
    quantize_attention_output_hadacore4096_experimental,
    quantize_grouped_reference,
    quantize_s4_reference,
)


def selected_decode_slot(kv_data, kv_param, metadata, seq_len):
    entry = (seq_len - 1) % metadata.page_size
    page_ids = metadata.indices[metadata.indptr[:-1] + (seq_len - 1) // metadata.page_size]
    k_got = kv_data[page_ids, 0, 0, :, entry, :]
    v_got = kv_data[page_ids, 0, 1, :, entry, :]
    kp_got = kv_param[page_ids, 0, 0, :, entry, :]
    vp_got = kv_param[page_ids, 0, 1, :, entry, :]
    return k_got, v_got, kp_got, vp_got


def assert_close_enough(name, mismatches, total, max_mismatch_rate, scale_err, max_scale_err):
    mismatch_rate = mismatches / total
    print(
        f"{name}: mismatches={mismatches}/{total} "
        f"rate={mismatch_rate:.6f} scale_err={scale_err:.6g}"
    )
    assert mismatch_rate <= max_mismatch_rate, (
        f"{name} mismatch rate {mismatch_rate:.6f} > {max_mismatch_rate:.6f}"
    )
    assert scale_err <= max_scale_err, f"{name} scale err {scale_err:.6g} > {max_scale_err:.6g}"


def test_k1(batch, heads, seq_len, page_size, rope):
    torch.manual_seed(100 + int(rope))
    key = torch.randn((batch, heads, 128), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    metadata = make_uniform_paged_kv_metadata(batch, seq_len, page_size, key.device)
    kv_data, kv_param = append_quantized_kv_decode(
        key,
        value,
        metadata,
        num_layers=1,
        layer_idx=0,
        apply_rope_to_k=rope,
    )
    torch.cuda.synchronize()

    k_ref_input = apply_rope_k_reference(key, seq_len - 1) if rope else key
    k_ref_packed, k_ref_param = quantize_s4_reference(hadamard_reference(k_ref_input))
    v_ref_packed, v_ref_param = quantize_s4_reference(hadamard_reference(value))
    k_got, v_got, kp_got, vp_got = selected_decode_slot(kv_data, kv_param, metadata, seq_len)

    k_mismatch = (k_got.reshape_as(k_ref_packed) != k_ref_packed).sum().item()
    v_mismatch = (v_got.reshape_as(v_ref_packed) != v_ref_packed).sum().item()
    k_scale_err = (kp_got.reshape_as(k_ref_param).float() - k_ref_param.float()).abs().max().item()
    v_scale_err = (vp_got.reshape_as(v_ref_param).float() - v_ref_param.float()).abs().max().item()
    total = k_ref_packed.numel()
    label = f"K1 rope={int(rope)} batch={batch} heads={heads} seq={seq_len}"
    assert_close_enough(label + " K", k_mismatch, total, 0.01, k_scale_err, 0.006)
    assert_close_enough(label + " V", v_mismatch, total, 0.01, v_scale_err, 0.006)


def test_k3(rows):
    torch.manual_seed(200 + rows)
    out = torch.randn((rows, 4096), device="cuda", dtype=torch.float16)
    packed, scales = quantize_attention_output(out)
    torch.cuda.synchronize()

    ref_packed, ref_scales = quantize_grouped_reference(
        hadamard_reference(out.reshape(rows, 16, 256)).reshape(rows, 4096),
        256,
    )
    mismatch = (packed != ref_packed).sum().item()
    scale_err = (scales.float() - ref_scales.float()).abs().max().item()
    assert_close_enough(f"K3 rows={rows}", mismatch, ref_packed.numel(), 0.01, scale_err, 0.001)


def test_k3_hadacore256(rows):
    torch.manual_seed(300 + rows)
    out = torch.randn((rows, 4096), device="cuda", dtype=torch.float16)
    packed, scales = quantize_attention_output_hadacore256(out)
    torch.cuda.synchronize()

    ref_packed, ref_scales = quantize_grouped_reference(
        hadamard_reference(out.reshape(rows, 16, 256)).reshape(rows, 4096),
        256,
    )
    mismatch = (packed != ref_packed).sum().item()
    scale_err = (scales.float() - ref_scales.float()).abs().max().item()
    assert_close_enough(f"K3 hadacore256 rows={rows}", mismatch, ref_packed.numel(), 0.01, scale_err, 0.001)


def test_k3_hadacore4096_experimental(rows):
    torch.manual_seed(400 + rows)
    out = torch.randn((rows, 4096), device="cuda", dtype=torch.float16)
    packed, scales = quantize_attention_output_hadacore4096_experimental(out)
    torch.cuda.synchronize()

    ref_packed, ref_scales = quantize_grouped_reference(hadamard_reference(out), 256)
    mismatch = (packed != ref_packed).sum().item()
    scale_err = (scales.float() - ref_scales.float()).abs().max().item()
    assert_close_enough(
        f"K3 hadacore4096 experimental rows={rows}",
        mismatch,
        ref_packed.numel(),
        0.01,
        scale_err,
        0.001,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--k3-rows", default="1,4,16")
    args = parser.parse_args()

    test_k1(args.batch, args.heads, args.seq_len, args.page_size, False)
    test_k1(args.batch, args.heads, args.seq_len, args.page_size, True)
    for rows in [int(x) for x in args.k3_rows.replace(",", " ").split()]:
        test_k3(rows)
        test_k3_hadacore256(rows)
        test_k3_hadacore4096_experimental(rows)
    print("attention fusion correctness: PASS")


if __name__ == "__main__":
    main()
