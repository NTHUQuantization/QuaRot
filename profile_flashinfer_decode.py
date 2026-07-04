import argparse

import torch

import flashinfer_test._HIP as flashinfer_hip


def quantize_s4(x):
    scale = (x.float().abs().amax(dim=-1) / 7.0).clamp_min(1e-6)
    q = torch.round(x.float() / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    dequant = q.float() * scale[..., None]
    param = torch.stack([scale, scale * 8.0], dim=-1).half()
    return packed, param, dequant


def make_paged_f16(k, v, page_size):
    batch, heads, seq_len, head_dim = k.shape
    assert batch == 1, "this profile currently uses batch=1"
    pages = (seq_len + page_size - 1) // page_size
    kv = torch.zeros((pages, 1, 2, heads, page_size, head_dim), device=k.device, dtype=k.dtype)
    for page_idx in range(pages):
        begin = page_idx * page_size
        end = min(begin + page_size, seq_len)
        kv[page_idx, 0, 0, :, : end - begin, :] = k[0, :, begin:end, :]
        kv[page_idx, 0, 1, :, : end - begin, :] = v[0, :, begin:end, :]
    return kv


def make_paged_i4(k_packed, v_packed, k_param, v_param, page_size):
    batch, heads, seq_len, packed_dim = k_packed.shape
    assert batch == 1, "this profile currently uses batch=1"
    pages = (seq_len + page_size - 1) // page_size
    kv = torch.zeros((pages, 1, 2, heads, page_size, packed_dim), device=k_packed.device, dtype=torch.uint8)
    kv_param = torch.zeros((pages, 1, 2, heads, page_size, 2), device=k_packed.device, dtype=torch.float16)
    for page_idx in range(pages):
        begin = page_idx * page_size
        end = min(begin + page_size, seq_len)
        kv[page_idx, 0, 0, :, : end - begin, :] = k_packed[0, :, begin:end, :]
        kv[page_idx, 0, 1, :, : end - begin, :] = v_packed[0, :, begin:end, :]
        kv_param[page_idx, 0, 0, :, : end - begin, :] = k_param[0, :, begin:end, :]
        kv_param[page_idx, 0, 1, :, : end - begin, :] = v_param[0, :, begin:end, :]
    return kv, kv_param


def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kv_read_bytes(seq_len, heads, head_dim):
    elements = seq_len * heads * head_dim * 2
    fp16_bytes = elements * 2
    i4_packed_bytes = elements // 2
    i4_scale_bytes = seq_len * heads * 2 * 4
    return fp16_bytes, i4_packed_bytes, i4_packed_bytes + i4_scale_bytes


def profile(seq_len, heads, head_dim, page_size, iters, warmup):
    torch.manual_seed(0)
    q = torch.randn((1, heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((1, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((1, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)

    pages = (seq_len + page_size - 1) // page_size
    kv_indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last_page_offset = torch.tensor([seq_len - (pages - 1) * page_size], device="cuda", dtype=torch.int32)

    kv_f16 = make_paged_f16(k, v, page_size)
    kv_param_f16 = torch.zeros((pages, 1, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    out_f16 = torch.empty_like(q)
    flashinfer_hip.batch_decode_f16(
        out_f16, q, kv_f16, kv_param_f16, kv_indptr, kv_indices, last_page_offset,
        1, 0, heads, page_size, 1,
    )
    torch.cuda.synchronize()
    ref_f16 = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None, :].float(), k.float(), v.float()
    ).squeeze(2)
    f16_err = (out_f16.float() - ref_f16).abs()

    k_packed, k_param, k_dequant = quantize_s4(k)
    v_packed, v_param, v_dequant = quantize_s4(v)
    kv_i4, kv_param_i4 = make_paged_i4(k_packed, v_packed, k_param, v_param, page_size)
    out_i4 = torch.empty_like(q)
    flashinfer_hip.batch_decode_i4(
        out_i4, q, kv_i4, kv_param_i4, kv_indptr, kv_indices, last_page_offset,
        1, 0, heads, page_size, 1,
    )
    torch.cuda.synchronize()
    ref_i4 = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None, :].float(), k_dequant, v_dequant
    ).squeeze(2)
    i4_err = (out_i4.float() - ref_i4).abs()
    quant_err = (out_i4.float() - ref_f16).abs()

    f16_ms = time_call(
        lambda: flashinfer_hip.batch_decode_f16(
            out_f16, q, kv_f16, kv_param_f16, kv_indptr, kv_indices, last_page_offset,
            1, 0, heads, page_size, 1,
        ),
        iters,
        warmup,
    )
    i4_ms = time_call(
        lambda: flashinfer_hip.batch_decode_i4(
            out_i4, q, kv_i4, kv_param_i4, kv_indptr, kv_indices, last_page_offset,
            1, 0, heads, page_size, 1,
        ),
        iters,
        warmup,
    )

    fp16_bytes, i4_packed_bytes, i4_total_bytes = kv_read_bytes(seq_len, heads, head_dim)
    print(
        f"L={seq_len:5d} pages={pages:3d} "
        f"f16_ms={f16_ms:.6f} i4_ms={i4_ms:.6f} speedup={f16_ms / i4_ms:.2f}x "
        f"fp16_bytes={fp16_bytes} i4_packed_bytes={i4_packed_bytes} "
        f"i4_with_scale_bytes={i4_total_bytes} "
        f"byte_reduction_packed={fp16_bytes / i4_packed_bytes:.2f}x "
        f"byte_reduction_with_scale={fp16_bytes / i4_total_bytes:.2f}x "
        f"f16_maxerr={f16_err.max().item():.6g} f16_meanerr={f16_err.mean().item():.6g} "
        f"i4_dequant_maxerr={i4_err.max().item():.6g} i4_dequant_meanerr={i4_err.mean().item():.6g} "
        f"i4_vs_f16_ref_maxerr={quant_err.max().item():.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="10,128,1024,4096")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    for seq_len in [int(x) for x in args.lengths.replace(",", " ").split()]:
        profile(seq_len, args.heads, args.head_dim, args.page_size, args.iters, args.warmup)


if __name__ == "__main__":
    main()
