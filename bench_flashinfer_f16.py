import argparse
import math

import torch

import flashinfer_test._HIP as flashinfer_hip


def make_paged_kv(k, v, page_size):
    batch, heads, seq_len, head_dim = k.shape
    assert batch == 1, "this smoke benchmark currently uses batch=1"
    pages = (seq_len + page_size - 1) // page_size
    kv = torch.zeros(
        (pages, 1, 2, heads, page_size, head_dim),
        device=k.device,
        dtype=k.dtype,
    )
    for page_idx in range(pages):
        begin = page_idx * page_size
        end = min(begin + page_size, seq_len)
        kv[page_idx, 0, 0, :, : end - begin, :] = k[0, :, begin:end, :]
        kv[page_idx, 0, 1, :, : end - begin, :] = v[0, :, begin:end, :]
    return kv


def bench(seq_len, heads, head_dim, page_size, iters, warmup):
    torch.manual_seed(0)
    q = torch.randn((1, heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((1, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((1, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    kv = make_paged_kv(k, v, page_size)
    pages = kv.size(0)
    kv_param = torch.zeros((pages, 1, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    kv_indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last_page_offset = torch.tensor(
        [seq_len - (pages - 1) * page_size],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty_like(q)

    flashinfer_hip.batch_decode_f16(
        out, q, kv, kv_param, kv_indptr, kv_indices, last_page_offset,
        1, 0, heads, page_size, 1,
    )
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None, :].float(), k.float(), v.float()
    ).squeeze(2)
    err = (out.float() - ref).abs()

    for _ in range(warmup):
        flashinfer_hip.batch_decode_f16(
            out, q, kv, kv_param, kv_indptr, kv_indices, last_page_offset,
            1, 0, heads, page_size, 1,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        flashinfer_hip.batch_decode_f16(
            out, q, kv, kv_param, kv_indptr, kv_indices, last_page_offset,
            1, 0, heads, page_size, 1,
        )
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    print(
        f"L={seq_len:5d} pages={pages:3d} "
        f"maxerr={err.max().item():.6g} meanerr={err.mean().item():.6g} "
        f"avg_ms={avg_ms:.6f}"
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
        bench(seq_len, args.heads, args.head_dim, args.page_size, args.iters, args.warmup)


if __name__ == "__main__":
    main()
