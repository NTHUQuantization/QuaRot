import argparse
import math

import torch

import attention_fusion_hip


def hadamard(x):
    y = x.clone()
    size = y.size(-1)
    stride = 1
    while stride < size:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        a = y[..., :, :stride].clone()
        b = y[..., :, stride:].clone()
        y[..., :, :stride] = a + b
        y[..., :, stride:] = a - b
        y = y.reshape(*x.shape)
        stride <<= 1
    return y / math.sqrt(size)


def apply_rope_k(k, pos, theta=10000.0):
    x = k.float().reshape(*k.shape[:-1], -1, 2)
    pair_idx = torch.arange(x.size(-2), device=k.device, dtype=torch.float32)
    freq = (1.0 / theta) ** (2.0 * pair_idx / k.size(-1))
    angle = pos * freq
    c = torch.cos(angle)
    s = torch.sin(angle)
    y0 = x[..., 0] * c - x[..., 1] * s
    y1 = x[..., 0] * s + x[..., 1] * c
    return torch.stack((y0, y1), dim=-1).reshape_as(k).half()


def quantize_s4(x):
    xf = x.float()
    scale = (xf.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(xf / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    return packed, torch.stack((scale, scale * 8.0), dim=-1).half()


def quantize_grouped(x, group_size):
    grouped = x.float().reshape(-1, x.size(-1) // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(grouped / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*x.shape[:-1], x.size(-1) // 2)
    return packed.contiguous(), scale.reshape(*x.shape[:-1], x.size(-1) // group_size).half()


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


def bench_kernel1(batch, heads, page_size, seq_len, iters, warmup, apply_rope):
    torch.manual_seed(0)
    k = torch.randn((batch, heads, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    pages = (seq_len + page_size - 1) // page_size
    indptr = torch.arange(0, (batch + 1) * pages, pages, device="cuda", dtype=torch.int32)
    indices = torch.arange(batch * pages, device="cuda", dtype=torch.int32)
    last = torch.full((batch,), seq_len - (pages - 1) * page_size, device="cuda", dtype=torch.int32)

    kv_data = torch.empty((batch * pages, 1, 2, heads, page_size, 64), device="cuda", dtype=torch.uint8)
    kv_param = torch.empty((batch * pages, 1, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    attention_fusion_hip.append_kv_had_quant_inplace(
        k, v, kv_data, kv_param, indptr, indices, last, 1, 0, heads, page_size, batch, apply_rope
    )
    torch.cuda.synchronize()

    k_ref_input = apply_rope_k(k, seq_len - 1) if apply_rope else k
    k_ref_packed, k_ref_param = quantize_s4(hadamard(k_ref_input))
    v_ref_packed, v_ref_param = quantize_s4(hadamard(v))

    entry = (seq_len - 1) % page_size
    page_ids = indices[indptr[:-1] + (seq_len - 1) // page_size]
    k_got = kv_data[page_ids, 0, 0, :, entry, :].reshape(batch, heads, 64)
    v_got = kv_data[page_ids, 0, 1, :, entry, :].reshape(batch, heads, 64)
    kp_got = kv_param[page_ids, 0, 0, :, entry, :].reshape(batch, heads, 2)
    vp_got = kv_param[page_ids, 0, 1, :, entry, :].reshape(batch, heads, 2)

    k_mismatch = (k_got != k_ref_packed).sum().item()
    v_mismatch = (v_got != v_ref_packed).sum().item()
    kp_err = (kp_got.float() - k_ref_param.float()).abs().max().item()
    vp_err = (vp_got.float() - v_ref_param.float()).abs().max().item()
    ms = time_call(
        lambda: attention_fusion_hip.append_kv_had_quant_inplace(
            k, v, kv_data, kv_param, indptr, indices, last, 1, 0, heads, page_size, batch, apply_rope
        ),
        iters,
        warmup,
    )
    print(
        f"kernel1 batch={batch} heads={heads} rope={int(apply_rope)} "
        f"mismatch_k={k_mismatch} mismatch_v={v_mismatch} "
        f"scale_err_k={kp_err:.6g} scale_err_v={vp_err:.6g} avg_ms={ms:.6f}"
    )


def bench_kernel3(rows, iters, warmup):
    torch.manual_seed(1)
    out = torch.randn((rows, 4096), device="cuda", dtype=torch.float16)
    packed = torch.empty((rows, 2048), device="cuda", dtype=torch.uint8)
    scales = torch.empty((rows, 16), device="cuda", dtype=torch.float16)
    attention_fusion_hip.output_had_quant_inplace(out, packed, scales)
    torch.cuda.synchronize()
    ref_packed, ref_scales = quantize_grouped(hadamard(out.reshape(rows, 16, 256)).reshape(rows, 4096), 256)
    mismatch = (packed != ref_packed).sum().item()
    scale_err = (scales.float() - ref_scales.float()).abs().max().item()
    ms = time_call(lambda: attention_fusion_hip.output_had_quant_inplace(out, packed, scales), iters, warmup)
    print(f"kernel3 rows={rows} mismatch={mismatch} scale_err={scale_err:.6g} avg_ms={ms:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()

    bench_kernel1(args.batch, args.heads, args.page_size, args.seq_len, args.iters, args.warmup, False)
    bench_kernel1(args.batch, args.heads, args.page_size, args.seq_len, args.iters, args.warmup, True)
    bench_kernel3(args.rows, args.iters, args.warmup)


if __name__ == "__main__":
    main()
