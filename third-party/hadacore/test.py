import math

import torch
from hadacore_for_hip import hadacore


def ref_hadamard(x):
    shape = x.shape
    n = shape[-1]
    y = x.reshape(-1, n).float().clone()
    step = 1
    while step < n:
        y = y.reshape(-1, n // (2 * step), 2, step)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        step <<= 1
    return (y / math.sqrt(float(n))).reshape(shape)


def tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float16:
        return 2e-2
    if dtype == torch.bfloat16:
        return 8e-2
    raise AssertionError(dtype)


def check(rows, n, dtype):
    x = torch.randn((rows, n), device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(float(n))
    got = hadacore(x, scale).float()
    ref = ref_hadamard(x)
    diff = (got - ref).abs().max().item()
    print(f"rows={rows:4d} n={n:5d} dtype={str(dtype):14s} max_diff={diff:.6g}")
    if not torch.allclose(got, ref, atol=tolerance(dtype), rtol=0):
        raise AssertionError(f"failed rows={rows} n={n} dtype={dtype} diff={diff}")


def check_shape(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    n = shape[-1]
    scale = 1.0 / math.sqrt(float(n))
    got = hadacore(x, scale).float()
    ref = ref_hadamard(x)
    diff = (got - ref).abs().max().item()
    print(f"shape={str(shape):18s} dtype={str(dtype):14s} max_diff={diff:.6g}")
    if not torch.allclose(got, ref, atol=tolerance(dtype), rtol=0):
        raise AssertionError(f"failed shape={shape} dtype={dtype} diff={diff}")


def check_twice(n, dtype):
    x = torch.randn((n,), device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(float(n))
    got = hadacore(hadacore(x, scale), scale).float()
    diff = (got - x.float()).abs().max().item()
    print(f"twice n={n:5d} dtype={str(dtype):14s} max_diff={diff:.6g}")
    if not torch.allclose(got, x.float(), atol=tolerance(dtype), rtol=0):
        raise AssertionError(f"twice failed n={n} dtype={dtype} diff={diff}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    torch.manual_seed(0)
    for n in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        check(3, n, torch.float16)
        check(7, n, torch.bfloat16)

    for shape in [(2048, 128, 32), (2048, 172, 64)]:
        check_shape(shape, torch.float16)
        check_shape(shape, torch.bfloat16)

    check(3, 512, torch.float32)
    check(3, 8192, torch.float16)

    for n in [32, 64, 128, 256, 4096]:
        check_twice(n, torch.float16)

    print("All hadacore tests passed.")


if __name__ == "__main__":
    main()
