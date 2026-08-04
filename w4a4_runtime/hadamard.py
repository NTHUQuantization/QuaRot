import math

import torch


_PALEY_CACHE: dict[tuple[str, torch.dtype, bool], torch.Tensor] = {}


def sylvester_hadamard(order: int, *, device=None, dtype=torch.float32, normalized: bool = True) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Sylvester Hadamard order must be a power of two")
    result = torch.ones((1, 1), device=device, dtype=dtype)
    while result.size(0) < order:
        result = torch.cat((torch.cat((result, result), 1), torch.cat((result, -result), 1)), 0)
    return result / math.sqrt(order) if normalized else result


def _gf27_mul(lhs: int, rhs: int) -> int:
    """Multiply in GF(3^3), using x^3 + 2x + 1."""
    a = [(lhs // (3**i)) % 3 for i in range(3)]
    b = [(rhs // (3**i)) % 3 for i in range(3)]
    product = [0] * 5
    for i in range(3):
        for j in range(3):
            product[i + j] = (product[i + j] + a[i] * b[j]) % 3
    # x^3 = x + 2 and x^4 = x^2 + 2x in this field.
    for degree in (4, 3):
        coefficient = product[degree] % 3
        product[degree] = 0
        product[degree - 3] = (product[degree - 3] + 2 * coefficient) % 3
        product[degree - 2] = (product[degree - 2] + coefficient) % 3
    return product[0] + 3 * product[1] + 9 * product[2]


def _gf27_sub(lhs: int, rhs: int) -> int:
    coefficients = [((lhs // (3**i)) - (rhs // (3**i))) % 3 for i in range(3)]
    return coefficients[0] + 3 * coefficients[1] + 9 * coefficients[2]


def paley_hadamard_28(*, device=None, dtype=torch.float32, normalized: bool = True) -> torch.Tensor:
    """Construct the order-28 Paley type-I matrix over GF(27)."""
    device_key = str(torch.device(device)) if device is not None else "cpu"
    cache_key = (device_key, dtype, normalized)
    if cache_key in _PALEY_CACHE:
        return _PALEY_CACHE[cache_key]
    q = 27
    residues = {_gf27_mul(value, value) for value in range(1, q)}
    core = torch.empty((q, q), dtype=dtype, device="cpu")
    for row in range(q):
        for col in range(q):
            if row == col:
                core[row, col] = -1
            else:
                core[row, col] = 1 if _gf27_sub(row, col) in residues else -1
    result = torch.ones((q + 1, q + 1), dtype=dtype, device="cpu")
    result[1:, 1:] = core
    result = result / math.sqrt(q + 1) if normalized else result
    result = result.to(device=device)
    _PALEY_CACHE[cache_key] = result
    return result


def randomized_hadamard_4096(seed: int = 0, *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (4096,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    base = sylvester_hadamard(4096, device=device, dtype=dtype)
    return base * signs.to(device=device, dtype=dtype).unsqueeze(0), signs


def _butterfly_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester transform with a shape-independent operation order."""
    width = x.size(-1)
    if width <= 0 or width & (width - 1):
        raise ValueError("butterfly width must be a power of two")
    result = x.contiguous().clone()
    stride = 1
    while stride < width:
        view = result.reshape(*result.shape[:-1], -1, stride * 2)
        first = view[..., :stride].clone()
        second = view[..., stride:].clone()
        view[..., :stride] = first + second
        view[..., stride:] = first - second
        result = view.reshape_as(result)
        stride *= 2
    return result / math.sqrt(width)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Reference normalized transform for 128, 4096, or Llama-3 FFN width 14336."""
    width = x.size(-1)
    if width in (32, 128, 4096):
        return _butterfly_last_dim(x)
    if width == 14336:
        # The upstream QuaRot factorization is H_28 (x) H_512.
        h28 = paley_hadamard_28(device=x.device, dtype=x.dtype)
        shaped = x.reshape(*x.shape[:-1], 28, 512)
        shaped = _butterfly_last_dim(shaped)
        # An explicit fixed-order sum avoids M-dependent GEMM algorithms.
        rows = []
        for row in range(28):
            value = shaped[..., 0, :] * h28[row, 0]
            for column in range(1, 28):
                value = value + shaped[..., column, :] * h28[row, column]
            rows.append(value)
        return torch.stack(rows, dim=-2).reshape_as(x)
    raise ValueError(f"unsupported Hadamard width: {width}")


def cross_head_hadamard(attention: torch.Tensor) -> torch.Tensor:
    """Apply H32 across Llama query heads; the V projection already contains H128."""
    if attention.shape[-2:] != (32, 128):
        raise ValueError("attention must end in [32, 128]")
    return _butterfly_last_dim(attention.transpose(-1, -2)).transpose(-1, -2)
