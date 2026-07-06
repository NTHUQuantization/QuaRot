import torch
import flashinfer_test

assert torch.cuda.is_available()

device = "cuda"

# -----------------------
# configuration
# -----------------------
batch_size = 1
num_layers = 1
num_heads = 2
head_dim = 128
page_size = 16
seq_len = 8

page_cnt = (seq_len + page_size - 1) // page_size

# -----------------------
# KV cache
# -----------------------
kv_data = torch.zeros(
    (
        page_cnt * batch_size,
        num_layers,
        2,
        num_heads,
        page_size,
        head_dim,
    ),
    dtype=torch.float16,
    device=device,
)

kv_param = torch.zeros(
    (
        page_cnt * batch_size,
        num_layers,
        2,
        num_heads,
        page_size,
        2,
    ),
    dtype=torch.float16,
    device=device,
)

# -----------------------
# page table
# -----------------------
kv_indptr = torch.arange(
    0,
    batch_size + 1,
    device=device,
    dtype=torch.int32,
) * page_cnt
print("kv_indptr:")
print(kv_indptr)

kv_indices = (
    (
        torch.arange(page_cnt, device=device)
        * batch_size
    ).unsqueeze(0)
    + torch.arange(batch_size, device=device).unsqueeze(1)
).reshape(-1).to(torch.int32)
print("kv_indices:")
print(kv_indices)

last_page_offset = torch.tensor(
    [seq_len],
    dtype=torch.int32,
    device=device,
)
print("last_page_offset:")
print(last_page_offset)

# -----------------------
# Initial KV
# -----------------------

k = torch.randn(
    seq_len,
    num_heads,
    head_dim,
    device=device,
    dtype=torch.float16,
)

v = torch.randn_like(k)

k_param = torch.zeros(
    seq_len,
    num_heads,
    2,
    device=device,
    dtype=torch.float16,
)

v_param = torch.zeros_like(k_param)

seqlen_indptr = torch.tensor(
    [0, seq_len],
    dtype=torch.int32,
    device=device,
)

print("=== init_kv_f16 ===")

flashinfer_test.init_kv_f16(
    kv_data,
    kv_param,
    kv_indptr,
    kv_indices,
    last_page_offset,
    k,
    v,
    k_param,
    v_param,
    seqlen_indptr,
    0,
)

torch.cuda.synchronize()

# -----------------------
# append one token
# -----------------------

k2 = torch.randn(
    batch_size,
    num_heads,
    head_dim,
    device=device,
    dtype=torch.float16,
)

v2 = torch.randn_like(k2)

k2_param = torch.zeros(
    batch_size,
    num_heads,
    2,
    device=device,
    dtype=torch.float16,
)

v2_param = torch.zeros_like(k2_param)

print("=== append_kv_f16 ===")

flashinfer_test.append_kv_f16(
    kv_data,
    kv_param,
    kv_indptr,
    kv_indices,
    last_page_offset,
    k2,
    v2,
    k2_param,
    v2_param,
    0,
)

torch.cuda.synchronize()

print("append_kv_f16 OK")

# -----------------------
# decode
# -----------------------

q = torch.randn(
    batch_size,
    num_heads,
    head_dim,
    device=device,
    dtype=torch.float16,
)

o = torch.empty_like(q)

print("=== batch_decode_f16 ===")

flashinfer_test.batch_decode_f16(
    o,
    q,
    kv_data,
    kv_param,
    kv_indptr,
    kv_indices,
    last_page_offset,
    0,
)

torch.cuda.synchronize()

print("batch_decode_f16 OK")
print(o)