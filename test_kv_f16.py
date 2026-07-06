import torch
import flashinfer_test

assert torch.cuda.is_available()

device = "cuda"

batch_size = 1
num_layers = 1
num_heads = 2
head_dim = 128
page_size = 16
seq_len = 8
page_cnt = 1

k = torch.arange(
    seq_len * num_heads * head_dim,
    dtype=torch.float16,
    device=device,
).reshape(seq_len, num_heads, head_dim)

v = k + 10000

k_param = torch.ones(
    (seq_len, num_heads, 2),
    dtype=torch.float16,
    device=device,
)

v_param = torch.ones(
    (seq_len, num_heads, 2),
    dtype=torch.float16,
    device=device,
) * 2

kv_data = torch.zeros(
    (
        page_cnt,
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
        page_cnt,
        num_layers,
        2,
        num_heads,
        page_size,
        2,
    ),
    dtype=torch.float16,
    device=device,
)

kv_indptr = torch.tensor(
    [0, 1],
    dtype=torch.int32,
    device=device,
)

kv_indices = torch.tensor(
    [0],
    dtype=torch.int32,
    device=device,
)

last_page_offset = torch.tensor(
    [8],
    dtype=torch.int32,
    device=device,
)

seqlen_indptr = torch.tensor(
    [0, 8],
    dtype=torch.int32,
    device=device,
)

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

append_k = torch.full(
    (batch_size, num_heads, head_dim),
    999,
    dtype=torch.float16,
    device=device,
)

append_v = torch.full(
    (batch_size, num_heads, head_dim),
    888,
    dtype=torch.float16,
    device=device,
)

append_k_param = torch.ones(
    (batch_size, num_heads, 2),
    dtype=torch.float16,
    device=device,
) * 3

append_v_param = torch.ones(
    (batch_size, num_heads, 2),
    dtype=torch.float16,
    device=device,
) * 4

last_page_offset = torch.tensor(
    [9],
    dtype=torch.int32,
    device=device,
)

kv_indptr = torch.tensor([0,1], dtype=torch.int32, device=device)
kv_indices = torch.tensor([0], dtype=torch.int32, device=device)

flashinfer_test.append_kv_f16(
    kv_data,
    kv_param,
    kv_indptr,
    kv_indices,
    last_page_offset,
    append_k,
    append_v,
    append_k_param,
    append_v_param,
    0,
)

torch.cuda.synchronize()

q = torch.ones(
    (batch_size, num_heads, head_dim),
    dtype=torch.float16,
    device=device,
)
o = torch.zeros_like(q)

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

print("output has nan:", torch.isnan(o).any())
print("output has inf:", torch.isinf(o).any())
print(o.abs().sum())
print(o[0,0,:16])
print(o[0,1,:16])