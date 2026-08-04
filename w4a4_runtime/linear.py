import torch

from .ops import embedding_lookup_w4, quantize_activation, w4a4_linear
from .packing import PackedWeight


class W4A4Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        bias: bool = False,
        device=None,
    ):
        super().__init__()
        if in_features % group_size or out_features % 16:
            raise ValueError("W4A4Linear requires K%128==0 and N%16==0")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "qweight",
            torch.empty(
                out_features // 16,
                in_features // group_size,
                16,
                group_size // 2,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "scales", torch.empty(out_features, in_features // group_size, dtype=torch.float16, device=device)
        )
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=torch.float16, device=device) if bias else None
        )

    def packed_weight(self) -> PackedWeight:
        return PackedWeight(self.qweight, self.scales, self.out_features, self.in_features, self.group_size)

    def forward(self, x, *, input_clip_ratio: float = 0.9):
        packed = x if hasattr(x, "qdata") else quantize_activation(
            x, group_size=self.group_size, clip_ratio=input_clip_ratio
        )
        out = w4a4_linear(packed, self.packed_weight(), self.bias)
        return out.reshape(*x.shape[:-1], self.out_features) if isinstance(x, torch.Tensor) else out


class W4Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *, group_size: int = 128, device=None):
        super().__init__()
        if embedding_dim % group_size:
            raise ValueError("embedding_dim must be divisible by group_size")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.register_buffer(
            "qweight", torch.empty(num_embeddings, embedding_dim // 2, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "scales",
            torch.empty(num_embeddings, embedding_dim // group_size, dtype=torch.float16, device=device),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return embedding_lookup_w4(token_ids, self.qweight, self.scales, group_size=self.group_size)
