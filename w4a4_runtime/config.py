from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class W4A4Config:
    schema_version: int = SCHEMA_VERSION
    weight_bits: int = 4
    activation_bits: int = 4
    kv_bits: int = 4
    group_size: int = 128
    activation_clip_ratio: float = 0.9
    kv_clip_ratio: float = 0.95
    rotation_seed: int = 0
    weight_quant_method: str = "gptq"
    weight_sym: bool = True
    activation_sym: bool = True
    weight_layout: str = "n16_k128_twos_complement"
    activation_layout: str = "row_k_twos_complement"
    kv_layout: str = "flashinfer_biased_signed_nibble"
    model_type: str = "llama"
    head_dim: int = 128
    model_revision: str = "unknown"
    converter_commit: str = "unknown"
    calibration_dataset: str = "wikitext-2-raw-v1"
    calibration_samples: int = 128
    calibration_sequence_length: int = 2048
    gptq_damping: float = 0.01
    gptq_act_order: bool = False
    gptq_mse_clipping: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported W4A4 schema version: {self.schema_version}")
        if (self.weight_bits, self.activation_bits, self.kv_bits) != (4, 4, 4):
            raise ValueError("this runtime only supports W4A4KV4")
        if self.group_size != 128:
            raise ValueError("the first RDNA4 runtime requires group_size=128")
        if not (0.0 < self.activation_clip_ratio <= 1.0):
            raise ValueError("activation_clip_ratio must be in (0, 1]")
        if not (0.0 < self.kv_clip_ratio <= 1.0):
            raise ValueError("kv_clip_ratio must be in (0, 1]")
        if not self.weight_sym or not self.activation_sym:
            raise ValueError("linear operands must use signed symmetric INT4")
        if self.head_dim != 128:
            raise ValueError("the first runtime only supports head_dim=128")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W4A4Config":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in value.items() if key in known})
