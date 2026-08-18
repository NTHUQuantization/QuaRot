import quarot
import torch

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=0.9):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (
            x.abs().amax(dim=-1, keepdim=True).div(7)
            * self.input_clip_ratio
        ).to(torch.float16)
        scales_x.clamp_min_(torch.finfo(torch.float16).tiny)
        quantized_x = quarot.sym_quant(x, scales_x)
        packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor
