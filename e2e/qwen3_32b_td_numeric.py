"""Numerically safe selected-feature restoration for the TD32 experiments."""
from __future__ import annotations

import torch

from e2e.speculative import SelectedHiddenCollector, _inverse_hadamard_cuda


class SafeSelectedHiddenCollector(SelectedHiddenCollector):
    """Avoid FP16 FHT intermediate overflow without changing the linear map.

    A power-of-two prescale is exact in FP16 exponent arithmetic. The inverse
    HadK result is promoted to FP32 before undoing the scale, so the observed
    one-element +Inf in each non-final 32B tap cannot propagate into affine
    statistics or the PARD-2 projection.
    """

    prescale = 16.0

    def features(self, rows=None):
        if self.folded_basis or self.rotation_signs is None:
            return super().features(rows)
        missing = [index for index in self.indices if index not in self.values]
        if missing:
            raise RuntimeError(f"selected hidden taps were not produced: {missing}")
        values = [self.values[index] for index in self.indices]
        if rows is not None:
            values = [value[:, rows] for value in values]
        signs = (
            self.rotation_signs.float()
            if self.basis_cached
            else self.rotation_signs.to(device=values[0].device).float()
        )
        inverse_kwargs = (
            {"had_k": self.inverse_had_k, "remainder": self.inverse_remainder}
            if self.basis_cached
            else {}
        )
        restored = [
            _inverse_hadamard_cuda(value / self.prescale, **inverse_kwargs).float()
            * self.prescale
            * signs
            for value in values
        ]
        final = restored[0]
        final = final * torch.rsqrt(
            final.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
        )
        norm_weight = (
            self.final_norm_weight.float()
            if self.basis_cached
            else self.final_norm_weight.to(device=final.device).float()
        )
        final = final * norm_weight
        features = torch.cat([final, *restored[1:]], dim=-1)
        if not torch.isfinite(features).all():
            raise RuntimeError("safe selected-feature restoration is still non-finite")
        return features


def install_safe_collector():
    import e2e.speculative as speculative

    speculative.SelectedHiddenCollector = SafeSelectedHiddenCollector
