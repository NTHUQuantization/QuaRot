"""Fit Phase-2 per-channel affine calibration from a frozen tune split.

Input is a torch file containing paired 2-D/3-D tensors named
``fused_raw``, ``reference_raw``, ``fused_projected`` and
``reference_projected``.  Formal-split tensors are intentionally rejected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def fit_affine(source, reference):
    if source.shape != reference.shape or source.ndim < 2:
        raise ValueError("paired calibration tensors must have equal [..., channel] shape")
    x = source.float().reshape(-1, source.shape[-1])
    y = reference.float().reshape(-1, reference.shape[-1])
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered = x - mean_x
    variance = centered.square().mean(0)
    covariance = (centered * (y - mean_y)).mean(0)
    scale = torch.where(variance > 1e-12, covariance / variance,
                        torch.ones_like(variance))
    bias = mean_y - scale * mean_x
    return scale.half(), bias.half()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = torch.load(args.tune_features, map_location="cpu", weights_only=True)
    if payload.get("split") != "tune":
        raise ValueError("calibration is permitted only on a file marked split='tune'")
    raw_scale, raw_bias = fit_affine(payload["fused_raw"], payload["reference_raw"])
    projected_scale, projected_bias = fit_affine(
        payload["fused_projected"], payload["reference_projected"])
    output = {"raw_scale": raw_scale, "raw_bias": raw_bias,
              "projected_scale": projected_scale,
              "projected_bias": projected_bias,
              "source_split": "tune", "runtime": "fused_v1"}
    torch.save(output, Path(args.output))


if __name__ == "__main__":
    main()
