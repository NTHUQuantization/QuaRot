"""Native HIP kernels for the packed W4A4 runtime."""

try:
    from . import w4a4_kernels_hip
except ImportError:
    w4a4_kernels_hip = None

__all__ = ["w4a4_kernels_hip"]
