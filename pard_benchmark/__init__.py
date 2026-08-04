"""Standalone PARD/PARD2 decode evaluation for Llama 3.1.

The speculative decoding algorithm is adapted from AMD-AGI/PARD commit
6f279bf3f1680e0b5d71c562ca5b91bdeef4c038 (MIT license).  The surrounding
configuration, measurement, isolation, and reporting code is local to this
repository.
"""

from .config import MODEL_SPECS, UPSTREAM_COMMIT
from .w4a4_engine import W4A4PardRuntime, load_w4a4_runtime

__all__ = ["MODEL_SPECS", "UPSTREAM_COMMIT", "W4A4PardRuntime", "load_w4a4_runtime"]
