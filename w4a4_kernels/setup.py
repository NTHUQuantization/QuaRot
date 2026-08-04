import os
import shlex

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


arches = [item for item in os.environ.get("W4A4_HIP_ARCHS", "gfx1201").replace(",", ";").split(";") if item]
os.environ.setdefault("PYTORCH_ROCM_ARCH", ";".join(arches))
flags = [
    "-O3",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
    # torch's default HIP extension macros disable conversions required by the
    # public rocWMMA headers. Undefine them only for this translation unit.
    "-U__HIP_NO_HALF_CONVERSIONS__",
    "-U__HIP_NO_HALF_OPERATORS__",
]
flags += [f"--offload-arch={arch}" for arch in arches]
flags += shlex.split(os.environ.get("W4A4_HIP_EXTRA_FLAGS", ""))

setup(
    name="w4a4_kernels_hip",
    ext_modules=[
        CUDAExtension(
            name="w4a4_kernels_hip",
            sources=["w4a4_kernels.hip"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
