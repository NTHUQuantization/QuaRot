import os
import shlex

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def split_arches(value):
    return [x.strip() for x in value.replace(",", ";").split(";") if x.strip()]


hip_arches = split_arches(os.environ.get("FFN_FUSION_HIP_ARCHS", "gfx1201"))
os.environ.setdefault("PYTORCH_ROCM_ARCH", ";".join(hip_arches))

hip_flags = [
    "-O3",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
]
hip_flags.extend(shlex.split(os.environ.get("FFN_FUSION_HIP_EXTRA_FLAGS", "")))
for arch in hip_arches:
    hip_flags.append(f"--offload-arch={arch}")

setup(
    name="ffn_fusion_hip",
    ext_modules=[
        CUDAExtension(
            name="ffn_fusion_hip",
            sources=["ffn_fusion.hip"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": hip_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
