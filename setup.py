import os

from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup_dir = os.path.dirname(os.path.realpath(__file__))

# Build for the project's primary target unless the caller requests others.
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get(
    "QUAROT_HIP_ARCHS", "gfx1201"
)

def remove_unwanted_pytorch_flags():
    flags = [
        "-D__HIP_NO_HALF_OPERATORS__=1",
        "-D__HIP_NO_HALF_CONVERSIONS__=1",
    ]

    for flag in flags:
        for flag_list in [
            torch_cpp_ext.COMMON_NVCC_FLAGS,
            torch_cpp_ext.COMMON_HIP_FLAGS,
        ]:
            try:
                while flag in flag_list:
                    flag_list.remove(flag)
            except AttributeError:
                pass


if __name__ == '__main__':
    remove_unwanted_pytorch_flags()
    setup(
        name='quarot',
        ext_modules=[
            # PyTorch uses CUDAExtension for both CUDA and ROCm/HIP sources.
            CUDAExtension(
                name='quarot._HIP',
                sources=[
                    'quarot/kernels/bindings.cpp',
                    'quarot/kernels/gemm.hip',
                    'quarot/kernels/quant.hip',
                    'quarot/kernels/flashinfer.hip',
                    'quarot/kernels/fused_hip.hip',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'quarot/kernels/include_hip'),
                ],
                extra_compile_args={
                    "cxx": [
                        "-O3",
                        "-std=c++17",
                    ],
                    "nvcc": [
                        "-O3",
                        # Host-side dispatch also needs this definition; HIP
                        # architecture macros are device-pass-only.
                        "-DQUAROT_BPRE_GFX12=1",
                        # PyTorch may add these flags after COMMON_HIP_FLAGS
                        # has been edited, so undefine them here as well.
                        "-U__HIP_NO_HALF_OPERATORS__",
                        "-U__HIP_NO_HALF_CONVERSIONS__",
                    ],
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
