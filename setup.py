from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import pathlib
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

# ROCm/PyTorch selects the installed target by default.  Set
# QUAROT_HIP_ARCHS (for example "gfx1100;gfx1201") only for cross-compiles.
if os.environ.get("QUAROT_HIP_ARCHS"):
    os.environ["PYTORCH_ROCM_ARCH"] = os.environ["QUAROT_HIP_ARCHS"]

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


def require_complete_ck_checkout():
    """Fail early when the vendored CK tree contains placeholder files."""
    ck_root = HERE / 'third-party/rocm-libraries/projects/composablekernel'
    required_headers = [
        ck_root / 'include/ck/ck.hpp',
        ck_root / 'include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp',
    ]
    missing = [str(path) for path in required_headers if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            'Composable Kernel checkout is incomplete (missing or empty headers):\n  - ' +
            '\n  - '.join(missing) +
            '\nRepopulate third-party/rocm-libraries before building this extension.'
        )

# def third_party_cmake():
#     import subprocess, sys, shutil
    
#     cmake = shutil.which('cmake')
#     if cmake is None: 
#             raise RuntimeError('Cannot find CMake executable.')

#     retcode = subprocess.call([cmake, HERE])
#     if retcode != 0:
#         sys.stderr.write("Error: CMake configuration failed.\n")
#         sys.exit(1)

    # install fast hadamard transform
    # hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    # pip = shutil.which('pip')
    # retcode = subprocess.call([pip, 'install', '-e', hadamard_dir])

if __name__ == '__main__':
    # third_party_cmake()
    remove_unwanted_pytorch_flags()
    require_complete_ck_checkout()
    setup(
        name='quarot',
        ext_modules=[
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
                    os.path.join(setup_dir, 'third-party/rocm-libraries/projects/composablekernel/include'),
                    os.path.join(setup_dir, 'third-party/rocm-libraries/projects/composablekernel/library/include'),
                ],
                # library_dirs=[
                #     os.path.join(setup_dir, "third-party/rocm-libraries/projects/composablekernel/build/lib"),
                # ],
                # libraries=[
                #     "device_gemm_operations",
                #     "utility",
                # ],
                extra_compile_args={
                    "cxx": [
                        "-O3",
                        "-std=c++17",
                    ],
                    "nvcc": [
                        "-O3",
                        "--offload-arch=gfx1201",
                        "-DQUAROT_BPRE_GFX12=1",
                        # PyTorch may add these after COMMON_HIP_FLAGS has
                        # been edited.  CK's half helpers require the HIP
                        # conversions, so undefine them at the end as well.
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
