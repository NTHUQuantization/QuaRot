from setuptools import setup, find_packages
import os

os.makedirs("flashinfer_test", exist_ok=True)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup_dir = os.path.dirname(os.path.realpath(__file__))


setup(
    name="flashinfer_test",

    ext_modules=[
        CUDAExtension(
            name="flashinfer_test._HIP",

            sources=[
                "flashinfer.hip",
                "binding.cpp"
            ],

            include_dirs=[
                os.path.join(
                    setup_dir,
                    "include_hip"
                ),
            ],

            extra_compile_args={
                "cxx": [
                    "-O3",
                ],

                "hipcc": [
                    "-O3",

                    # 你的 GPU
                    "--offload-arch=gfx1201",

                    # HIPify後避免某些 CUDA header
                    "-D__HIP_PLATFORM_AMD__",
                ],
            },
        )
    ],

    cmdclass={
        "build_ext": BuildExtension
    }
)