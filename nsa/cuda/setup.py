"""Build for the NSA selected-branch CUDA extension. JIT via the
package __init__, or `python -m nsa.cuda.setup install` standalone.
Hopper sm_90a + CUDA 12.4 (inline-PTX wgmma; no CUTLASS dependency).
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = os.path.dirname(os.path.abspath(__file__))


def _extension():
    return CUDAExtension(
        name="nsa_cuda_selected",
        sources=[
            os.path.join(HERE, "selected_fwd.cu"),
            os.path.join(HERE, "selected_bwd.cu"),
            os.path.join(HERE, "bindings.cpp"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-arch=sm_90a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--ptxas-options=-v",
            ],
        },
    )


if __name__ == "__main__":
    setup(
        name="nsa_cuda_selected",
        ext_modules=[_extension()],
        cmdclass={"build_ext": BuildExtension},
    )
