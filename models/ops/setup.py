import os
import glob
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    # Default to CPU extension
    sources = main_file + source_cpu
    extension = CppExtension
    define_macros = []

    if torch.cuda.is_available():
        from torch.utils.cpp_extension import CUDAExtension
        from torch.utils.cpp_extension import CUDA_HOME

        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

        # Switch to CUDA extension if CUDA is available
        sources += source_cuda
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args = {"cxx": [], "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]}
    else:
        print('CUDA is not available. Compiling for CPU...')
        extra_compile_args = {}

    # Adjust paths
    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
