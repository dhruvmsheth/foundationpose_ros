# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or

from setuptools import setup
import os, sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This extension requires CUDA.")

if CUDA_HOME is None:
    raise RuntimeError("CUDA_HOME environment variable is not set. Please set it to your CUDA installation directory.")

code_dir = os.path.dirname(os.path.realpath(__file__))

nvcc_flags = ['-Xcompiler', '-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '--expt-relaxed-constexpr', '-arch=sm_87']
c_flags = ['-O3', '-std=c++17']

setup(
    name='common',
    ext_modules=[
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ], extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ], extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags}),
    ],
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
