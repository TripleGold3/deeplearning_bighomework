from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MyBatchNorm1D_cuda',
    ext_modules=[
        CUDAExtension('MyBatchNorm1D_cuda', [
            'MyBatchNorm1D_cuda.cpp',
            'MyBatchNorm1D_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
