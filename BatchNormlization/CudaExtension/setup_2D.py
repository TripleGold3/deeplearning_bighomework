from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MyBatchNorm2D_cuda',
    ext_modules=[
        CUDAExtension('MyBatchNorm2D_cuda', [
            'MyBatchNorm2D_cuda.cpp',
            'MyBatchNorm2D_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
