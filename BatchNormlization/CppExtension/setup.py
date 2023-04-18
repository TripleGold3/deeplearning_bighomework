from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='MyBatchNorm1D_CPP',
    ext_modules=[
        CppExtension('MyBatchNorm1D_CPP', ['MyBatchNorm1D_CPP.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })