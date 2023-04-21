from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='MyBatchNorm2D_CPP',
    ext_modules=[
        CppExtension('MyBatchNorm2D_CPP', ['MyBatchNorm2D_CPP.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

  
  
   