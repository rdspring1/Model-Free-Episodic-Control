from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "clsh",                                # the extension name
           sources=["clsh.pyx", "LSH.cpp"], # the Cython source and additional C++ source files
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11"]
      )))
