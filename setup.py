from setuptools import Extension, setup

import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize

args = ["-Wall", "-O3"]

# python setup.py build_ext --inplace

ext_modules = [Extension("hausdorff_dist", ["dist.pyx"],
                         include_dirs=[np.get_include()],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                         )]

setup(name="hausdorff_dist", version="0.0.1", author="Adam Spannaus",
      cmdclass={"build_ext": build_ext},
      include_dirs=[np.get_include()],
      ext_modules=cythonize(ext_modules))
