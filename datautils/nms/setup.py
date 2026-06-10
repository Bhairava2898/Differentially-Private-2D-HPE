# PROJECT_2/scripts/datautils/nms/setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        ["cpu_nms.pyx"],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
