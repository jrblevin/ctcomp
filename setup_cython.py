"""
Setup script for building Cython extensions.

Run with:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Use -march=native for local builds, but avoid it in CI environments
compile_args = ["-O3", "-ffast-math"]
if not any(ci_var in os.environ for ci_var in ["CI", "GITHUB_ACTIONS", "TRAVIS", "CIRCLECI"]):
    compile_args.append("-march=native")

extensions = [
    Extension(
        "model_cython",
        ["model_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'embedsignature': True,
        },
        annotate=True  # Generates HTML file showing optimization
    )
)
