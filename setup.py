"""
Build script: python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
import pybind11
import sys
import os

def is_msvc():
    if sys.platform == "win32":
        cc  = os.environ.get("CC",  "").lower()
        cxx = os.environ.get("CXX", "").lower()
        if any(tok in cc+cxx for tok in ("gcc","g++","mingw","clang")):
            return False
        return True
    return False

if is_msvc():
    extra_compile = ["/O2", "/std:c++17", "/GL", "/openmp"]
    extra_link    = ["/LTCG"]
elif sys.platform == "darwin":
    extra_compile = ["-O3","-march=native","-std=c++17","-ffast-math","-Xpreprocessor","-fopenmp"]
    extra_link    = ["-lomp"]
else:
    extra_compile = ["-O3","-march=native","-std=c++17","-ffast-math","-fopenmp"]
    extra_link    = ["-fopenmp"]

ext = Extension(
    "fractal_core",
    sources=["src/fractal_engine.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
    language="c++",
)

setup(
    name="fractal_core",
    version="3.0.0",
    description="Fractal Engine – C++ core via pybind11",
    ext_modules=[ext],
    python_requires=">=3.9",
)
