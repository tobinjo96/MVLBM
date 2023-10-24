# from distutils.core import setup, Extension
# import numpy
# from Cython.Distutils import build_ext
# 
# 
# setup(
#     name='Bos_utils',
#     version='1.3',
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[Extension("Bos_utils",
#                  sources=["Bos_utils.pyx"],
#                  language="c++",
#                  include_dirs=[numpy.get_include()])]
# 
# )

from distutils.core import setup, Extension
import numpy
#om setuptools import setup
from Cython.Build import cythonize

extensions = [Extension("Bos_utils",
                  sources=["Bos_utils.pyx"],
                  language="c++",
                  include_dirs=[numpy.get_include()])]


setup(ext_modules = cythonize(extensions))
