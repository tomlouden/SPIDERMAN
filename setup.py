from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_test", ["_test.c", "test.c"]),Extension("_web", ["_web.c","heron.c","segment.c","areas.c","intersection.c","generate.c","blocked.c","util.c","pyutil.c","main.c","orthographic.c","ephemeris.c"]),Extension("_polar", ["_polar.c","polar.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)