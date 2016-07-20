from distutils.core import setup, Extension
import numpy.distutils.misc_util
import numpy as np

_test = Extension("spiderman._test", ["c_src/_test.c", "c_src/test.c"])
_web = Extension("spiderman._web",["c_src/_web.c","c_src/heron.c","c_src/segment.c","c_src/areas.c","c_src/intersection.c","c_src/generate.c","c_src/blocked.c","c_src/util.c","c_src/pyutil.c","c_src/main.c","c_src/orthographic.c","c_src/ephemeris.c"])
_polar = Extension("spiderman._polar", ["c_src/_polar.c","c_src/polar.c"])

setup(	name='spiderman-package', 
	version="0.1.0",
	author='Tom Louden',
	author_email = 't.m.louden@warwick.ac.uk',
	url = 'https://github.com/tmlouden/spiderman',
	packages =['spiderman'],
	license = ['GNU GPLv3'],
	description ='Fast secondary eclipse and phase curve modeling',
	classifiers = [
		'Development Status :: 5 - Beta/Stable',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering',
		'Programming Language :: Python'
		],
	include_dirs = [np.get_include()],
	install_requires = ['numpy'],
	ext_modules=[_test,_web,_polar]
)
