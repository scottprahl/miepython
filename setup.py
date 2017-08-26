"""
Copyright 2017 Scott Prahl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from setuptools import setup

setup(
	name='miepython',
	packages=['miepython'],
	version='0.4.0',
	description='Mie scattering of a plane wave by a sphere',
	url='https://github.com/scottprahl/miepython.git',  
	author='Scott Prahl',
	author_email='scott.prahl@oit.edu',
	license='MIT',
	classifiers=[
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: MIT License',
		'Intended Audience :: Science/Research',
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering :: Physics',
	],
	keywords=['mie', 'scattering', 'rainbow', 'droplet'],
	install_requires=['numpy'],
	test_suite='miepython.test.test',
	long_description=
	"""
	When a plane wave encounters a perfect sphere then some of the light will
	be absorbed and some will be scattered.  Mie developed the equations that
	describe the scattered light wave.  These equations are complicated and 
	involve infinite sums of Bessel functions.  Not surprisingly, calculating
	the scattered profiles is complicated and easily gotten wrong.  Fortunately
	Wiscombe identified the challenges and implemented these in Fortran code.
	
	This code is pure python and uses many of the ideas that Wiscombe developed.
	However, it is *not* as accurate (especially for large spheres with size parameters
	larger than 100).
	
	This code provides python functions for calculating the extinction efficiency,
	scattering efficiency, backscattering, and scattering asymmetry.  Moreover, a
	set of angles can be given and the scattering will be calculated for each angle.
	""",
)