"""
Copyright 2017 Scott Prahl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from setuptools import setup, find_packages

setup(
	name='miepython',
	version='0.3.0',
	description='Mie scattering from spheres',
	long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
	url='http://omlc.org/software/mie/',  
	author='Scott Prahl',
	author_email='scott.prahl@oit.edu',
	license='MIT',
	classifiers=[
		'Development Status :: 4 - Beta',
		'License :: OSI Approved :: MIT License',
		'Intended Audience :: Science/Research'
		'Programming Language :: Python :: 3.5',
		'Topic :: Text Processing :: Linguistic',
	],
	keywords='mie scattering rainbow droplet',
	packages=find_packages(),
	install_requires=['numpy'],
	test_suite='nose.collector',
	tests_require=['nose', 'nose-cover3'],
	include_package_data=True,
	zip_safe=False
)