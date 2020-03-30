from setuptools import setup

use README as the long description
make sure to use the syntax that works in both ReST and markdown
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type='text/x-rst'
)

# setup(
# 	long_description=
# 	"""
# 	When a plane wave encounters a perfect sphere then some of the light will
# 	be absorbed and some will be scattered.  Mie developed the equations that
# 	describe the scattered light wave.  These equations are complicated and 
# 	involve infinite sums of Bessel functions.  Not surprisingly, calculating
# 	the scattered profiles is complicated and easily gotten wrong.  Fortunately
# 	Wiscombe identified the challenges and implemented these in Fortran code.
# 	
# 	This code is pure python and uses many of the ideas that Wiscombe developed
# 	and published in papers and in Fortran code.  This code has been validated
# 	against that of Wiscombe.
# 	
# 	This code provides python functions for calculating the extinction efficiency,
# 	scattering efficiency, backscattering, and scattering asymmetry.  Moreover, a
# 	set of angles can be given and the scattering will be calculated for each angle.
# 	""",
# )
