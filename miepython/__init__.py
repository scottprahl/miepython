"""
Mie scattering calculations for perfect spheres.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-absorbing, partially-absorbing, or perfectly conducting
spheres.

The extinction efficiency, scattering efficiency, backscattering, and
scattering asymmetry for a sphere with complex index of refraction m,
diameter d, and wavelength lambda can be found by::

    qext, qsca, qback, g = miepython.ez_mie(m, d, lambda0)

The normalized scattering values for angles mu=cos(theta) are::

    Ipar, Iper = miepython.ez_intensities(m, d, lambda0, mu)

If the size parameter is known, then use::

    miepython.mie(m, x)

Mie scattering amplitudes S1 and S2 (complex numbers):

    miepython.mie_S1_S2(m, x, mu)

Normalized Mie scattering intensities for angles mu=cos(theta)::

    miepython.i_per(m, x, mu)
    miepython.i_par(m, x, mu)
    miepython.i_unpolarized(m, x, mu)

Mie scattering intensities normalized to one when integrated over all angles::

    miepython.i_per(m, x, mu, norm='one')
    miepython.i_par(m, x, mu, norm='one')
    miepython.i_unpolarized(m, x, mu, norm='one')

The scattering matrix

    miepython.mie_phase_matrix(m, x, mu)

"""
__version__ = '2.4.0'
__author__ = 'Scott Prahl'
__email__ = 'scott.prahl@oit.edu'
__copyright__ = 'Copyright 2017-23, Scott Prahl'
__license__ = 'MIT'
__url__ = 'https://github.com/scottprahl/miepython.git'

from .miepython import *
