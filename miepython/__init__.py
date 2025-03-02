"""
Mie scattering calculations for perfect spheres.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-absorbing, partially-absorbing, or perfectly conducting
spheres.

The extinction efficiency, scattering efficiency, backscattering, and
scattering asymmetry for a sphere with complex index of refraction m,
diameter d, and wavelength lambda can be found by::

    import miepython as mie
    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0)

The normalized scattering values for angles mu=cos(theta) are::

    Ipar, Iper = mie.intensities(m, d, lambda0, mu)

If the size parameter is known, then use::

    mie.efficiencies_mx(m, x)

Mie scattering amplitudes S1 and S2 (complex numbers):

    mie.S1_S2(m, x, mu)

Normalized Mie scattering intensities for angles mu=cos(theta)::

    mie.i_per(m, x, mu)
    mie.i_par(m, x, mu)
    mie.i_unpolarized(m, x, mu)

Mie scattering intensities normalized to one when integrated over all angles::

    mie.i_per(m, x, mu, norm='one')
    mie.i_par(m, x, mu, norm='one')
    mie.i_unpolarized(m, x, mu, norm='one')

The scattering matrix

    mie.phase_matrix(m, x, mu)

"""

import os

USE_JIT = os.environ.get("MIEPYTHON_USE_JIT", "1").lower() == "1"

if USE_JIT:
    from .mie_jit import _an_bn, _cn_dn, _S1_S2, _D_calc, _pi_tau

else:
    from .mie_nojit import _an_bn, _cn_dn, _S1_S2, _D_calc, _pi_tau

from .core import efficiencies, intensities, i_par, i_per, i_unpolarized, normalization_factor
from .core import efficiencies_mx, S1_S2, phase_matrix, coefficients, an_bn, cn_dn

# The only functions exposed to the user
__all__ = (
    "intensities",
    "i_par",
    "i_per",
    "i_unpolarized",
    "phase_matrix",
    "coefficients",
    "efficiencies",
    "efficiencies_mx",
    "an_bn",
    "cn_dn",
    "S1_S2",
    "_an_bn",
    "_cn_dn",
    "_S1_S2",
    "_D_calc",
    "_pi_tau",
    "normalization_factor",
)

__version__ = "3.0.0"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2017-25, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/miepython.git"
