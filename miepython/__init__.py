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

USE_JIT = os.environ.get("MIEPYTHON_USE_JIT", "0") == "1"

if USE_JIT:
    # pull in the jit‐compiled backends under the "private" names core.py expects
    from .mie_jit import _an_bn_nb as an_bn
    from .mie_jit import _cn_dn_nb as cn_dn
    from .mie_jit import _single_sphere_nb as single_sphere
    from .mie_jit import _small_sphere_nb as small_sphere
    from .mie_jit import _small_conducting_sphere_nb as small_conducting_sphere
    from .mie_jit import _pi_tau_nb as _pi_tau
    from .mie_jit import _D_calc_nb as _D_calc
    from .mie_jit import _S1_S2_nb as _S1_S2
    from .mie_jit import _Lentz_Dn
    from .mie_jit import _D_upwards
    from .mie_jit import _D_downwards
else:
    from .mie_nojit import _an_bn_py as an_bn
    from .mie_nojit import _cn_dn_py as cn_dn
    from .mie_nojit import _single_sphere_py as single_sphere
    from .mie_nojit import _small_sphere_py as small_sphere
    from .mie_nojit import _small_conducting_sphere_py as small_conducting_sphere
    from .mie_nojit import _pi_tau_py as _pi_tau
    from .mie_nojit import _D_calc_py as _D_calc
    from .mie_nojit import _S1_S2_py as _S1_S2
    from .mie_nojit import _Lentz_Dn
    from .mie_nojit import _D_upwards
    from .mie_nojit import _D_downwards

from .core import efficiencies, intensities, i_par, i_per, i_unpolarized
from .core import efficiencies_mx, S1_S2, phase_matrix, coefficients

# functions exposed to the user
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
    "single_sphere",
    "small_sphere",
    "small_conducting_sphere",
    "_S1_S2",
    "_D_calc",
    "_pi_tau",
    "_Lentz_Dn",
    "_D_upwards",
    "_D_downwards",
)

__version__ = "3.0.3"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2017-26, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/miepython.git"
