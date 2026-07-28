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

Submodules
----------

``miepython.rayleigh`` is imported with the package and needs nothing beyond NumPy.
It mirrors the API above using the small-sphere limit, which is handy for checking
the full solution as x goes to zero::

    mie.rayleigh.efficiencies_mx(1.5, 0.01)

``miepython.vsh`` holds the vector spherical harmonics as a direct transcription of
the textbook formulas.  It is a reference for development and cross-checking rather
than a fast path, it requires the optional ``scipy`` dependency, and so it is bound
only when scipy is installed::

    mie.vsh.M_odd(1, k, d_sphere, r, theta, phi)

``miepython.monte_carlo`` samples scattering angles from the Mie phase function.  It
imports this package, so importing it from here would be circular; ask for it
explicitly instead::

    import miepython.monte_carlo as mc
    mu, cdf = mc.mu_with_uniform_cdf(m, x, 100)

Near-field calculations are provided by the ``miepython.field`` module::

    from miepython.field import e_near, h_near, eh_near
    from miepython.field import e_near_cartesian, h_near_cartesian, eh_near_cartesian
    from miepython.field import e_far

Example near-field usage::

    import numpy as np
    from miepython.field import eh_near_cartesian

    u = np.linspace(-1.5, 1.5, 101)
    X, Z = np.meshgrid(u, u, indexing='xy')
    E_xyz, H_xyz = eh_near_cartesian(
        lambda0=1.0, d_sphere=1.0, m_sphere=1.5 + 0.0j, n_env=1.0,
        x=X, y=np.zeros_like(X), z=Z
    )

"""

from ._backend import D_calc, USE_JIT, _D_downwards, _D_upwards, _Lentz_Dn, _S1_S2, an_bn, cn_dn, pi_tau
from ._backend import single_sphere, small_conducting_sphere, small_sphere

from .core import efficiencies, intensities, i_par, i_per, i_unpolarized
from .core import efficiencies_mx, S1_S2, phase_matrix, coefficients

from . import rayleigh

_FIELD_IMPORT_ERROR = None


def _raise_field_import_error():
    """Raise a consistent error when optional near-field dependencies are missing."""
    raise ModuleNotFoundError(
        "Near-field functions require the optional dependency 'scipy'. "
        "Install miepython with SciPy support to use e_far/e_near/h_near/eh_near."
    ) from _FIELD_IMPORT_ERROR


def _missing_field_function(name):
    """Create a placeholder near-field function when SciPy is unavailable."""

    def _missing(*_args, **_kwargs):
        _raise_field_import_error()

    _missing.__name__ = name
    _missing.__qualname__ = name
    _missing.__doc__ = f"{name} requires the optional dependency 'scipy'."
    return _missing


try:
    from . import field as _field
    from . import vsh  # noqa: F401  reachable as miepython.vsh; needs scipy, so guarded here
except ModuleNotFoundError as exc:
    missing_module = str(getattr(exc, "name", "") or "")
    if missing_module.startswith("scipy"):
        _FIELD_IMPORT_ERROR = exc
        # ``miepython.vsh`` is left unbound rather than stubbed out.  It is a
        # reference implementation used for development and cross-checking, not part
        # of the public surface, and every one of its functions needs scipy.
        e_far = _missing_field_function("e_far")
        e_near = _missing_field_function("e_near")
        h_near = _missing_field_function("h_near")
        eh_near = _missing_field_function("eh_near")
        e_near_cartesian = _missing_field_function("e_near_cartesian")
        h_near_cartesian = _missing_field_function("h_near_cartesian")
        eh_near_cartesian = _missing_field_function("eh_near_cartesian")
    else:
        raise
else:
    e_far = _field.e_far
    e_near = _field.e_near
    h_near = _field.h_near
    eh_near = _field.eh_near
    e_near_cartesian = _field.e_near_cartesian
    h_near_cartesian = _field.h_near_cartesian
    eh_near_cartesian = _field.eh_near_cartesian

# Names exposed to the user.  ``vsh`` is deliberately absent: it is bound only when
# scipy is installed, and a star-import must not depend on that.
__all__ = (
    "rayleigh",
    "intensities",
    "i_par",
    "i_per",
    "i_unpolarized",
    "phase_matrix",
    "coefficients",
    "efficiencies",
    "efficiencies_mx",
    "e_far",
    "e_near",
    "h_near",
    "eh_near",
    "e_near_cartesian",
    "h_near_cartesian",
    "eh_near_cartesian",
    "an_bn",
    "cn_dn",
    "S1_S2",
    "single_sphere",
    "small_sphere",
    "small_conducting_sphere",
    "USE_JIT",
    "D_calc",
    "pi_tau",
    "_S1_S2",
    "_Lentz_Dn",
    "_D_upwards",
    "_D_downwards",
)

__version__ = "3.3.0"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2017-2026, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/miepython.git"
