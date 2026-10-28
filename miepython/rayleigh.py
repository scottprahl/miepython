"""
Rayleigh approximation for scattering by spheres much smaller than the wavelength.

When a sphere is small both outside (x << 1) and inside (abs(m) * x << 1), the
incident field is nearly uniform across it, and it radiates as a single
oscillating electric dipole. Every scattering quantity then follows in closed
form from the ratio (m**2 - 1) / (m**2 + 2), with no series of multipoles to
sum. This module evaluates those expressions, following Bohren and Huffman,
equations 5.4 and 5.7 to 5.9. A perfect conductor also radiates a magnetic
dipole, and has its own expressions, described below.

The functions mirror those of the same names in ``miepython``, with the same
arguments and the same ``norm`` choices, so one can stand in for the other:

``efficiencies_mx(m, x)``, ``efficiencies(m, d, lambda0, n_env)``
    qext, qsca, qback and g.
``S1_S2(m, x, mu, norm)``
    The scattering amplitudes at each mu = cos(theta).
``i_par``, ``i_per``, ``i_unpolarized``, ``intensities``
    Scattered intensities at each mu.
``phase_matrix(m, x, mu, norm)``
    The 4x4 Mueller matrix at each mu.

Nothing here takes ``n_pole``, since there is only the dipole.

Order of approximation:
    Strictly, Rayleigh scattering is only the leading term in x: an electric
    dipole whose strength is set by the static polarizability. Not every
    expression here stops at that term, so this is better described as a
    small-particle approximation.

    - qsca and qback are the leading terms, 8/3 x**4 abs(ratio)**2 and
      4 x**4 abs(ratio)**2.
    - qext carries Bohren and Huffman's next-order correction. Its absorption
      term includes a relative x**2 correction, the order at which the
      magnetic dipole and the electric quadrupole first contribute (the
      2 m**2 + 3 in its denominator is the quadrupole's), and scattering is
      added as 8/3 x**4 Re(ratio**2).
    - ``S1_S2`` gives the electric dipole its own relative x**2 correction, the
      x**5 term, but leaves out the magnetic dipole and quadrupole. The angular
      pattern is therefore exactly that of a dipole, and g is 0.
    - A perfect conductor gets only the leading terms of its electric and
      magnetic dipoles.

    The corrections are not applied evenly, so a dielectric's qext is more
    accurate than its qsca. ``miepython`` itself goes further for x < 0.1,
    using Wiscombe's expansions, which keep the electric dipole, the magnetic
    dipole and the electric quadrupole, each with its own corrections.

Accuracy:
    Relative error in qext against the full Mie solution:

    =================  =========  ========  ========
    m                  x = 0.01   x = 0.1   x = 0.5
    =================  =========  ========  ========
    1.5                -7e-6      -7e-4     -1e-2
    1.5 - 0.1j         8e-10      6e-6      4e-3
    0.05 - 1.43j       2e-6       2e-2      14
    -1j (conductor)    -2e-5      -2e-3     -4e-2
    =================  =========  ========  ========

    The last row is a silver-like sphere near its plasmon resonance, where
    m**2 is close to -2 and the denominator of the ratio nearly vanishes. There
    the approximation fails at much smaller x than for an ordinary dielectric.

    For a dielectric the asymmetry parameter g is 0, because dipole scattering
    is the same forwards and backwards. The Mie value is 0.002 at x = 0.1 and
    0.05 at x = 0.5 for m = 1.5.

Conventions:
    The index is m = n - ik, as in the rest of the package, and an index with a
    positive imaginary part is conjugated. The inputs that ``miepython``
    rejects, a NaN or infinite m or x, a negative x or a negative real part of
    m, raise the same ``ValueError`` here. The efficiencies accept an array of
    m or x and return arrays, g included. The angular functions describe one
    sphere and vectorize over mu only.

Perfect conductors:
    As in the rest of the package, m.real == 0 means a perfect conductor,
    whatever the imaginary part, and m = 0 is one too. No field enters such a
    sphere, so the dielectric expressions, which describe the field inside,
    do not apply. Instead the surface currents radiate an electric dipole and
    a magnetic dipole half as strong and opposite in sign, a1 = 2i x**3/3 and
    b1 = -a1/2. The sphere absorbs nothing, qext = qsca = 10/3 x**4, and
    qback = 9 x**4. The two dipoles add backwards and partly cancel forwards,
    so g = -2/5 and the backscattered intensity is nine times the forward one.

Examples:
    A small absorbing sphere, compared with the full Mie solution:

    >>> import numpy as np
    >>> import miepython as mie
    >>> from miepython import rayleigh
    >>> qext, qsca, qback, g = rayleigh.efficiencies_mx(1.5 - 0.1j, 0.1)
    >>> print(f"{qext:.5f} {qsca:.6f} {g:.4f}")
    0.02006 0.000024 0.0000
    >>> qext, qsca, qback, g = mie.efficiencies_mx(1.5 - 0.1j, 0.1)
    >>> print(f"{qext:.5f} {qsca:.6f} {g:.4f}")
    0.02006 0.000024 0.0020

    The dipole pattern scatters equally forwards and backwards, and half as
    much at 90 degrees:

    >>> mu = np.array([-1.0, 0.0, 1.0])
    >>> print(np.round(rayleigh.i_unpolarized(1.5, 0.1, mu), 4))
    [0.1195 0.0597 0.1195]
    >>> print(np.round(mie.i_unpolarized(1.5, 0.1, mu), 4))
    [0.1188 0.0597 0.1199]

    A perfect conductor scatters mostly backwards:

    >>> qext, qsca, qback, g = rayleigh.efficiencies_mx(-1j, 0.1)
    >>> print(f"{qsca:.4e} {qback:.4e} {g:.4f}")
    3.3333e-04 9.0000e-04 -0.4000
    >>> qext, qsca, qback, g = mie.efficiencies_mx(-1j, 0.1)
    >>> print(f"{qsca:.4e} {qback:.4e} {g:.4f}")
    3.3413e-04 8.9834e-04 -0.3973
"""

import numpy as np

from .core import _check_m_x, _check_one_sphere
from .core import normalization_factor as core_normalization_factor

__all__ = (
    "efficiencies_mx",
    "efficiencies",
    "S1_S2",
    "i_par",
    "i_per",
    "i_unpolarized",
    "intensities",
    "phase_matrix",
)


def efficiencies_mx(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Based on equations 5.7 - 5.9 in Bohren and Huffman.  A perfect conductor,
    m.real == 0, uses the electric and magnetic dipoles instead; the module
    docstring gives the details.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function, 0 for a dielectric, -0.4 for a conductor

    Raises:
        ValueError: if m or x is not finite, x is negative, or the real part of
            m is negative, exactly as for the full Mie functions
    """
    _check_m_x(m, x)

    # ensure imaginary part of refractive index is negative, as core.py does
    if np.isscalar(m):
        m = np.conj(m) if np.imag(m) > 0 else m
    else:
        m = np.where(np.imag(m) > 0, np.conj(m), m)

    # the dielectric expressions are evaluated for every entry and then replaced
    # where m is a conductor; a few conductor indices, m = -1j*sqrt(2) and
    # m = -1j*sqrt(1.5), make their denominators vanish, so silence the warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (m**2 - 1) / (m**2 + 2)
        qsca = 8 / 3 * x**4 * abs(ratio) ** 2
        # Bohren & Huffman write m = n + ik, so with m = n - ik the absorption term
        # changes sign.  The x**4 term is Re(ratio**2), not |ratio|**2: the two only
        # agree for a non-absorbing sphere, and the difference matters near resonance.
        qabs = -4 * x * (ratio * (1 + x**2 / 15 * ratio * (m**4 + 27 * m**2 + 38) / (2 * m**2 + 3))).imag
        qext = qabs + 8 / 3 * x**4 * (ratio**2).real
        qback = 4 * x**4 * abs(ratio) ** 2

    # A perfect conductor (m.real == 0) radiates a magnetic dipole half as strong
    # as, and opposite to, its electric one: a1 = 2i x**3/3 and b1 = -a1/2.  It
    # absorbs nothing, scatters 6/x**2 (|a1|**2 + |b1|**2) = 10/3 x**4, and the
    # interference of the two dipoles throws light backwards, so g = -2/5.
    conductor = np.real(m) == 0
    qsca = np.where(conductor, 10 / 3 * x**4, qsca)[()]
    qext = np.where(conductor, 10 / 3 * x**4, qext)[()]
    qback = np.where(conductor, 9 * x**4, qback)[()]
    # a dielectric's electric dipole alone scatters symmetrically fore and aft
    g = np.where(conductor, -0.4, 0 * qsca)[()]
    return qext, qsca, qback, g


def efficiencies(m, d, lambda0, n_env=1.0):
    """
    Calculate the efficiencies of a sphere using Rayleigh's approximation.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        n_env: real index of medium around sphere, optional.

    Returns:
        qext: the total extinction efficiency                  [-]
        qsca: the scattering efficiency                        [-]
        qback: the backscatter efficiency                      [-]
        g: the average cosine of the scattering phase function [-]
    """
    m_env = m / n_env
    x_env = np.pi * d / (lambda0 / n_env)
    return efficiencies_mx(m_env, x_env)


def normalization_factor(m, x, norm_str):
    """
    Figure out scattering function normalization for a small sphere.

    The rules are the same as for the full Mie solution, so they live in
    ``miepython.core``; only the efficiencies feeding them are the Rayleigh ones.

    Args:
        m: complex index of refraction of sphere
        x: dimensionless sphere size
        norm_str: string describing type of normalization

    Returns:
        scaling factor needed for scattering function
    """
    return core_normalization_factor(m, x, norm_str, efficiency_source=efficiencies_mx)


def S1_S2(m, x, mu, norm="albedo"):
    """
    Calculate the scattering amplitude functions for small spheres.

    Based on equation 5.4 in Bohren and Huffman.  A perfect conductor,
    m.real == 0, adds the magnetic dipole b1 = -a1/2 to its electric one.

    The amplitude functions are normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes
        norm: (optional) string describing scattering function normalization

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]

    Raises:
        ValueError: if m or x is not finite, x is negative, or the real part of
            m is negative, exactly as for the full Mie functions, or if m or x
            is an array, since this describes one sphere
    """
    _check_m_x(m, x)
    _check_one_sphere(m, x)
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    if np.real(m) == 0:
        # perfect conductor: an electric dipole and a magnetic one, b1 = -a1/2
        a1 = 2j * x**3 / 3
        b1 = -a1 / 2
    else:
        a1 = (2 * x**3) / 3 * (m**2 - 1) / (m**2 + 2) * 1j
        a1 += (2 * x**5) / 5 * (m**2 - 2) * (m**2 - 1) / (m**2 + 2) ** 2 * 1j
        b1 = 0

    # the n = 1 terms of the Mie series, where pi_1 = 1 and tau_1 = mu
    S1 = (3 / 2) * (a1 * np.ones_like(mu) + b1 * mu)
    S2 = (3 / 2) * (a1 * mu + b1 * np.ones_like(mu))

    normalization = normalization_factor(m, x, norm)

    S1 /= normalization
    S2 /= normalization

    return S1, S2


def i_per(m, x, mu, norm="albedo"):
    """
    Return the perpendicular scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, _ = S1_S2(m, x, mu, norm)
    intensity = np.abs(s1) ** 2
    return intensity.astype("float")


def i_par(m, x, mu, norm="albedo"):
    """
    Return the parallel scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    _, s2 = S1_S2(m, x, mu, norm)
    intensity = np.abs(s2) ** 2
    return intensity.astype("float")


def i_unpolarized(m, x, mu, norm="albedo"):
    """
    Return the unpolarized scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = S1_S2(m, x, mu, norm)
    intensity = (abs(s1) ** 2 + abs(s2) ** 2) / 2
    return intensity.astype("float")


def intensities(m, d, lambda0, mu, n_env=1.0, norm="albedo"):
    """
    Return the scattered intensities from a sphere.

    These are the scattered intensities in a plane that is parallel (ipar) and
    perpendicular (iper) to the field of the incident plane wave.

    The scattered intensity is normalized such that the integral of the
    unpolarized intensity over 4𝜋 steradians is equal to the single scattering
    albedo.  The scattered intensity has units of inverse steradians [1/sr].

    The unpolarized scattering is the average of the two scattered intensities.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        mu: the cos(theta) of each direction desired     [-]
        n_env: real index of medium around sphere, optional.
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        ipar, iper: scattered intensity in parallel and perpendicular planes [1/sr]
    """
    m_env = m / n_env
    lambda_env = lambda0 / n_env
    x_env = np.pi * d / lambda_env
    s1, s2 = S1_S2(m_env, x_env, mu, norm)
    ipar = np.abs(s2) ** 2
    iper = np.abs(s1) ** 2
    Ipar = ipar.astype("float")
    Iper = iper.astype("float")
    return Ipar, Iper


def phase_matrix(m, x, mu, norm="albedo"):
    """
    Calculate the scattering (Mueller) matrix.

    If mu has length N, then the returned matrix is 4x4xN.  If mu is a scalar
    then the matrix is 4x4

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), of the phase scattering matrix
        n_pole: return n_pole term from series (default=0 means include all terms)
        norm: (optional) string describing scattering function normalization

    Returns:
        p: the phase scattering matrix [sr**(-1.0)]
    """
    mu = np.atleast_1d(mu)
    s1, s2 = S1_S2(m, x, mu, norm)

    s1_star = np.conjugate(s1)
    s2_star = np.conjugate(s2)
    m1 = (s1 * s1_star).real
    m2 = (s2 * s2_star).real
    s21 = (0.5 * (s1 * s2_star + s2 * s1_star)).real
    d21 = (-0.5j * (s1 * s2_star - s2 * s1_star)).real
    phase = np.zeros(shape=(4, 4, mu.size))
    phase[0, 0] = 0.5 * (m2 + m1)
    phase[0, 1] = 0.5 * (m2 - m1)
    phase[1, 0] = phase[0, 1]
    phase[1, 1] = phase[0, 0]
    phase[2, 2] = s21
    phase[2, 3] = -d21
    phase[3, 2] = d21
    phase[3, 3] = s21

    # squeeze returns a (4, 4) matrix rather than (4, 4, 1) when mu.size == 1
    return phase.squeeze()
