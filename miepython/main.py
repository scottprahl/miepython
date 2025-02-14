# pylint: disable=unused-argument
# pylint: disable=too-many-return-statements
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
"""
Mie scattering calculations for perfect spheres.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-np.absorbing, partially-np.absorbing, or perfectly conducting
spheres.

The extinction efficiency, scattering efficiency, backscattering, and
scattering asymmetry for a sphere with complex index of refraction m,
diameter d, and wavelength lambda can be found by::

    import miepython as mie
    qext, qsca, qback, g = mie.ez_mie(m, d, lambda0)

The normalized scattering values for angles mu=cos(theta) are::

    import miepython as mie
    Ipar, Iper = mie.ez_intensities(m, d, lambda0, mu)

If the size parameter is known, then use::

    import miepython as mie
    mie.mie(m, x)

Mie scattering amplitudes S1 and S2 (complex numbers):

    mie.mie_S1_S2(m, x, mu)

Normalized Mie scattering intensities for angles mu=cos(theta)::

    mie.i_per(m, x, mu)
    mie.i_par(m, x, mu)
    mie.i_unpolarized(m, x, mu)
"""
import os
import importlib
import sys
import numpy as np
from miepython import _an_bn, _cn_dn, _mie_S1_S2

# not really needed but included for clarity
__all__ = (
    "ez_mie",
    "ez_intensities",
    "i_par",
    "i_per",
    "i_unpolarized",
    "mie",
    "mie_S1_S2",
    "mie_phase_matrix",
    "mie_coefficients",
    "an_bn",
    "cn_dn",
    "abcd",
)


def an_bn(m, x, n_pole=0):
    """
    Compute arrays of Mie coefficients A and B for a sphere.

    When n_pole=0, the routine estimates the size of the arrays based on Wiscombe's
    formula. The length of the arrays is chosen so that the error when the series
    is summed is around 1e-6.

    If n_pole>0, then the array sizes will be n_pole+1. This is useful when
    trying to isolate the behavior of a particular multipole.

    To support resonance calculations, one can specify the number of terms
    to be calculated.  In general, using too few or too many terms increases the
    error rate.  So if you specify the number of terms be aware that you are
    playing with fire.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: the number of An and Bn terms (0 does autosizing)

    Returns:
        a, b: arrays of Mie coefficents An and Bn
    """
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    return _an_bn(np.complex128(m), np.float64(x), np.int64(n_pole))


def cn_dn(m, x, n_pole=0):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (complex): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    return _cn_dn(np.complex128(m), np.float64(x), np.int64(n_pole))


def abcd(m, x, n_pole=0):
    """Calculate inner and outer sphere coefficients."""
    a, b = an_bn(m, x, n_pole)
    c, d = cn_dn(m, x, n_pole)
    return [a, b, c, d]


def mie_coefficients(m, x, n_pole=0):
    """
    Computes the Mie coefficients (A_n and B_n) for a sphere.

    This function calculates the Mie coefficients for electromagnetic scattering
    by a sphere. It supports both single values and arrays for the refractive
    index and size parameter. For arrays, the lengths of `m` and `x` must match.
    If the length of `m` or `x` is zero, the function assumes scalar values.

    If `n_pole > 0`, the function returns only the nth multipole coefficient
    for each input.

    Args:
        m (complex or array-like): The complex refractive index of the sphere.
            If an array, must match the length of `x`.
        x (float or array-like): The size parameter of the sphere. If an array,
            must match the length of `m`.
        n_pole (int, optional): The specific multipole order to compute. Defaults to 0,
            which calculates all terms and returns the full arrays of coefficients.

    Returns:
        tuple:
            - a (complex or array-like): The computed Mie coefficient A_n or an array of A_n values.
            - b (complex or array-like): The computed Mie coefficient B_n or an array of B_n values.

    Notes:
        - If the imaginary part of the refractive index is positive, it is
          automatically corrected to its conjugate value to ensure a valid input.

    Examples:
        Compute coefficients for a single sphere:
        >>> m = 1.5 - 0.1j
        >>> x = 0.1
        >>> mie_coefficients(m, x)
        (array([3.33370015e-05+1.97363100e-04j, 1.77504613e-08+1.11834113e-07j,
        4.60529820e-12+2.97368373e-11j, 1.58460985e-28+1.25881287e-14j]),
        array([6.67296256e-08+2.75469444e-07j, 1.90474848e-11+7.86835968e-11j,
        3.02615454e-15+1.24937186e-14j, 1.58460985e-28+1.25881287e-14j]))

        Compute first multipoles for multiple spheres:
        >>> m = [1.5 + 0.1j, 1.4 + 0.05j]
        >>> x = [2.0, 1.8]
        >>> mie_coefficients(m, x, 1)
        (array([0.44794644+0.38665192j, 0.27813728+0.38495241j]),
        array([0.5621083 +0.25504616j, 0.20206531+0.29589842j]))
    """
    #    from .miepython_core import _an_bn  # Ensure the latest function is used
    mlen = 0
    try:
        mlen = len(m)
    except TypeError:
        pass

    xlen = 0
    try:
        xlen = len(x)
    except TypeError:
        pass

    if xlen > 0 and mlen > 0 and xlen != mlen:
        raise RuntimeError("m and x arrays to mie must be same length")

    if mlen == 0 and xlen == 0:
        a, b = _an_bn(m, x, n_pole)
        if n_pole == 0:
            return a, b
        return a[n_pole - 1], b[n_pole - 1]

    if np.isscalar(m):
        m = np.conj(m) if np.imag(m) > 0 else m
    else:
        m = np.where(np.imag(m) > 0, np.conj(m), m)

    thelen = max(xlen, mlen)
    a = np.zeros(thelen, dtype=np.complex128)
    b = np.zeros(thelen, dtype=np.complex128)

    mm = m
    xx = x
    for i in range(thelen):
        if mlen > 0:
            mm = m[i]

        if xlen > 0:
            xx = x[i]

        an, bn = _an_bn(mm, xx, n_pole)
        a[i] = an[n_pole - 1]
        b[i] = bn[n_pole - 1]

    return a, b


def _small_conducting_mie(_m, x):
    """
    Calculate the efficiencies for a small conducting spheres.

    Typically used for small conducting spheres where x < 0.1 and
    m.real == 0

    Args:
        _m: the complex index of refraction of the sphere (unused)
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ahat1 = complex(0, 2.0 / 3.0 * (1 - 0.2 * x**2))
    ahat1 /= complex(1 - 0.5 * x**2, 2.0 / 3.0 * x**3)

    bhat1 = complex(0.0, (x**2 - 10.0) / 30.0)
    bhat1 /= complex(1 + 0.5 * x**2, -(x**3) / 3.0)
    ahat2 = complex(0.0, x**2 / 30.0)
    bhat2 = complex(0.0, -(x**2) / 45.0)

    qsca = x**4 * (
        6 * np.abs(ahat1) ** 2
        + 6 * np.abs(bhat1) ** 2
        + 10 * np.abs(ahat2) ** 2
        + 10 * np.abs(bhat2) ** 2
    )
    qext = qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0 / 9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6 * x**4 / qsca

    qback = 9 * x**4 * np.abs(ahat1 - bhat1 - 5 / 3 * (ahat2 - bhat2)) ** 2

    return qext, qsca, qback, g


def _small_mie(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Typically used for small spheres where x<0.1

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    m2 = m * m
    x2 = x * x

    D = m2 + 2 + (1 - 0.7 * m2) * x2
    D -= (8 * m**4 - 385 * m2 + 350) * x**4 / 1400.0
    D += 2j * (m2 - 1) * x**3 * (1 - 0.1 * x2) / 3
    ahat1 = 2j * (m2 - 1) / 3 * (1 - 0.1 * x2 + (4 * m2 + 5) * x**4 / 1400) / D

    bhat1 = 1j * x2 * (m2 - 1) / 45 * (1 + (2 * m2 - 5) / 70 * x2)
    bhat1 /= 1 - (2 * m2 - 5) / 30 * x2

    ahat2 = 1j * x2 * (m2 - 1) / 15 * (1 - x2 / 14)
    ahat2 /= 2 * m2 + 3 - (2 * m2 - 7) / 14 * x2

    T = np.abs(ahat1) ** 2 + np.abs(bhat1) ** 2 + 5 / 3 * np.abs(ahat2) ** 2
    temp = ahat2 + bhat1
    g = (ahat1 * temp.conjugate()).real / T

    qsca = 6 * x**4 * T

    if m.imag == 0:
        qext = qsca
    else:
        qext = 6 * x * (ahat1 + bhat1 + 5 * ahat2 / 3).real

    sback = 1.5 * x**3 * (ahat1 - bhat1 - 5 * ahat2 / 3)
    qback = 4 * np.abs(sback) ** 2 / x2

    return qext, qsca, qback, g


def _mie_scalar(m, x, n_pole=0, e_field=True):
    """
    Calculate the efficiencies for a sphere when both m and x are scalars.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: a non-zero value returns the contribution by the n_pole multipole
        e_field: Electric (True) or Magnetic Field otherwise

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    #    from .miepython_core import _an_bn  # Ensure the latest function is used

    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0, 0, 0, 0

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_mie(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_mie(m, x)

    # sometimes m=0 is used to signal perfectly conducting sphere
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        m = 1 - 10000j

    a, b = _an_bn(m, x, n_pole)

    if n_pole == 0:
        n = np.arange(1, len(a) + 1)
        cn = 2.0 * n + 1.0

        qext = 2 * np.sum(cn * (a.real + b.real)) / x**2

        if m.imag == 0:
            qsca = qext
        else:
            qsca = 2 * np.sum(cn * (np.abs(a) ** 2 + np.abs(b) ** 2)) / x**2

        qback = np.abs(np.sum((-1) ** n * cn * (a - b))) ** 2 / x**2

        c1n = n * (n + 2) / (n + 1)
        c2n = cn / n / (n + 1)
        asy1 = c1n[:-1] * (a[:-1] * a[1:].conjugate() + b[:-1] * b[1:].conjugate()).real
        asy2 = c2n[:-1] * (a[:-1] * b[:-1].conjugate()).real
        g = 4 * np.sum(asy1 + asy2) / qsca / x**2

    else:
        cn = 2.0 * n_pole + 1
        c1n = n_pole * (n_pole + 2) / (n_pole + 1)
        if e_field:
            qext = 2 * cn * a[n_pole - 1].real / x**2
            qsca = 2 * cn * np.abs(a[n_pole - 1]) ** 2 / x**2
            qback = qsca / 2
            g = 4 * c1n * (a[n_pole - 2] * a[n_pole - 1].conjugate()).real / qsca / x**2
        else:
            qext = 2 * cn * b[n_pole - 1].real / x**2
            qsca = 2 * cn * np.abs(b[n_pole - 1]) ** 2 / x**2
            qback = qsca / 2
            g = 4 * c1n * (b[n_pole - 2] * b[n_pole - 1].conjugate()).real / qsca / x**2

    return qext, qsca, qback, g


def mie(m, x, n_pole=0, field="Electric"):
    """
    Computes scattering and extinction efficiencies for a spherical particle using Mie theory.

    Supports array inputs for wavelength-dependent calculations of the refractive index
    and size parameter.

    Args:
        m (complex or array-like):
            Complex refractive index of the sphere, defined as m = n - ik,
            where n is the real part (phase velocity) and k is the
            imaginary part (absorption). Can be a scalar or an array.
        x (float or array-like):
            Size parameter of the sphere, defined as x = π d / λ,
            where d is the diameter of the sphere and λ is the
            wavelength in the surrounding medium. Can be a scalar or an array.
        n_pole (int):
            Multipole order to compute:
            - If 0 (default), computes contributions from all multipoles.
            - If non-zero, computes contributions from the specified multipole order.
        field (str):
            If "Electric" (default) If n_pole>0, then True value returns the electric
            multipole contribution otherwise the Magnetic field contribution
    Returns:
        tuple:
            qext (float or array-like):
                Extinction efficiency, representing the total attenuation of light
                (scattering plus absorption) by the particle.
            qsca (float or array-like):
                Scattering efficiency, representing the fraction of incident light
                scattered by the particle.
            qback (float or array-like):
                Backscattering efficiency, describing the fraction of incident light
                scattered in the exact backward direction (theta = 180 degrees).
            g (float or array-like):
                Asymmetry parameter, representing the average cosine of the scattering
                angle over all angles.

    Notes:
        - Ensure m and x have compatible dimensions if passed as arrays.
        - For accurate results, n_pole should be within the range of significant
          multipole contributions, typically n ~ x + 4*x**(1/3).
        - This implementation assumes spherical, homogeneous particles in a
          non-absorbing medium.

    Examples:
        Compute efficiencies for a sphere with fixed parameters:

        >>> mie(1.5 - 0.01j, 2.0, 0)
        (qext: 1.81, qsca: 1.72, qback: 0.27, g: 0.63)

        Compute efficiencies for wavelength-dependent refractive indices:

        >>> m_array = np.array([1.5 - 0.01j, 1.45 - 0.02j])
        >>> x_array = np.array([2.0, 2.5])
        >>> mie(m_array, x_array, 0)
        (qext: [1.81, 2.15], qsca: [1.72, 1.95], qback: [0.27, 0.31], g: [0.63, 0.70])
    """
    # ensure imaginary part of refractive index is negative
    if np.isscalar(m):
        m = np.conj(m) if np.imag(m) > 0 else m
    else:
        m = np.where(np.imag(m) > 0, np.conj(m), m)

    mlen = 0
    if hasattr(m, "__len__"):
        mlen = len(m)

    xlen = 0
    if hasattr(x, "__len__"):
        xlen = len(x)

    if mlen == 0 and xlen == 0:
        return _mie_scalar(m, x, n_pole)

    if xlen > 0 and mlen > 0 and xlen != mlen:
        raise RuntimeError("m and x arrays to mie must be same length")

    thelen = max(xlen, mlen)
    qext = np.empty(thelen, dtype=np.float64)
    qsca = np.empty(thelen, dtype=np.float64)
    qback = np.empty(thelen, dtype=np.float64)
    g = np.empty(thelen, dtype=np.float64)

    mm = m
    xx = x
    for i in range(thelen):
        if mlen > 0:
            mm = m[i]

        if xlen > 0:
            xx = x[i]

        qext[i], qsca[i], qback[i], g[i] = _mie_scalar(mm, xx, n_pole)

    return qext, qsca, qback, g


def normalization_factor(m, x, norm_str):
    """
    Figure out scattering function normalization.

    Args:
        m: complex index of refraction of sphere
        x: dimensionless sphere size
        norm_str: string describing type of normalization

    Returns:
        scaling factor needed for scattering function
    """
    factor = None
    norm = norm_str.lower()

    if norm in ["bohren"]:
        factor = 1 / 2

    elif norm in ["wiscombe"]:
        factor = 1

    elif norm in ["qsca", "scattering_efficiency"]:
        factor = x * np.sqrt(np.pi)

    else:
        qext, qsca, _, _ = _mie_scalar(m, x, 0)

        if norm in ["a", "albedo"]:
            factor = x * np.sqrt(np.pi * qext)

        if norm in ["1", "one", "unity"]:
            factor = x * np.sqrt(qsca * np.pi)

        if norm in ["four_pi", "4pi"]:
            factor = x * np.sqrt(qsca / 4)

        if norm in ["qext", "extinction_efficiency"]:
            factor = x * np.sqrt(qsca * np.pi / qext)

    if factor is None:
        raise ValueError(
            "normalization must be one of 'albedo' (default), 'one'"
            "'4pi', 'qext', 'qsca', 'bohren', or 'wiscombe'"
        )
    return factor


def mie_S1_S2(m, x, mu, norm="albedo", n_pole=0):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    if np.isscalar(mu):
        mu_array = np.array([mu], dtype=float)
        S1, S2 = _mie_S1_S2(m, x, mu_array, n_pole)
    else:
        S1, S2 = _mie_S1_S2(m, x, mu, n_pole)

    normalization = normalization_factor(m, x, norm)

    S1 /= normalization
    S2 /= normalization

    return S1, S2


def mie_phase_matrix(m, x, mu, norm="albedo", n_pole=0):
    """
    Calculate the scattering (Mueller) matrix.

    If mu has length N, then the returned matrix is 4x4xN.  If mu is a scalar
    then the matrix is 4x4

    The phase scattering matrix is computed from the scattering amplitude
    functions, according to equations 5.2.105-6 in K. N. Liou (**2002**) -
    *An Introduction to Atmospheric Radiation*, Second Edition.

    or

    Bohren and Huffman, *Absorption and Scattering of Light by Small Particles*,
    JOHN WILEY & SONS, page 112, (1983).

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
    s1, s2 = mie_S1_S2(m=m, x=x, mu=mu, norm=norm, n_pole=n_pole)

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

    return phase.squeeze()


def i_per(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the scattered intensity in a plane normal to the incident light.

    This is the scattered intensity in a plane that is perpendicular to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, _ = mie_S1_S2(m, x, mu, norm, n_pole)
    intensity = np.abs(s1) ** 2
    return intensity.astype("float")


def i_par(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the scattered intensity in a plane parallel to the incident light.

    This is the scattered intensity in a plane that is parallel to the
    field of the incident plane wave. The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    _, s2 = mie_S1_S2(m, x, mu, norm, n_pole)
    intensity = np.abs(s2) ** 2
    return intensity.astype("float")


def i_unpolarized(m, x, mu, norm="albedo", n_pole=0):
    """
    Return the unpolarized scattered intensity at specified angles.

    This is the average value for randomly polarized incident light.
    The intensity is normalized such
    that the integral of the unpolarized intensity over 4π steradians
    is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means add all terms)

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = mie_S1_S2(m, x, mu, norm, n_pole)
    intensity = (np.abs(s1) ** 2 + np.abs(s2) ** 2) / 2
    return intensity.astype("float")


def ez_mie(m, d, lambda0, n_env=1.0):
    """
    Calculate the efficiencies of a sphere.

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
    return mie(m_env, x_env)


def ez_intensities(m, d, lambda0, mu, n_env=1.0, norm="albedo", n_pole=0):
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
    s1, s2 = mie_S1_S2(m_env, x_env, mu, norm, n_pole)
    ipar = np.abs(s2) ** 2
    iper = np.abs(s1) ** 2
    Ipar = ipar.astype("float")
    Iper = iper.astype("float")
    return Ipar, Iper


def dPnm_dcos(n, m, mu):
    """
    Compute the derivative of the associated Legendre polynomial P_n^m(mu).

    This is the derivative relative to the argument, i.e. with respect to mu = cos(theta).

    Args:
        n (int): Degree of the polynomial.
        m (int): Order of the polynomial.
        mu (float): cos(theta), where -1 <= mu <= 1.

    Returns:
        float: Derivative of P_n^m(mu) with respect to mu.
    """
    if abs(mu) == 1:
        # Handle singularities at mu = ±1
        return 0.0
    Pnm = lpmv(m, n, mu)  # P_n^m(mu)
    if n == 0:
        return 0.0
    Pn_minus_1_m = lpmv(m, n - 1, mu)  # P_{n-1}^m(mu)
    return (n * mu * Pnm - (n + m) * Pn_minus_1_m) / (1 - mu**2)


def dYnm_dtheta(n, m, theta, phi):
    """
    Compute the derivative of spherical harmonics with respect to theta.

    Args:
        n (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.
        theta (float): Polar angle (0 <= theta <= pi).
        phi (float): Azimuthal angle (0 <= phi <= 2*pi).

    Returns:
        complex: Derivative of Y_n^m with respect to theta.
    """
    mu = np.cos(theta)
    dPnm = dPnm_dcos(n, m, mu)  # Derivative of P_n^m with respect to cos(theta)
    return -np.sin(theta) * dPnm * np.exp(1j * m * phi)


def dYnm_dphi(n, m, theta, phi):
    """
    Compute the derivative of spherical harmonics with respect to phi.

    Args:
        n (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.
        theta (float): Polar angle (0 <= theta <= pi).
        phi (float): Azimuthal angle (0 <= phi <= 2*pi).

    Returns:
        complex: Derivative of Y_n^m with respect to phi.
    """
    Ynm = sph_harm_y(m, n, phi, theta)
    return 1j * m * Ynm


def efield_inside(abcd, kr, theta, phi):
    """
    Calculate the electric field inside a sphere given c_n and d_n.

    Args:
        abcd (tuple): (a_n, b_n, c_n, d_n) coefficients.
        kr (float): Radial position inside the sphere in size parameter units (k*r).
        theta (float): Polar angle (0 <= theta <= pi).
        phi (float): Azimuthal angle (0 <= phi <= 2*pi).

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    _, _, c_n, d_n = abcd
    E_r = 0
    E_theta = 0
    E_phi = 0

    for n in range(1, len(c_n) + 1):
        psi_n_kr = spherical_jn(n, kr)
        P_n_theta = sph_harm_y(0, n, phi, theta)
        dP_n_theta = dYnm_dtheta(n, 0, theta, phi)
        dP_n_phi = dYnm_dphi(n, 0, theta, phi)

        E_r += c_n[n - 1] * psi_n_kr * P_n_theta
        E_theta += d_n[n - 1] * psi_n_kr * dP_n_theta
        E_phi += d_n[n - 1] * psi_n_kr * dP_n_phi

    return np.real(E_r), np.real(E_theta), np.real(E_phi)


def efield_outside(abcd, kr, theta, phi):
    """
    Calculate the electric field outside a sphere given a_n and b_n.

    Args:
        abcd (tuple): (a_n, b_n, c_n, d_n) coefficients.
        kr (float): Radial position outside the sphere in size parameter units (k*r).
        theta (float): Polar angle (0 <= theta <= pi).
        phi (float): Azimuthal angle (0 <= phi <= 2*pi).

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    a_n, b_n, _, _ = abcd
    E_r = 0
    E_theta = 0
    E_phi = 0

    for n in range(1, len(a_n) + 1):
        psi_n_kr = spherical_jn(n, kr)
        chi_n_kr = spherical_yn(n, kr)
        xi_n_kr = psi_n_kr + 1j * chi_n_kr

        P_n_theta = sph_harm_y(0, n, phi, theta)
        dP_n_theta = dYnm_dtheta(n, 0, theta, phi)
        dP_n_phi = dYnm_dphi(n, 0, theta, phi)

        E_r += (a_n[n - 1] + b_n[n - 1]) * xi_n_kr * P_n_theta
        E_theta += b_n[n - 1] * xi_n_kr * dP_n_theta
        E_phi += b_n[n - 1] * xi_n_kr * dP_n_phi

    return np.real(E_r), np.real(E_theta), np.real(E_phi)


def efield(d_sphere, n_sph, n_env, lambda0, abcd, r, theta, phi):
    """
    Calculate the electric field inside or outside a sphere.

    Args:
        d_sphere (float): Diameter of the sphere.
        n_sph (float): Refractive index of the sphere.
        n_env (float): Refractive index of the surrounding medium.
        lambda0 (float): Wavelength in vacuum.
        abcd (tuple): (a_n, b_n, c_n, d_n) coefficients.
        r (float): Radial position.
        theta (float): Polar angle (0 <= theta <= pi).
        phi (float): Azimuthal angle (0 <= phi <= 2*pi).

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    if r <= d_sphere / 2:
        k = 2 * np.pi * n_env / lambda0
        kr = k * r
        return efield_inside(abcd, kr, theta, phi)

    k = 2 * np.pi * n_env / lambda0
    kr = k * r
    return efield_outside(abcd, kr, theta, phi)
