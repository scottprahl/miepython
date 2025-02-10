# pylint: disable=unused-argument
# pylint: disable=too-many-return-statements
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
"""
Mie scattering calculations for perfect spheres JITTED!.

Extensive documentation is at <https://miepython.readthedocs.io>

`miepython` is a pure Python module to calculate light scattering of
a plane wave by non-np.absorbing, partially-np.absorbing, or perfectly conducting
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
"""

import numpy as np
from numba import njit, int64, float64, complex128, bool_

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
    )


@njit((complex128, int64), cache=True)
def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function

    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    zinv = 2.0 / z
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = 1.0 / alpha_j1 + aj
        alpha_j2 = 1.0 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -1
        runratio = ratio * runratio

    return -N / z + runratio


@njit((complex128, int64, complex128[:]), cache=True)
def _D_downwards(z, N, D):
    """
    Compute the logarithmic derivative by downwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the downwards recurrence relations.
    """
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        last_D = n / z - 1.0 / (last_D + n / z)
        D[n - 1] = last_D


@njit((complex128, int64, complex128[:]), cache=True)
def _D_upwards(z, N, D):
    """
    Compute the logarithmic derivative by upwards recurrence.

    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the upwards recurrence relations.
    """
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z


@njit((complex128, float64, int64), cache=True)
def _D_calc(m, x, N):
    """
    Compute the logarithmic derivative using best method.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function

    Returns:
        The values of the Ricatti-Bessel function for orders from 0 to N.
    """
    n = m.real
    kappa = np.abs(m.imag)
    D = np.zeros(N, dtype=np.complex128)
    mx = np.complex128(m * x)  # ensure complex

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n**2:
        _D_downwards(mx, N, D)
    else:
        _D_upwards(mx, N, D)
    return D[1:]


@njit((complex128, float64, int64), cache=True)
def an_bn(m, x, n_pole):
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
    # ensure imaginary part of refractive index is negative
    if np.imag(m) > 0:
        m = np.conj(m)

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    a = np.zeros(nstop, dtype=np.complex128)
    b = np.zeros(nstop, dtype=np.complex128)

    psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - np.cos(x)
    xi_nm1 = complex(psi_nm1, np.cos(x))
    xi_n = complex(psi_n, np.cos(x) / x + np.sin(x))

    if m.real > 0.0:
        D = _D_calc(m, x, nstop + 1)

        for n in range(1, nstop):
            temp = D[n - 1] / m + n / x
            a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n - 1] * m + n / x
            b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            psi = (2 * n + 1) * psi_n / x - psi_nm1
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = psi

    else:
        for n in range(1, nstop):
            a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
            b[n - 1] = psi_n / xi_n
            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    if n_pole != 0:
        a = a[:-1]
        b = b[:-1]

    return np.conjugate(a), np.conjugate(b)


@njit((complex128, float64, int64), fastmath=True)
def cn_dn(m, x, n_pole):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (complex): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    # ensure imaginary part of refractive index is negative
    m = np.where(np.imag(m) > 0, np.conj(m), m)
    mx = m * x

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    c = np.zeros(nstop, dtype=np.complex128)
    d = np.zeros(nstop, dtype=np.complex128)

    # no need to calculate anything when sphere is perfectly conducting
    if m.real > 0.0 and not np.isinf(m.real) or not np.isinf(m.imag):
        psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
        psi_n = psi_nm1 / x - np.cos(x)

        psi_nm1_mx = np.sin(mx)  # nm1 = n-1 = 0
        psi_n_mx = psi_nm1_mx / mx - np.cos(mx)

        xi_nm1 = complex(psi_nm1, np.cos(x))
        xi_n = complex(psi_n, np.cos(x) / x + np.sin(x))

        Dmx = _D_calc(np.complex128(m), x, nstop + 1)
        Dx = _D_calc(np.complex128(1), x, nstop + 1)

        for n in range(1, nstop + 1):
            common = (psi_n / psi_n_mx) * ((Dx[n - 1] + n / x) * xi_n - xi_nm1)

            c[n - 1] = m * common / ((m * Dmx[n - 1] + n / x) * xi_n - xi_nm1)
            d[n - 1] = common / ((Dmx[n - 1] / m + n / x) * xi_n - xi_nm1)

            psi = (2 * n + 1) * psi_n / x - psi_nm1
            psi_nm1 = psi_n
            psi_n = psi

            psi_mx = (2 * n + 1) * psi_n_mx / mx - psi_nm1_mx
            psi_nm1_mx = psi_n_mx
            psi_n_mx = psi_mx

            xi = (2 * n + 1) * xi_n / x - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi

    if n_pole != 0:
        c = c[:-1]
        d = d[:-1]
    return np.conjugate(c), np.conjugate(d)


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
            - a (complex or array-like): Mie coefficient A_n or an array of A_k values.
            - b (complex or array-like): Mie coefficient B_n or an array of B_k values.

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
        a, b = an_bn(m, x, n_pole)
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

        an, bn = an_bn(mm, xx, n_pole)
        a[i] = an[n_pole - 1]
        b[i] = bn[n_pole - 1]

    return a, b


@njit((complex128, float64), cache=True)
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


@njit((complex128, float64), cache=True)
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


@njit((complex128, float64, int64, bool_), cache=True)
def _mie_scalar(m, x, n_pole, e_field):
    """
    Calculate the efficiencies for a sphere when both m and x are scalars.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: a non-zero value returns the contribution by the n_pole multipole
        e_field: True ==> Electric Field, False ==> Magnetic Field

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    # wierd case of conducting sphere with no absorption
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        return 0, 0, 0, 0

    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0, 0, 0, 0

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_mie(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_mie(m, x)

    a, b = an_bn(m, x, n_pole)

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
        return _mie_scalar(m, x, n_pole, True)

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
        qext[i], qsca[i], qback[i], g[i] = _mie_scalar(mm, xx, n_pole, True)

    return qext, qsca, qback, g


def normalization_factor(m, x, norm_int):
    """
    Figure out scattering function normalization.

    Args:
        m: the complex index of refraction of the sphere
        x: dimensionless sphere size
        norm_int: integer that specifies normalization

    Returns:
        scaling factor needed for scattering function
    """
    norm = None

    # Qsca normalization
    if norm_int == 3:
        norm = x * np.sqrt(np.pi)

    # Bohren Normalization
    elif norm_int == 5:
        norm = 0.5

    # Wiscombe Normalization
    elif norm_int == 6:
        norm = 1

    else:
        # calculate qsca and qext
        qext, qsca, _, _ = _mie_scalar(m, x, 0, True)

        # albedo Normalization
        if norm_int == 0:
            norm = x * np.sqrt(np.pi * qext)

        # Unity normalization
        elif norm_int == 1:
            norm = x * np.sqrt(qsca * np.pi)

        # 4pi Normalization
        elif norm_int == 2:
            norm = x * np.sqrt(qsca / 4)

        # Qext Normalization
        elif norm_int == 4:  # 4pi
            norm = x * np.sqrt(qsca * np.pi / qext)

    if norm is None:
        raise ValueError("norm-int must be in the range 0..6")

    return norm


def norm_string_to_integer(s):
    """
    Encode normalization choice as an integer.

    This is needed because these string operations cannot be
    done in a jitted function under numba.  We cannot use enums for
    the same reason!

    Args:
        s: string describing normalization desired.

    Returns:
        integer used in _mie_S1_S2() determine normalization
    """
    ii = None
    norm = s.lower()

    if norm in ["a", "albedo"]:
        ii = 0

    elif norm in ["1", "one", "unity"]:
        ii = 1

    elif norm in ["four_pi", "4pi"]:
        ii = 2

    elif norm in ["qsca", "scattering_efficiency"]:
        ii = 3

    elif norm in ["qext", "extinction_efficiency"]:
        ii = 4

    elif norm in ["bohren"]:
        ii = 5

    elif norm in ["wiscombe"]:
        ii = 6

    if ii is None:
        raise ValueError(
            "normalization must be one of 'albedo' (default), 'one'"
            "'4pi', 'qext', 'qsca', 'bohren', or 'wiscombe'"
        )
    return ii


@njit((complex128, float64, float64[:], int64), cache=True)
def _mie_S1_S2(m, x, mu, n_pole):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: array of angles, cos(theta), to calculate scattering amplitudes
        norm_int: integer describing type of normalization
        n_pole: return n_pole term from series (default=0 means include all terms)
        e_field: If True then Electric field (does not currently work)

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a, b = an_bn(m, x, 0)

    nangles = len(mu)
    S1 = np.zeros(nangles, dtype=np.complex128)
    S2 = np.zeros(nangles, dtype=np.complex128)

    nstop = len(a)
    for k in range(nangles):
        pi_nm2 = 0
        pi_nm1 = 1
        for n in range(1, nstop):
            tau_nm1 = n * mu[k] * pi_nm1 - (n + 1) * pi_nm2
            if n_pole in (0, n):
                S1[k] += (2 * n + 1) * (pi_nm1 * a[n - 1] + tau_nm1 * b[n - 1]) / (n + 1) / n
                S2[k] += (2 * n + 1) * (tau_nm1 * a[n - 1] + pi_nm1 * b[n - 1]) / (n + 1) / n
            temp = pi_nm1
            pi_nm1 = ((2 * n + 1) * mu[k] * pi_nm1 - (n + 1) * pi_nm2) / n
            pi_nm2 = temp

    return [S1, S2]


def mie_S1_S2(m, x, mu, norm="albedo", n_pole=0, field="Electric"):
    """
    Calculate the scattering amplitude functions for spheres.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: cos(theta) or array of angles [cos(theta_i)]
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)
        field: Electric of Magnetic field.  Only used if n_pole > 0

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    norm_int = norm_string_to_integer(norm)

    normalization = normalization_factor(m, x, norm_int)

    if np.isscalar(mu):
        mu_array = np.array([mu], dtype=float)
        S1, S2 = _mie_S1_S2(m, x, mu_array, n_pole)
    else:
        S1, S2 = _mie_S1_S2(m, x, mu, n_pole)

    S1 = np.conjugate(S1 / normalization)
    S2 = np.conjugate(S2 / normalization)

    if np.isscalar(mu):
        return S1[0], S2[0]

    return S1, S2


def mie_phase_matrix(m, x, mu, norm="albedo"):
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
        mu: the angles, cos(theta), for the phase scattering matrix
        norm: (optional) string describing scattering function normalization

    Returns:
        p: the phase scattering matrix [sr**(-1.0)]
    """
    s1, s2 = mie_S1_S2(m=m, x=x, mu=mu, norm=norm)

    mu = np.atleast_1d(mu)
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
        n_pole: return n_pole term from series (default=0 means include all terms)

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
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

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
    The intensity is normalized such that the integral of the unpolarized
    intensity over 4π steradians is equal to the single scattering albedo.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate intensities
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

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
        qext: the total extinction efficiency                  [-].
        qsca: the scattering efficiency                        [-].
        qback: the backscatter efficiency                      [-].
        g: the average cosine of the scattering phase function [-].
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
        n_env: real index of medium around sphere, optional
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


#
# @njit
# def spherical_bessel_jn_array(n, z):
#     """
#     Compute the spherical Bessel function of the first kind for array inputs.
#     Args:
#         n (np.ndarray): Array of orders of the spherical Bessel function.
#         z (float): Argument of the function.
#     Returns:
#         np.ndarray: Values of j_n(z) for each order in n.
#     """
#     if z == 0:
#         result = np.zeros_like(n, dtype=np.float64)
#         result[n == 0] = 1.0  # j_0(0) = 1; all other j_n(0) = 0
#         return result
#
#     return np.sqrt(np.pi / (2 * z)) * np.sin(z - n * np.pi / 2)
#
#
# @njit
# def spherical_bessel_yn_array(n, z):
#     """
#     Compute the spherical Bessel function of the second kind for array inputs.
#     Args:
#         n (np.ndarray): Array of orders of the spherical Bessel function.
#         z (float): Argument of the function.
#     Returns:
#         np.ndarray: Values of y_n(z) for each order in n.
#     """
#     if z == 0:
#         result = np.full_like(n, -np.inf, dtype=np.float64)
#         result[n != 0] = 0.0
#         return result
#
#     return np.sqrt(np.pi / (2 * z)) * np.cos(z - n * np.pi / 2)
#
#
# @njit((float64, float64, complex128, float64, float64[3], complex128[:,4]), cache=True)
# def E_field_spherical(d_sphere, lambda0, n_sphere, n_env, sph_point, abcd):
#     m = n_sphere / n_env
#     lam = lambda0 / n_env
#     k = 2 * np.pi / lam
#
#     r, theta, phi = sph_point
#     an, bn, cn, dn = abcd
#
#     n = np.arange(1, len(an) + 1)
#
#     # Compute spherical Hankel functions h_n^(1)
#     h_n = spherical_bessel_jn_array(n, k*r) + 1j * spherical_bessel_yn_array(n, k*r)
#
#     # Compute angular terms
#     Pn_theta = np.cos(theta) ** n
#     dPn_dtheta = -n * np.sin(theta) * np.cos(theta) ** (n - 1)
#
#     # Compute vector spherical harmonics components
#     if r > d_sphere/2:
#         E_r = np.sum((2 * n + 1) * (an + bn) * h_n * Pn_theta)
#         E_theta = np.sum((2 * n + 1) * (an - bn) * h_n * dPn_dtheta)
#     else:
#         E_r = np.sum((2 * n + 1) * (cn + dn) * h_n * Pn_theta)
#         E_theta = np.sum((2 * n + 1) * (cn - dn) * h_n * dPn_dtheta)
#
#     return np.array([E_r, E_theta, 0.0])
#
#
# @njit((float64, float64, complex128, float64, float64[3], complex128[:,4]), cache=True)
# def E_field(d_sphere, lambda0, n_sphere, n_env, xyz_point, abcd):
#
#     r = np.sqrt(xyz_point[0]**2 + xyz_point[1]**2 + xyz_point[2]**2)
#     theta = np.arccos(xyz_point[2] / r) if r != 0 else 0
#     phi = np.arctan2(xyz_point[1], xyz_point[0])
#
#     sph_point = np.array([r, theta, phi], dtype=float64)
#     E_spherical = scattered_E_field_spherical(d_sphere, lambda0, n_sphere, n_env, sph_point, abcd):
#
#     E_r, E_theta, E_phi = E_spherical
#     Ex = E_r * np.sin(theta) * np.cos(phi) + E_theta * np.cos(theta) * np.cos(phi)
#     Ey = E_r * np.sin(theta) * np.sin(phi) + E_theta * np.cos(theta) * np.sin(phi)
#     Ez = E_r * np.cos(theta) - E_theta * np.sin(theta)
#
#     return np.array([Ex, Ey, Ez])
#
#
# def plot_density(d, lambda0, n_sphere, n_env=1, nx=3, grid_size=100, projection="XZ"):
#     """
#     Create a 2D density plot of the electric field magnitude in a specified plane.
#
#     Args:
#         d (float): Diameter of the sphere.
#         lambda0 (float): Vacuum wavelength.
#         n_sphere (float): Refractive index of the sphere.
#         n_env (float): Refractive index of the surrounding environment.
#         grid_size (int): Number of points along each axis in the grid.
#         projection (str): Plane of projection ('XZ', 'YZ', or 'XY').
#
#     Returns:
#         None
#     """
#     m = n_sphere / n_env
#     lam = lambda0 / n_env
#     x = 2 * np.pi * (d / 2) / lam
#     k = 2 * np.pi / lam
#
#     an, bn = an_bn(m, x, 0)
#
#     ext = int(nx) * lam
#     extent = (-ext, ext, -ext, ext)
#     coords = np.linspace(-ext, ext, grid_size)
#     field_magnitude = np.zeros((grid_size, grid_size), dtype=np.float64)
#
#     rotation = -90
#     if projection == "XZ":
#         X, Z = np.meshgrid(coords, coords)
#         Y = np.zeros_like(X)
#         xlabel, ylabel = "z", "x"
#     elif projection == "YZ":
#         Y, Z = np.meshgrid(coords, coords)
#         X = np.zeros_like(Y)
#         xlabel, ylabel = "z", "y"
#     elif projection == "XY":
#         X, Y = np.meshgrid(coords, coords)
#         Z = np.zeros_like(X)
#         xlabel, ylabel = "x", "y"
#     else:
#         raise ValueError("Invalid projection. Choose from 'XZ', 'YZ', or 'XY'.")
#
#     positions = zip(X.ravel(), Y.ravel(), Z.ravel())
#     radius = d / 2
#     for idx, (x_pos, y_pos, z_pos) in enumerate(positions):
#         r = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
#         if r >= radius:  # Skip points inside the sphere
#             i, j = divmod(idx, grid_size)
#             Ex, Ey, Ez = scattered_E_field_cartesian(an, bn, k, x_pos, y_pos, z_pos)
#             field_magnitude[i, j] = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
#
#     if rotation != 0:
#         field_magnitude = np.rot90(field_magnitude, k=rotation // 90)
#
#     ticks = np.linspace(-ext, ext, 2 * int(nx) + 1)
#     tick_labels = [f"{int(t)} λ" if t != 0 else "0" for t in ticks]
#
#     plt.figure(figsize=(8, 6))
#     plt.imshow(field_magnitude, extent=extent, origin="lower", cmap="viridis", aspect="auto")
#     plt.colorbar(label="Electric Field Magnitude")
#
#     # Draw a circle representing the sphere
#     circle = plt.Circle((0, 0), radius / lam, color="white", fill=False, linewidth=2)
#     plt.gca().add_artist(circle)
#
#     plt.title(f"2D Density Plot of Electric Field in the {projection} Plane")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xticks(ticks=ticks, labels=tick_labels)
#     plt.yticks(ticks=ticks, labels=tick_labels)
#     plt.show()
