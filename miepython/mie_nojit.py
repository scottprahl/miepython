"""
Low-level Mie calculations that do not use numba.
"""

from functools import lru_cache

import numpy as np

__all__ = (
    "_D_calc_py",
    "_an_bn_py",
    "_cn_dn_py",
    "_pi_tau_py",
    "_S1_S2_py",
    "_single_sphere_py",
    "_small_conducting_sphere_py",
    "_small_sphere_py",
)


@lru_cache(maxsize=128)
def _series_scale_factors(n_terms):
    """Return cached per-order scale factors for Mie series summations."""
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    scale = (2.0 * n + 1.0) / ((n + 1.0) * n)
    scale.setflags(write=False)
    return scale


@lru_cache(maxsize=128)
def _single_sphere_factors(n_terms):
    """Return cached per-order factors used by ``_single_sphere_py``."""
    n_int = np.arange(1, n_terms + 1, dtype=np.int64)
    n = n_int.astype(np.float64)
    cn = 2.0 * n + 1.0
    alt = np.where((n_int % 2) == 0, 1.0, -1.0)  # (-1)^n with n starting at 1
    c1n = n * (n + 2.0) / (n + 1.0)
    c2n = cn / n / (n + 1.0)
    cn.setflags(write=False)
    alt.setflags(write=False)
    c1n.setflags(write=False)
    c2n.setflags(write=False)
    return cn, alt, c1n, c2n


def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.

    D_n(z) = d[log psi_n(z)] = psi_n'(z)/psi_n(z)

    This returns the logarithmic derivative of the Ricatti-Bessel function of order N
    with argument z using the continued fraction technique of Lentz, Appl. Opt., 15,
    668-671, (1976).

    Args:
        z: function argument
        N: order of Ricatti-Bessel function

    Returns:
        logarithmic derivative Dn(z)
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


def _D_calc_py(m, x, N):
    """
    Compute the logarithmic derivative of ψ_n(z) using the best method.

    D_n(z) = d[log ψ_n(z)] = ψ_n'(z)/ψ_n(z)

    here ψ_n(z) is the Riccati-Bessel function of the first kind ψ_n(z)=z*j_n(z)
    were j_n(z) is the spherical Bessel function of order n.

    The zero-based array, D[:], is shifted so that D[0] = D₁(z) = ψ₁'(z)/ψ₁(z)

    Args:
        m: the np.complex128 index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function

    Returns:
        Array of logarithmic derivatives D_k(z) for k=1 to N-1.
    """
    n = m.real
    kappa = np.abs(m.imag)
    D = np.zeros(N + 1, dtype=np.complex128)
    mx = np.complex128(m * x)  # ensure np.complex128

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n**2:
        _D_downwards(mx, N, D)
    else:
        _D_upwards(mx, N, D)
    return D[1:]


def _an_bn_py(m, x, n_pole=0):
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
        m: the np.complex128 index of refraction of the sphere
        x: the size parameter of the sphere
        n_pole: the number of An and Bn terms (0 does autosizing)

    Returns:
        a, b: arrays of Mie coefficents An and Bn
    """
    if m.imag > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    a = np.zeros(nstop, dtype=np.complex128)
    b = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return a, b

    inv_x = 1.0 / x
    psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
    psi_n = psi_nm1 * inv_x - np.cos(x)
    xi_nm1 = np.complex128(psi_nm1 + 1j * np.cos(x))
    xi_n = np.complex128(psi_n + 1j * (np.cos(x) * inv_x + np.sin(x)))

    if m.real > 0.0:
        D = _D_calc_py(m, x, nstop + 1)

        for n in range(1, nstop):
            n_over_x = n * inv_x
            temp = D[n - 1] / m + n_over_x
            a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            temp = D[n - 1] * m + n_over_x
            b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
            two_np1_over_x = (2 * n + 1) * inv_x
            psi = two_np1_over_x * psi_n - psi_nm1
            xi = two_np1_over_x * xi_n - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = psi

    else:
        for n in range(1, nstop):
            n_over_x = n * inv_x
            a[n - 1] = (n_over_x * psi_n - psi_nm1) / (n_over_x * xi_n - xi_nm1)
            b[n - 1] = psi_n / xi_n
            xi = (2 * n + 1) * inv_x * xi_n - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi
            psi_nm1 = psi_n
            psi_n = xi_n.real

    if n_pole != 0:
        a = a[:-1]
        b = b[:-1]

    return np.conjugate(a), np.conjugate(b)


def _cn_dn_py(m, x, n_pole):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    Args:
        m (np.complex128): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        (np.ndarray, np.ndarray): Arrays of c_n and d_n coefficients.
    """
    # ensure imaginary part of refractive index is negative
    if m.imag > 0:
        m = np.conj(m)
    mx = m * x

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    c = np.zeros(nstop, dtype=np.complex128)
    d = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return c, d

    inv_x = 1.0 / x
    # no need to calculate anything when sphere is perfectly conducting
    if m.real > 0.0 and not np.isinf(m.real) or not np.isinf(m.imag):
        psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
        psi_n = psi_nm1 * inv_x - np.cos(x)

        inv_mx = 1.0 / mx
        psi_nm1_mx = np.sin(mx)  # nm1 = n-1 = 0
        psi_n_mx = psi_nm1_mx * inv_mx - np.cos(mx)

        xi_nm1 = np.complex128(psi_nm1 + 1j * np.cos(x))
        xi_n = np.complex128(psi_n + 1j * (np.cos(x) * inv_x + np.sin(x)))

        Dmx = _D_calc_py(np.complex128(m), x, nstop + 1)
        Dx = _D_calc_py(np.complex128(1), x, nstop + 1)

        for n in range(1, nstop + 1):
            n_over_x = n * inv_x
            common = (psi_n / psi_n_mx) * ((Dx[n - 1] + n_over_x) * xi_n - xi_nm1)

            c[n - 1] = m * common / ((m * Dmx[n - 1] + n_over_x) * xi_n - xi_nm1)
            d[n - 1] = common / ((Dmx[n - 1] / m + n_over_x) * xi_n - xi_nm1)

            two_np1 = 2 * n + 1
            psi = two_np1 * inv_x * psi_n - psi_nm1
            psi_nm1 = psi_n
            psi_n = psi

            psi_mx = two_np1 * inv_mx * psi_n_mx - psi_nm1_mx
            psi_nm1_mx = psi_n_mx
            psi_n_mx = psi_mx

            xi = two_np1 * inv_x * xi_n - xi_nm1
            xi_nm1 = xi_n
            xi_n = xi

    if n_pole != 0:
        c = c[:-1]
        d = d[:-1]
    return np.conjugate(c), np.conjugate(d)


def _pi_tau_py(mu, pi, tau):
    """
    Compute the Mie scattering functions π_n and τ_n for given cosine angles.

    This function fills the pre-allocated arrays `pi` and `tau` with values
    of the Mie scattering functions for a given `mu = cos𝜃`. The function
    uses the recurrence relations for the associated Legendre polynomials
    of the first kind P_n^1. The recurrence relations ensure numerical stability
    and avoids calling scipi.special.lpmv(1, n, cos𝜃) for each n.

    `pi` and `tau` are **zero-based** arrays and therefore

    `pi[n-1]` = 𝜋_n(cos𝜃) = P_n^1(cos𝜃) / sin𝜃

    `tau[n-1]` = 𝜏_n(cos𝜃) = d/d𝜃 P_n^1(cos𝜃)`.

    Args:
        mu (float): The cosine of the scattering angle, `cos(𝜃)`.
        pi (numpy.ndarray): A pre-allocated array to store `pi_n` values.
        tau (numpy.ndarray): A pre-allocated array to store `tau_n` values.

    Returns:
        nothing.  pi and tau are modified
    """
    n_terms = len(pi)
    pi_nm2 = 0
    pi[0] = 1
    for n in range(1, n_terms):
        tau[n - 1] = n * mu * pi[n - 1] - (n + 1) * pi_nm2
        temp = pi[n - 1]
        pi[n] = ((2 * n + 1) * mu * temp - (n + 1) * pi_nm2) / n
        pi_nm2 = temp


def _S1_S2_py(m, x, mu, n_pole):
    """
    Calculate the scattering amplitude functions for spheres.

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: array of angles, cos(theta), to calculate scattering amplitudes
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    a, b = _an_bn_py(m, x, 0)
    N = len(a)
    pi = np.zeros(N)
    tau = np.zeros(N)
    scale = _series_scale_factors(N)
    scale_a = scale * a
    scale_b = scale * b

    nangles = len(mu)
    S1 = np.zeros(nangles, dtype=np.complex128)
    S2 = np.zeros(nangles, dtype=np.complex128)

    for k in range(nangles):
        _pi_tau_py(mu[k], pi, tau)
        if n_pole == 0:
            S1[k] = np.dot(pi, scale_a) + np.dot(tau, scale_b)
            S2[k] = np.dot(tau, scale_a) + np.dot(pi, scale_b)
        else:
            S1[k] = scale[n_pole] * (pi[n_pole] * a[n_pole] + tau[n_pole] * b[n_pole])
            S2[k] = scale[n_pole] * (tau[n_pole] * a[n_pole] + pi[n_pole] * b[n_pole])

    return np.conjugate(S1), np.conjugate(S2)


def _small_conducting_sphere_py(_m, x):
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

    qsca = x**4 * (6 * np.abs(ahat1) ** 2 + 6 * np.abs(bhat1) ** 2 + 10 * np.abs(ahat2) ** 2 + 10 * np.abs(bhat2) ** 2)
    qext = qsca
    g = ahat1.imag * (ahat2.imag + bhat1.imag)
    g += bhat2.imag * (5.0 / 9.0 * ahat2.imag + bhat1.imag)
    g += ahat1.real * bhat1.real
    g *= 6 * x**4 / qsca

    qback = 9 * x**4 * np.abs(ahat1 - bhat1 - 5 / 3 * (ahat2 - bhat2)) ** 2

    return qext, qsca, qback, g


def _small_sphere_py(m, x):
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


def _single_sphere_py(m, x, n_pole, e_field):
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
    _ = e_field  # currently unused in scalar aggregate efficiencies

    # case when sphere matches its environment
    if abs(m.real - 1) <= 1e-8 and abs(m.imag) < 1e-8:
        return 0, 0, 0, 0

    # small conducting spheres --- see Wiscombe
    if m.real == 0 and x < 0.1 and n_pole == 0:
        return _small_conducting_sphere_py(m, x)

    if m.real > 0.0 and np.abs(m) * x < 0.1 and n_pole == 0:
        return _small_sphere_py(m, x)

    # sometimes m=0 is used to signal perfectly conducting sphere
    if abs(m.real) < 1e-8 and abs(m.imag) < 1e-8:
        m = 1 - 10000j

    a, b = _an_bn_py(m, x, n_pole)
    x2 = x * x

    if n_pole == 0:
        n_terms = len(a)
        cn, alt, c1n, c2n = _single_sphere_factors(n_terms)
        a_re = a.real
        b_re = b.real
        a_abs2 = a_re * a_re + a.imag * a.imag
        b_abs2 = b_re * b_re + b.imag * b.imag

        qext = 2.0 * np.dot(cn, a_re + b_re) / x2

        if m.imag == 0:
            qsca = qext
        else:
            qsca = 2.0 * np.dot(cn, a_abs2 + b_abs2) / x2

        qback = np.abs(np.dot(alt * cn, a - b)) ** 2 / x2

        asy1 = c1n[:-1] * (a[:-1] * a[1:].conjugate() + b[:-1] * b[1:].conjugate()).real
        asy2 = c2n[:-1] * (a[:-1] * b[:-1].conjugate()).real
        g = 4.0 * np.sum(asy1 + asy2) / qsca / x2

    else:
        cn = 2.0 * n_pole + 1
        qback = np.abs((-1) ** n_pole * cn * (a[-1] - b[-1])) ** 2 / x2
        qext = 2.0 * cn * (a[-1].real + b[-1].real) / x2
        qsca = qext
        if m.imag < 0:
            qsca = 2.0 * cn * (np.abs(a[-1]) ** 2 + np.abs(b[-1]) ** 2) / x2
        g = None

    return qext, qsca, qback, g
