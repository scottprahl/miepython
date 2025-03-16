"""
Low-level Mie calculations that use numba.
"""

import numpy as np
from numba import njit, complex128, float64, int64

__all__ = (
    "_D_calc",
    "_an_bn",
    "_cn_dn",
    "_pi_tau",
    "_S1_S2",
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
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
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
        D: gets filled with ψ_k'(z)/ψ_k(z) for k=0 to N-1
    """
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z


@njit((complex128, float64, int64), cache=True)
def _D_calc(m, x, N):
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
    mx = np.complex128(m * x)  # ensure complex

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n**2:
        _D_downwards(mx, N, D)
    else:
        _D_upwards(mx, N, D)
    return D[1:]


@njit((complex128, float64, int64), cache=True)
def _an_bn(m, x, n_pole):
    """
    Compute arrays of Mie coefficients a_n and b_n for a sphere.

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
    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    a = np.zeros(nstop, dtype=np.complex128)
    b = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return a, b

    psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
    psi_n = psi_nm1 / x - np.cos(x)
    xi_nm1 = np.complex128(psi_nm1 + 1j * np.cos(x))
    xi_n = np.complex128(psi_n + 1j * (np.cos(x) / x + np.sin(x)))

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
def _cn_dn(m, x, n_pole):
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
    m = np.where(np.imag(m) > 0, np.conj(m), m)
    mx = m * x

    if n_pole == 0:
        nstop = int(x + 4.05 * x**0.33333 + 2.0) + 1
    else:
        nstop = n_pole + 1

    c = np.zeros(nstop, dtype=np.complex128)
    d = np.zeros(nstop, dtype=np.complex128)
    if x <= 0:
        return c, d

    # no need to calculate anything when sphere is perfectly conducting
    if m.real > 0.0 and not np.isinf(m.real) or not np.isinf(m.imag):
        psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
        psi_n = psi_nm1 / x - np.cos(x)

        psi_nm1_mx = np.sin(mx)  # nm1 = n-1 = 0
        psi_n_mx = psi_nm1_mx / mx - np.cos(mx)

        xi_nm1 = np.complex128(psi_nm1 + 1j * np.cos(x))
        xi_n = np.complex128(psi_n + 1j * (np.cos(x) / x + np.sin(x)))

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


@njit((float64, float64[:], float64[:]), cache=True)
def _pi_tau(mu, pi, tau):
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


@njit((complex128, float64, float64[:], int64), cache=True)
def _S1_S2(m, x, mu, n_pole):
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
    a, b = _an_bn(m, x, 0)
    N = len(a)
    pi = np.zeros(N)
    tau = np.zeros(N)
    n = np.arange(1, N + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    nangles = len(mu)
    S1 = np.zeros(nangles, dtype=np.complex128)
    S2 = np.zeros(nangles, dtype=np.complex128)

    for k in range(nangles):
        _pi_tau(mu[k], pi, tau)
        if n_pole == 0:
            S1[k] = np.sum(scale * (pi * a + tau * b))
            S2[k] = np.sum(scale * (tau * a + pi * b))
        else:
            S1[k] = scale[n_pole] * (pi[n_pole] * a[n_pole] + tau[n_pole] * b[n_pole])
            S2[k] = scale[n_pole] * (tau[n_pole] * a[n_pole] + pi[n_pole] * b[n_pole])

    return [np.conjugate(S1), np.conjugate(S2)]
