"""Independent reference implementations used to check miepython.

These are deliberately slow and literal: they follow published formulations
(Bohren & Huffman, pyscatt) using scipy Bessel functions rather than the
recurrences miepython uses, so they make useful cross-checks.  They used to be
copy-pasted into both halves of the test_jit_abcd/test_nojit_abcd pair; keeping
them here means one copy and no test module importing another.

None of these depend on MIEPYTHON_USE_JIT, so they are valid for either backend.
"""

import numpy as np
from scipy.special import jv, spherical_jn, spherical_yn, yv

from miepython.bessel import (
    d_riccati_bessel_h2,
    d_riccati_bessel_jn,
    riccati_bessel_h2,
    riccati_bessel_jn,
)

__all__ = (
    "pyscatt_cd",
    "pyscatt_ab",
    "bohren_ab",
    "slow_cn_dn",
    "basic_abcd",
)


def pyscatt_cd(m, x):
    #  http://pyscatt.readthedocs.io/en/latest/forward.html#pyscatt_cd
    """Compute pyscatt cd."""
    if m.imag < 0:
        m = np.conjugate(m)
    mx = m * x
    nmax = np.round(2 + x + 4 * (x ** (1 / 3)))
    nmx = np.round(max(nmax, np.abs(mx)) + 16)
    n = np.arange(1, int(nmax) + 1)
    nu = n + 0.5

    cnx = np.zeros(int(nmx), dtype=complex)

    for j in np.arange(nmx, 1, -1):
        cnx[int(j) - 2] = j - mx * mx / (cnx[int(j) - 1] + j)

    cnn = np.array([cnx[b] for b in range(0, len(n))])

    jnx = np.sqrt(np.pi / (2 * x)) * jv(nu, x)
    jnmx = np.sqrt((2 * mx) / np.pi) / jv(nu, mx)

    yx = np.sqrt(np.pi / (2 * x)) * yv(nu, x)
    hx = jnx + (1.0j) * yx

    b1x = np.append(np.sin(x) / x, jnx[0 : int(nmax) - 1])
    y1x = np.append(-np.cos(x) / x, yx[0 : int(nmax) - 1])

    hn1x = b1x + (1.0j) * y1x
    ax = x * b1x - n * jnx
    ahx = x * hn1x - n * hx

    numerator = jnx * ahx - hx * ax
    c_denominator = ahx - hx * cnn
    d_denominator = m * m * ahx - hx * cnn

    cn = jnmx * numerator / c_denominator
    dn = jnmx * m * numerator / d_denominator

    return cn, dn


def pyscatt_ab(m, x):
    #  http://pyscatt.readthedocs.io/en/latest/forward.html#Mie_ab
    """Compute pyscatt ab."""
    if m.imag < 0:
        m = np.conjugate(m)
    mx = m * x
    nmax = np.round(2 + x + 4 * (x ** (1 / 3)))
    nmx = np.round(max(nmax, np.abs(mx)) + 16)
    n = np.arange(1, nmax + 1)  #
    nu = n + 0.5  #

    sx = np.sqrt(0.5 * np.pi * x)

    px = sx * jv(nu, x)  #
    p1x = np.append(np.sin(x), px[0 : int(nmax) - 1])  #

    chx = -sx * yv(nu, x)  #
    ch1x = np.append(np.cos(x), chx[0 : int(nmax) - 1])  #

    gsx = px - (0 + 1j) * chx  #
    gs1x = p1x - (0 + 1j) * ch1x  #

    # B&H Equation 4.89
    Dn = np.zeros(int(nmx), dtype=complex)
    for i in range(int(nmx) - 1, 1, -1):
        Dn[i - 1] = (i / mx) - (1 / (Dn[i] + i / mx))

    D = Dn[1 : int(nmax) + 1]  # Dn(mx), drop terms beyond nMax
    da = D / m + n / x
    db = m * D + n / x

    an = (da * px - p1x) / (da * gsx - gs1x)
    bn = (db * px - p1x) / (db * gsx - gs1x)
    qsca = (2 / x / x) * np.sum((2 * n + 1) * (an.real**2 + an.imag**2 + bn.real**2 + bn.imag**2))

    return an, bn, qsca


def bohren_ab(m, x):
    """
    Bohren-Huffman Mie scattering calculations.

    Parameters:
        x (float): Size parameter (2 * pi * a / lambda).
        m (complex): Complex refractive index ratio of the sphere and medium.

    Returns:
        a (np.ndarray): Mie scattering coefficient 'a' for each order.
        b (np.ndarray): Mie scattering coefficient 'b' for each order.
    """
    if m.imag < 0:
        m = np.conjugate(m)
    mx = x * m
    nstop = int(x + 4 * (x ** (1 / 3)) + 2)

    # Logarithmic derivatives D_n
    nmx = int(max(nstop, np.abs(mx)) + 15)
    d = np.zeros(nmx, dtype=complex)
    for n in range(nmx - 1, 0, -1):
        d[n - 1] = ((n + 1) / mx) - (1.0 / (d[n] + (n + 1) / mx))

    # Riccati-Bessel functions
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)
    xi0 = complex(psi0, -chi0)
    xi1 = complex(psi1, -chi1)

    a = np.zeros(nstop, dtype=complex)
    b = np.zeros(nstop, dtype=complex)

    qsca = 0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        xi = (2 * n - 1) * xi1 / x - xi0

        a[n - 1] = ((d[n - 1] / m + n / x) * psi - psi1) / ((d[n - 1] / m + n / x) * xi - xi1)
        b[n - 1] = ((m * d[n - 1] + n / x) * psi - psi1) / ((m * d[n - 1] + n / x) * xi - xi1)

        psi0, psi1 = psi1, psi
        xi0, xi1 = xi1, xi
        qsca += (2 * n + 1) * (np.abs(a[n - 1]) ** 2 + np.abs(b[n - 1]) ** 2)

    return a, b, qsca * 2 / x**2


def slow_cn_dn(m, x, n_pole=0):
    """
    Calculate Mie coefficients c_n and d_n for the internal field of a sphere.

    This follows Bohren and Huffman notation

    Args:
        m (complex): Refractive index of the sphere relative to the surrounding medium.
        x (float): Size parameter of the sphere (2πr/λ).
        n_pole (int): Number of terms to calculate (n_pole).

    Returns:
        cn, dn: Arrays of c_n and d_n coefficients.
    """
    if n_pole == 0:
        n_stop = int(x + 4.05 * x**0.33333 + 2.0)
    else:
        n_stop = n_pole + 1

    n = np.arange(1, n_stop + 1)

    if m.imag < 0:
        m = np.conjugate(m)

    # Riccati-Bessel functions ψ_n(x) and ψ_n'(x)
    psi_n_x = x * spherical_jn(n, x)
    psi_n_prime_x = spherical_jn(n, x, derivative=True) * x
    psi_n_prime_x += spherical_jn(n, x)

    # Riccati-Bessel functions ψ_n(mx) and ψ_n'(mx)
    mx = m * x
    psi_n_mx = m * x * spherical_jn(n, mx)
    psi_n_prime_mx = spherical_jn(n, mx, derivative=True) * mx
    psi_n_prime_mx += spherical_jn(n, mx)

    # Riccati-Bessel functions ξ_n(x) and ξ_n'(x)
    xi_n_x = psi_n_x + 1j * x * spherical_yn(n, x)
    xi_n_prime_x = 1j * x * spherical_yn(n, x, derivative=True) + xi_n_x / x
    xi_n_prime_x += x * spherical_jn(n, x, derivative=True)

    # c_n and d_n have the same numerator
    numerator = m * psi_n_prime_x * xi_n_x - m * psi_n_x * xi_n_prime_x
    cn_denominator = m * psi_n_prime_mx * xi_n_x - psi_n_mx * xi_n_prime_x
    dn_denominator = psi_n_prime_mx * xi_n_x - m * psi_n_mx * xi_n_prime_x
    cn = numerator / cn_denominator
    dn = numerator / dn_denominator

    return cn, dn


def basic_abcd(m, x, terms=10):
    """Compute basic abcd."""
    n = np.arange(1, terms + 1)
    mx = m * x
    psi_n_x = riccati_bessel_jn(n, x)
    psi_n_prime_x = d_riccati_bessel_jn(n, x)
    psi_n_mx = riccati_bessel_jn(n, mx)
    psi_n_prime_mx = d_riccati_bessel_jn(n, mx)
    xi_n_x = riccati_bessel_h2(n, x)
    xi_n_prime_x = d_riccati_bessel_h2(n, x)

    a_numerator = m * psi_n_mx * psi_n_prime_x - psi_n_x * psi_n_prime_mx
    b_numerator = psi_n_mx * psi_n_prime_x - m * psi_n_x * psi_n_prime_mx
    c_numerator = m * psi_n_x * xi_n_prime_x - m * xi_n_x * psi_n_prime_x
    d_numerator = c_numerator

    a_denominator = m * psi_n_mx * xi_n_prime_x - xi_n_x * psi_n_prime_mx
    b_denominator = psi_n_mx * xi_n_prime_x - m * xi_n_x * psi_n_prime_mx
    c_denominator = b_denominator
    d_denominator = a_denominator

    a = a_numerator / a_denominator
    b = b_numerator / b_denominator
    c = c_numerator / c_denominator
    d = d_numerator / d_denominator

    a = np.conjugate(a)
    b = np.conjugate(b)
    c = np.conjugate(c)
    d = np.conjugate(d)
    return a, b, c, d
