#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import unittest
import numpy as np
from scipy.special import spherical_jn, spherical_yn, jv, yv

import miepython as mie
from miepython.bessel import *
from miepython.util import cs


def pyscatt_cd(m, x):
    #  http://pyscatt.readthedocs.io/en/latest/forward.html#pyscatt_cd
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
        n_stop = int(x + 4.05 * x**0.33333 + 2.0) + 1
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


class Test1(unittest.TestCase):
    def test_09an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        m = 8.9 - 0.69j
        xx = np.array([0.385, 0.390, 0.395])
        a1 = np.array([0.0024 + 0.0410j, 0.0027 + 0.0428j, 0.0030 + 0.0447j])
        b1 = np.array([0.0258 - 0.0359j, 0.0231 - 0.0361j, 0.0208 - 0.0361j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        for n, x in enumerate(xx):
            a, b = mie.an_bn(m, x, 0)
            ab, bb, qsca_b = bohren_ab(m, x)
            ap, bp, qsca_p = pyscatt_ab(m, x)
            at, bt, _, _ = basic_abcd(m, x, 4)
            qext, qsca, qback, g = mie.efficiencies_mx(m, x)

            print("m=%s, x=%.4f" % (cs(m), x))
            print("Qsca %.5f %.5f" % (qsca, qsca_b))
            print("a")
            print("lowan    ", cs(a1[n]))
            print("miepython", cs(a[0]))
            print("bohren   ", cs(ab[0]))
            print("pyscat   ", cs(ap[0]))
            print("formulas ", cs(at[0]))
            print("b")
            print("lowan    ", cs(b1[n]))
            print("miepython", cs(b[0]))
            print("bohren   ", cs(bb[0]))
            print("pyscat   ", cs(bp[0]))
            print("formulas ", cs(bt[0]))

            self.assertAlmostEqual(a[0].real, a1[n].real, delta=1e-3)
            self.assertAlmostEqual(a[0].imag, a1[n].imag, delta=1e-3)
            self.assertAlmostEqual(b[0].real, b1[n].real, delta=1e-3)
            self.assertAlmostEqual(b[0].imag, b1[n].imag, delta=1e-3)
            self.assertAlmostEqual(at[0].real, a1[n].real, delta=1e-3)
            self.assertAlmostEqual(at[0].imag, a1[n].imag, delta=1e-3)
            self.assertAlmostEqual(bt[0].real, b1[n].real, delta=1e-3)
            self.assertAlmostEqual(bt[0].imag, b1[n].imag, delta=1e-3)

    def test_08an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        m = 3.41 - 1.94j
        x = 2
        qext_lowan = 2.952
        qsca_lowan = 1.698
        a1 = np.array([0.4559 + 0.2804j, 0.4197 + 0.2399j, 0.0332 + 0.0727j, 0.0010 + 0.0045j])
        b1 = np.array([0.4985 - 0.3538j, 0.1114 - 0.1633j, 0.0183 - 0.0214j, 0.0015 - 0.0010j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print("Qsca %.5f %.5f %.5f %.5f" % (qsca_lowan, qsca, qsca_b, qsca_p))
        print("a")
        print("lowan    ", cs(a1))
        print("miepython", cs(a))
        print("bohren   ", cs(ab))
        print("pyscat   ", cs(ap))
        print("formulas ", cs(at))
        print("b")
        print("lowan    ", cs(b1))
        print("miepython", cs(b))
        print("bohren   ", cs(bb))
        print("pyscat   ", cs(bp))
        print("formulas ", cs(bt))

        self.assertAlmostEqual(qsca, qsca_lowan, delta=1e-3)
        self.assertAlmostEqual(qext, qext_lowan, delta=1e-3)
        self.assertAlmostEqual(at[0].real, a1[0].real, delta=1e-3)
        self.assertAlmostEqual(at[0].imag, a1[0].imag, delta=1e-3)
        self.assertAlmostEqual(bt[0].real, b1[0].real, delta=1e-3)
        self.assertAlmostEqual(bt[0].imag, b1[0].imag, delta=1e-3)


class Test_NonAbsorbing_An_and_Bn(unittest.TestCase):
    def test_01an_bn(self):
        m = 1.5
        x = 1
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        print("m=%s, x=%.4f" % (cs(m), x))
        print("a")
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print(cs(at))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        print(cs(bt))
        for n in range(3):
            self.assertAlmostEqual(a[0].real, at[0].real, delta=0.00000001)
            self.assertAlmostEqual(a[0].imag, at[0].imag, delta=0.00000001)
            self.assertAlmostEqual(b[0].real, bt[0].real, delta=0.00000001)
            self.assertAlmostEqual(b[0].imag, bt[0].imag, delta=0.00000001)

    def test_02an_bn(self):
        m = 1.5
        x = 0.01
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        print("a m=%s, x=%.4f" % (cs(m), x))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        for n in range(3):
            self.assertAlmostEqual(a[0].real, at[0].real, delta=0.00000001)
            self.assertAlmostEqual(a[0].imag, at[0].imag, delta=0.00000001)
            self.assertAlmostEqual(b[0].real, bt[0].real, delta=0.00000001)
            self.assertAlmostEqual(b[0].imag, bt[0].imag, delta=0.00000001)

    def test_03an_bn(self):
        m = 1.5
        x = 10
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)

        print("a m=%s, x=%.4f" % (cs(m), x))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))

        for n in range(3):
            self.assertAlmostEqual(a[0].real, ab[0].real, delta=0.00000001)
            self.assertAlmostEqual(a[0].imag, ab[0].imag, delta=0.00000001)
            self.assertAlmostEqual(b[0].real, bb[0].real, delta=0.00000001)
            self.assertAlmostEqual(b[0].imag, bb[0].imag, delta=0.00000001)

    def test_05an_bn(self):
        m = 4.0 / 3.0
        x = 50
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        print("m=%s, x=%.4f" % (cs(m), x))
        print("a")
        print(cs(0.5311058892948411929 - 0.4990314856310943073j))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print(cs(at))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        print(cs(bt))
        self.assertAlmostEqual(a[0].real, 0.5311058892948411929, delta=0.00000001)
        self.assertAlmostEqual(a[0].imag, -0.4990314856310943073, delta=0.00000001)


class Test_Weak_Absorbing_An_and_Bn(unittest.TestCase):

    def test_00an_bn(self):
        """See table 4 in Bohren and Huffman."""
        ab = np.zeros(5, dtype=complex)
        ab[0] = 5.1631e-1 - 4.9973e-1j
        ab[1] = 3.4192e-1 - 4.7435e-1j
        ab[2] = 4.8467e-2 - 2.1475e-1j
        ab[3] = 1.0346e-3 - 3.2148e-2j
        ab[4] = 9.0375e-6 - 3.0062e-3j

        bb = np.zeros(5, dtype=complex)
        bb[0] = 7.3767e-1 - 4.3990e-1j
        bb[1] = 4.0079e-1 - 4.9006e-1j
        bb[2] = 9.3553e-3 - 9.6269e-2j
        bb[3] = 6.8810e-5 - 8.2949e-3j
        bb[4] = 2.8309e-7 - 5.3204e-4j

        m = 1.33 - 1e-8j
        x = 3
        a, b = mie.coefficients(m, x)

        print("a m=%s, x=%.4f" % (cs(m, 8), x))
        print(cs(a))
        print(cs(ab))
        print("b")
        print(cs(b))
        print(cs(bb))

        for n in range(5):
            self.assertAlmostEqual(ab[n].real, a[n].real, delta=1e-5)
            self.assertAlmostEqual(ab[n].imag, a[n].imag, delta=1e-5)
            self.assertAlmostEqual(bb[n].real, b[n].real, delta=1e-5)
            self.assertAlmostEqual(bb[n].imag, b[n].imag, delta=1e-5)

    def test_01an_bn(self):
        # Wiscombe MIEV0 Test Case 9
        m = 1.330 - 0.00001j
        x = 1
        qsca_wisc = 0.093923
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print("Qsca %.5f %.5f %.5f %.5f" % (qsca_wisc, qsca, qsca_b, qsca_p))
        print("a")
        print(cs(a, 10))
        print(cs(ab, 10))
        print(cs(ap, 10))
        print(cs(at, 10))
        print("b")
        print(cs(b, 10))
        print(cs(bb, 10))
        print(cs(bp, 10))
        print(cs(bt, 10))

        self.assertAlmostEqual(qsca, qsca_wisc, delta=1e-5)
        for n in range(len(a)):
            self.assertAlmostEqual(a[n].real, at[n].real, delta=0.00001)
            self.assertAlmostEqual(a[n].imag, at[n].imag, delta=0.00001)
            self.assertAlmostEqual(b[n].real, bt[n].real, delta=0.00001)
            self.assertAlmostEqual(b[n].imag, bt[n].imag, delta=0.00001)


class Test_Medium_Absorbing_An_and_Bn(unittest.TestCase):

    def test_01an_bn(self):
        # Wiscombe MIEV0 Test Case 12
        m = 1.5 - 1.0j
        x = 0.055
        qext_wisc = 0.101491
        qsca_wisc = 0.000011
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print("Qsca %.5f %.5f %.5f %.5f" % (qsca_wisc, qsca, qsca_b, qsca_p))
        print("a")
        print(cs(a, 10))
        print(cs(ab, 10))
        print(cs(ap, 10))
        print(cs(at, 10))
        print("b")
        print(cs(b, 10))
        print(cs(bb, 10))
        print(cs(bp, 10))
        print(cs(bt, 10))

        self.assertAlmostEqual(qsca, qsca_wisc, delta=1e-5)
        self.assertAlmostEqual(qext, qext_wisc, delta=1e-5)
        for n in range(len(a)):
            self.assertAlmostEqual(a[n].real, at[n].real, delta=0.00000001)
            self.assertAlmostEqual(a[n].imag, at[n].imag, delta=0.00000001)
            self.assertAlmostEqual(b[n].real, bt[n].real, delta=0.00000001)
            self.assertAlmostEqual(b[n].imag, bt[n].imag, delta=0.00000001)

    def test_09an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        m = 8.9 - 0.69j
        xx = np.array([0.385, 0.390, 0.395])
        a1 = np.array([0.0024 + 0.0410j, 0.0027 + 0.0428j, 0.0030 + 0.0447j])
        b1 = np.array([0.0258 - 0.0359j, 0.0231 - 0.0361j, 0.0208 - 0.0361j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        for n, x in enumerate(xx):
            a, b = mie.an_bn(m, x, 0)
            ab, bb, qsca_b = bohren_ab(m, x)
            ap, bp, qsca_p = pyscatt_ab(m, x)
            at, bt, _, _ = basic_abcd(m, x, len(a))
            qext, qsca, qback, g = mie.efficiencies_mx(m, x)

            print("m=%s, x=%.4f" % (cs(m), x))
            print("Qsca %.5f %.5f" % (qsca, qsca_b))
            print("a")
            print(cs(a1[n]))
            print(cs(a))
            print(cs(ab[0]))
            print(cs(ap[0]))
            print(cs(at[0]))
            print("b")
            print(cs(b1[n]))
            print(cs(b))
            print(cs(bb[0]))
            print(cs(bp[0]))
            print(cs(bt[0]))

            self.assertAlmostEqual(a[0].real, a1[n].real, delta=1e-3)
            self.assertAlmostEqual(a[0].imag, a1[n].imag, delta=1e-3)
            self.assertAlmostEqual(b[0].real, b1[n].real, delta=1e-3)
            self.assertAlmostEqual(b[0].imag, b1[n].imag, delta=1e-3)
            self.assertAlmostEqual(at[0].real, a1[n].real, delta=1e-3)
            self.assertAlmostEqual(at[0].imag, a1[n].imag, delta=1e-3)
            self.assertAlmostEqual(bt[0].real, b1[n].real, delta=1e-3)
            self.assertAlmostEqual(bt[0].imag, b1[n].imag, delta=1e-3)

    def test_08an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        m = 3.41 - 1.94j
        x = 2
        qext_lowan = 2.952
        qsca_lowan = 1.698
        a1 = np.array([0.4559 + 0.2804j, 0.4197 + 0.2399j, 0.0332 + 0.0727j, 0.0010 + 0.0045j])
        b1 = np.array([0.4985 - 0.3538j, 0.1114 - 0.1633j, 0.0183 - 0.0214j, 0.0015 - 0.0010j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print("Qsca %.5f %.5f %.5f %.5f" % (qsca_lowan, qsca, qsca_b, qsca_p))
        print("a")
        print(cs(a1))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print(cs(at))
        print("b")
        print(cs(b1))
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        print(cs(bt))

        self.assertAlmostEqual(qsca, qsca_lowan, delta=1e-3)
        self.assertAlmostEqual(qext, qext_lowan, delta=1e-3)
        self.assertAlmostEqual(a[0].real, a1[0].real, delta=1e-3)
        self.assertAlmostEqual(a[0].imag, a1[0].imag, delta=1e-3)
        self.assertAlmostEqual(b[0].real, b1[0].real, delta=1e-3)
        self.assertAlmostEqual(b[0].imag, b1[0].imag, delta=1e-3)

    def test_10an_bn(self):
        # From van de Hulst Table 28
        m = 3.41 - 1.94j
        x = 1.3
        qext_vdh = 3.053
        qsca_vdh = 1.669
        a1 = np.array([0.5538 + 0.2475j, 0.0468 + 0.1050j, 0.0008 + 0.0044j])
        b1 = np.array([0.1780 - 0.2153j, 0.0270 - 0.0228j, 0.0013 - 0.0003j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print("Qext %.5f %.5f" % (qext_vdh, qext))
        print("Qext %.5f %.5f %.5f" % (qsca_vdh, qsca, qsca_b))
        print("a")
        print(cs(a1))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print(cs(at))
        print("b")
        print(cs(b1))
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        print(cs(bt))

        for n in range(3):
            self.assertAlmostEqual(a[n].real, a1[n].real, delta=1e-3)
            self.assertAlmostEqual(a[n].imag, a1[n].imag, delta=1e-3)
            self.assertAlmostEqual(b[n].real, b1[n].real, delta=1e-3)
            self.assertAlmostEqual(b[n].imag, b1[n].imag, delta=1e-3)


class Test_Strong_Absorbing_An_and_Bn(unittest.TestCase):
    def test_07an_bn(self):
        m = 1.1 - 25j
        x = 2
        a, b = mie.an_bn(m, x, 0)
        self.assertAlmostEqual(a[1].real, 0.324433578437, delta=0.0001)
        self.assertAlmostEqual(a[1].imag, -0.465627763266, delta=0.0001)
        self.assertAlmostEqual(b[1].real, 0.060464399088, delta=0.0001)
        self.assertAlmostEqual(b[1].imag, 0.236805417045, delta=0.0001)


class TestCnDn(unittest.TestCase):
    """
    Test cn and dn against the results from pyscatt routine..
    """

    def test_00_cn_dn(self):
        m = 1.5 - 1j
        x = 2.5
        c, d = mie.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        at, bt, ct, dt = basic_abcd(m, x)
        cp, dp = pyscatt_cd(m, x)

        print("c")
        print("miepython", cs(c))
        print("pyscat   ", cs(cp))
        print("formulas ", cs(ct))
        print("slow     ", cs(cx))
        print("d")
        print("miepython", cs(d))
        print("pyscat   ", cs(dp))
        print("formulas ", cs(dt))
        print("slow     ", cs(dx))

        for i in range(5):
            self.assertAlmostEqual(c[i], cx[i], delta=1e-6)
            self.assertAlmostEqual(d[i], dx[i], delta=1e-6)

    def test_02_cn_dn(self):
        m = 1.5 - 0.1j
        x = 1
        c, d = mie.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        at, bt, ct, dt = basic_abcd(m, x)
        cp, dp = pyscatt_cd(m, x)

        print("c")
        print("miepython", cs(c))
        print("pyscat   ", cs(cp))
        print("formulas ", cs(ct))
        print("slow     ", cs(cx))
        print("d")
        print("miepython", cs(d))
        print("pyscat   ", cs(dp))
        print("formulas ", cs(dt))
        print("slow     ", cs(dx))

        for i in range(5):
            self.assertAlmostEqual(c[i], cx[i], delta=1e-6)
            self.assertAlmostEqual(d[i], dx[i], delta=1e-6)

    def test_03_cn_dn(self):
        m = 1.5 - 1j
        x = 1
        c, d = mie.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        at, bt, ct, dt = basic_abcd(m, x)
        cp, dp = pyscatt_cd(m, x)

        print("c")
        print("miepython", cs(c))
        print("pyscat   ", cs(cp))
        print("formulas ", cs(ct))
        print("slow     ", cs(cx))
        print("d")
        print("miepython", cs(d))
        print("pyscat   ", cs(dp))
        print("formulas ", cs(dt))
        print("slow     ", cs(dx))

        for i in range(5):
            self.assertAlmostEqual(c[i], cx[i], delta=1e-6)
            self.assertAlmostEqual(d[i], dx[i], delta=1e-6)


class TestAnBnCnDnLengths(unittest.TestCase):
    """
    Unit tests for Mie coefficients cn and dn.
    """

    def test_lengths_all(self):
        """
        Test that the lengths of the returned arrays match the expected lengths.
        """
        test_cases = [
            (1.5, 2.5, 0),
            (1.33 - 0.05j, 1.0, 0),
        ]

        for m, x, n_pole in test_cases:
            a, b = mie.an_bn(m, x, n_pole)
            c, d = mie.cn_dn(m, x, n_pole)

            expected = int(x + 4.05 * x**0.33333 + 2.0) + 1

            self.assertEqual(len(a), expected, "Length mismatch for a_n with n_pole=0")
            self.assertEqual(len(b), expected, "Length mismatch for b_n with n_pole=0")
            self.assertEqual(len(c), expected, "Length mismatch for c_n with n_pole=0")
            self.assertEqual(len(d), expected, "Length mismatch for d_n with n_pole=0")

        for m, x, n_pole in test_cases:
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)

            expected = int(x + 4.05 * x**0.33333 + 2.0) + 1

            self.assertEqual(len(a), expected, "Length mismatch for a_n with n_pole=0")
            self.assertEqual(len(b), expected, "Length mismatch for b_n with n_pole=0")
            self.assertEqual(len(c), expected, "Length mismatch for c_n with n_pole=0")
            self.assertEqual(len(d), expected, "Length mismatch for d_n with n_pole=0")

    def test_lengths_npole_scalar(self):
        """
        Test that scalars returned when we pass scalars
        """
        test_cases = [
            (1.5, 2.5, 1),
            (2.0 - 0.2j, 5.0, 2),
        ]

        for m, x, n_pole in test_cases:
            a, b = mie.an_bn(m, x, n_pole)
            c, d = mie.cn_dn(m, x, n_pole)
            for var, name in zip([a, b, c, d], "abcd"):
                self.assertTrue(np.isscalar(var), f"{name} is not a scalar, got {type(var)}")

        for m, x, n_pole in test_cases:
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)
            for var, name in zip([a, b, c, d], "abcd"):
                self.assertTrue(np.isscalar(var), f"{name} is not a scalar, got {type(var)}")

    def test_lengths_npole_arrays(self):
        """
        Test that returned lengths match input arrays.
        """
        xarr = np.array([1, 2])
        marr = np.array([2.0 - 0.2j, 2.0 - 0.1j])
        test_cases = [
            (marr, 2.5, 1),
            (marr, 5.0, 2),
            (2.0 - 0.2j, xarr, 3),
            (marr, xarr, 3),
        ]

        for m, x, n_pole in test_cases:
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)

            expected = 2

            self.assertEqual(len(a), expected, "Length wrong for a with n_pole=%d" % n_pole)
            self.assertEqual(len(b), expected, "Length wrong for b with n_pole=%d" % n_pole)
            self.assertEqual(len(c), expected, "Length wrong for c with n_pole=%d" % n_pole)
            self.assertEqual(len(d), expected, "Length wrong for d with n_pole=%d" % n_pole)


class TestBoundaryConditions(unittest.TestCase):
    def test_boundary_conditions(self):
        """
        Test the calculated c and d coefficients using boundary conditions at the sphere surface.
        """
        test_cases = [
            (1.5 - 0.1j, 1, 0),
            (1.5, 1, 0),
            (1.5, 10, 0),
            (1.5, 2.5, 0),
            (1.33 - 0.05j, 1.0, 0),
        ]

        # Define tolerance for boundary condition errors
        tolerance = 1e-5

        for m, x, n_pole in test_cases:
            # Compute Mie coefficients for scattered and internal fields
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)

            print("--------------------------------------")
            print("a")
            print("miepython", cs(a))
            print("c")
            print("miepython", cs(c))

            n = np.arange(1, len(a) + 1)
            m = np.conjugate(m)
            mx = m * x

            psi_n_x = riccati_bessel_jn(n, x)
            psi_n_prime_x = d_riccati_bessel_jn(n, x)
            psi_n_mx = riccati_bessel_jn(n, mx)
            psi_n_prime_mx = d_riccati_bessel_jn(n, mx)
            xi_n_x = riccati_bessel_h1(n, x)
            xi_n_prime_x = d_riccati_bessel_h1(n, x)

            error1 = np.abs(psi_n_mx * c + m * xi_n_x * b - m * psi_n_x)
            error2 = np.abs(psi_n_prime_mx * c + xi_n_prime_x * b - psi_n_prime_x)
            error3 = np.abs(psi_n_mx * d + xi_n_x * a - psi_n_x)
            error4 = np.abs(psi_n_prime_mx * d + m * xi_n_prime_x * a - m * psi_n_prime_x)
            print("**************************************************************")
            print("m=%s x=%.5f n=%d" % (cs(m), x, n_pole))
            print("bohren 4.51a LHS:\n", cs(psi_n_mx * c), "=", cs(psi_n_mx), "*", cs(c))
            print(
                "bohren 4.51a LHS:\n",
                cs(m * xi_n_x * b),
                "=",
                cs(m),
                "*",
                cs(xi_n_x),
                "*",
                cs(b),
            )
            print("bohren 4.51a RHS:\n", cs(-m * psi_n_x), "=", cs(-m), "*", cs(psi_n_x))
            print("bohren 4.51a BC1:\n", cs(psi_n_mx * c + m * xi_n_x * b - m * psi_n_x))
            print()
            print("bohren 4.51b LHS:\n", cs(psi_n_prime_mx * c + xi_n_prime_x * b))
            print("bohren 4.51b RHS:\n", cs(psi_n_prime_x))
            print("bohren 4.51b BC1:\n", cs(psi_n_prime_mx * c + xi_n_prime_x * b - psi_n_prime_x))
            print()
            print("bohren 4.51c LHS:\n", cs(psi_n_mx * d + xi_n_x * a))
            print("bohren 4.51c RHS:\n", cs(psi_n_x))
            print("bohren 4.51c BC1:\n", cs(psi_n_mx * d + xi_n_x * a - psi_n_x))
            print()
            print("bohren 4.51d LHS:\n", cs(psi_n_prime_mx * d + m * xi_n_prime_x * a))
            print("bohren 4.51d RHS:\n", cs(m * psi_n_prime_x))
            print(
                "bohren 4.51d BC1:\n",
                cs(psi_n_prime_mx * d + m * xi_n_prime_x * a - m * psi_n_prime_x),
            )
            print()

            # Compute maximum error
            max_error = max(np.max(error1), np.max(error2), np.max(error3), np.max(error4))

            # Assert that maximum error is below the tolerance
            self.assertLess(
                max_error,
                tolerance,
                f"Boundary condition test failed with max error {max_error:.2e}",
            )


if __name__ == "__main__":
    unittest.main()
