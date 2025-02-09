#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import unittest
import numpy as np

from miepython.miepython_nojit import _D_calc
from miepython.bessel import *


def cs_scalar(z, N=5):
    """Convert complex number to string for printing."""
    if z.imag < 0:
        form = "(%% .%df - %%.%dfj)" % (N, N)
    else:
        form = "(%% .%df + %%.%dfj)" % (N, N)
    return form % (z.real, abs(z.imag))


def cs(z, N=5):
    """Convert complex number to string for printing."""
    if np.isscalar(z):
        return cs_scalar(z, N)

    s = ""
    terms = 0
    for zz in z:
        if s != "":
            s += ", "
        s += cs_scalar(zz, N)
        terms += 1
        if terms >= 4:
            break

    return s


def pymiescatt_D(m, x):
    #  http://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_ab
    mx = m * x
    nmax = np.round(2 + x + 4 * (x ** (1 / 3)))
    nmx = np.round(max(nmax, np.abs(mx)) + 16)

    # B&H Equation 4.89
    Dn = np.zeros(int(nmx), dtype=complex)
    for i in range(int(nmx) - 1, 1, -1):
        Dn[i - 1] = (i / mx) - (1 / (Dn[i] + i / mx))

    D = Dn[1 : int(nmax) + 1]  # Dn(mx), drop terms beyond nMax

    return D


def bohren_D(m, x):
    """
    Bohren-Huffman calculation of Logarithmic derivatives D_n.

    Parameters:
        x (float): Size parameter (2 * pi * a / lambda).
        m (complex): Complex refractive index ratio of the sphere and medium.

    Returns:
        D (np.ndarray): Logarithmic derivatives D_n
    """
    mx = x * m
    nstop = int(x + 4 * (x ** (1 / 3)) + 2)
    nmx = int(max(nstop, np.abs(mx)) + 15)
    D = np.zeros(nmx, dtype=complex)

    for n in range(nmx - 1, 0, -1):
        D[n - 1] = ((n + 1) / mx) - (1.0 / (D[n] + (n + 1) / mx))
    return D


def basic_D(m, x, terms=10):
    # Riccati-Bessel functions at x and mx
    mx = x * m
    nstop = int(x + 4 * (x ** (1 / 3)) + 2)
    nmx = int(max(nstop, np.abs(mx)) + 15)
    n = np.arange(1, nmx)

    psi_n_mx = riccati_bessel_jn(n, mx)
    psi_n_prime_mx = d_riccati_bessel_jn(n, mx)

    D = psi_n_prime_mx / psi_n_mx
    return D


class TestD(unittest.TestCase):
    def test_01_Dn(self):
        m = 1.5
        x = 1
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_02_Dn(self):
        m = 1.5
        x = 0.01
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_03_Dn(self):
        m = 1.5
        x = 100
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Dt[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Dt[0].imag, delta=0.00000001)

    def test_04_Dn(self):
        m = 1.5 - 0.5j
        x = 1
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_05_Dn(self):
        m = 1.5 - 0.5j
        x = 0.01
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_06_Dn(self):
        m = 1.5 - 0.5j
        x = 100
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_07_Dn(self):
        m = 1.5 - 15j
        x = 1
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_08_Dn(self):
        m = 1.5 - 15j
        x = 0.01
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_09_Dn(self):
        m = 1.5 - 15j
        x = 10
        nstop = int(x + 4 * (x ** (1 / 3)) + 2)
        D = _D_calc(m, x, nstop)
        Dt = basic_D(m, x)
        Db = bohren_D(m, x)
        Dp = pymiescatt_D(m, x)

        print("m=%s, x=%.4f" % (cs(m), x))
        print(cs(D))
        print(cs(Db))
        print(cs(Dt))
        print(cs(Dp))
        for n in range(3):
            self.assertAlmostEqual(D[0].real, Db[0].real, delta=0.00000001)
            self.assertAlmostEqual(D[0].imag, Db[0].imag, delta=0.00000001)

    def test_10_Dn(self):
        x = 62
        m = 1.28 - 1.37j
        nstop = 50
        dn = _D_calc(m, x, nstop)
        self.assertAlmostEqual(dn[9].real, 0.004087, delta=0.00001)
        self.assertAlmostEqual(dn[9].imag, 1.0002620, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
