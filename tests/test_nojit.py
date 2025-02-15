#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=consider-using-f-string

import os
import unittest
import numpy as np

os.environ["MIEPYTHON_USE_JIT"] = "0"  # Set to "0" to disable JIT
import miepython as mie


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
    for zz in z:
        if s != "":
            s += ", "
        s += cs_scalar(zz, N)

    return s


class NonAbsorbing(unittest.TestCase):

    def test_03_bh_dielectric(self):
        m = 1.55
        lambda0 = 0.6328
        radius = 0.525
        x = 2 * np.pi * radius / lambda0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        self.assertAlmostEqual(qext, 3.10543, delta=0.00001)
        self.assertAlmostEqual(qsca, 3.10543, delta=0.00001)
        self.assertAlmostEqual(qback, 2.92534, delta=0.00001)
        self.assertAlmostEqual(g, 0.63314, delta=0.00001)

    def test_05_wiscombe_non_absorbing(self):

        # MIEV0 Test Case 5
        m = complex(0.75, 0.0)
        x = 0.099
        s1 = 1.81756e-8 - 1.64810e-4 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000007, delta=1e-6)
        self.assertAlmostEqual(g, 0.001448, delta=1e-6)
        self.assertAlmostEqual(qback, G, delta=1e-6)

        # MIEV0 Test Case 6
        m = complex(0.75, 0.0)
        x = 0.101
        s1 = 2.04875e-08 - 1.74965e-04 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000008, delta=1e-6)
        self.assertAlmostEqual(g, 0.001507, delta=1e-6)
        self.assertAlmostEqual(qback, G, delta=1e-6)

        # MIEV0 Test Case 7
        m = complex(0.75, 0.0)
        x = 10.0
        s1 = -1.07857e00 - 3.60881e-02 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.232265, delta=1e-6)
        self.assertAlmostEqual(g, 0.896473, delta=1e-6)
        self.assertAlmostEqual(qback, G, delta=1e-6)

        # MIEV0 Test Case 8
        m = complex(0.75, 0.0)
        x = 1000.0
        s1 = 1.70578e01 + 4.84251e02 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 1.997908, delta=1e-6)
        self.assertAlmostEqual(g, 0.844944, delta=1e-6)
        self.assertAlmostEqual(qback, G, delta=1e-6)

    def test_05_old_wiscombe_non_absorbing(self):

        # OLD MIEV0 Test Case 1
        m = complex(1.5, 0.0)
        x = 10
        s1 = 4.322e00 + 4.868e00 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.8820, delta=1e-4)
        self.assertAlmostEqual(qback, G, delta=1e-4)

        # OLD MIEV0 Test Case 2
        m = complex(1.5, 0.0)
        x = 100
        s1 = 4.077e01 + 5.175e01 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.0944, delta=1e-4)
        self.assertAlmostEqual(qback, G, delta=1e-4)

        # OLD MIEV0 Test Case 3
        m = complex(1.5, 0.0)
        x = 1000
        G = 4 * 2.576e06 / x**2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.0139, delta=1e-4)
        self.assertAlmostEqual(qback, G, delta=1e-3)

        # OLD MIEV0 Test Case 4
        m = complex(1.5, 0.0)
        x = 5000.0
        G = 4 * 2.378e08 / x**2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.0086, delta=1e-4)
        self.assertAlmostEqual(qback, G, delta=3e-3)

    def test_04_non_dielectric(self):
        m = 1.55 - 0.1j
        lambda0 = 0.6328
        radius = 0.525
        x = 2 * np.pi * radius / lambda0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        self.assertAlmostEqual(qext, 2.86165188243, delta=1e-7)
        self.assertAlmostEqual(qsca, 1.66424911991, delta=1e-7)
        self.assertAlmostEqual(qback, 0.20599534080, delta=1e-7)
        self.assertAlmostEqual(g, 0.80128972639, delta=1e-7)


class Absorbing(unittest.TestCase):
    def test_06_wiscombe_water_absorbing(self):

        # MIEV0 Test Case 9
        m = complex(1.33, -0.00001)
        x = 1.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.093923, delta=1e-6)
        self.assertAlmostEqual(g, 0.184517, delta=1e-6)

        # MIEV0 Test Case 10
        m = complex(1.33, -0.00001)
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.096594, delta=1e-6)
        self.assertAlmostEqual(g, 0.868959, delta=1e-6)

        # MIEV0 Test Case 11
        m = complex(1.33, -0.00001)
        x = 10000.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(g, 0.907840, delta=1e-6)
        self.assertAlmostEqual(qsca, 1.723857, delta=1e-6)

    def test_07_wiscombe_absorbing(self):

        # MIEV0 Test Case 12
        m = 1.5 - 1j
        x = 0.055
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000011, delta=1e-6)
        self.assertAlmostEqual(g, 0.000491, delta=1e-6)

        # MIEV0 Test Case 13
        m = 1.5 - 1j
        x = 0.056
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000012, delta=1e-6)
        self.assertAlmostEqual(g, 0.000509, delta=1e-6)

        # MIEV0 Test Case 14
        m = 1.5 - 1j
        x = 1
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.6634538, delta=1e-6)
        self.assertAlmostEqual(g, 0.192136, delta=1e-6)

        # MIEV0 Test Case 15
        m = 1.5 - 1j
        x = 100
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 1.283697, delta=1e-3)
        self.assertAlmostEqual(qext, 2.097502, delta=1e-2)
        self.assertAlmostEqual(g, 0.850252, delta=1e-3)

        # MIEV0 Test Case 16
        m = 1.5 - 1j
        x = 10000
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 1.236575, delta=1e-6)
        self.assertAlmostEqual(qext, 2.004368, delta=1e-6)
        self.assertAlmostEqual(g, 0.846309, delta=1e-6)

    def test_08_wiscombe_more_absorbing(self):

        # MIEV0 Test Case 17
        m = 10.0 - 10.0j
        x = 1.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.049405, delta=1e-6)
        self.assertAlmostEqual(g, -0.110664, delta=1e-6)

        # MIEV0 Test Case 18
        m = 10.0 - 10.0j
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 1.836785, delta=1e-6)
        self.assertAlmostEqual(g, 0.556215, delta=1e-6)

        # MIEV0 Test Case 19
        m = 10.0 - 10.0j
        x = 10000.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 1.795393, delta=1e-6)
        self.assertAlmostEqual(g, 0.548194, delta=1e-6)

    def test_09_single_nonmagnetic(self):
        m = 1.5 - 0.5j
        x = 2.5
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        self.assertAlmostEqual(qext, 2.562873497454734, delta=1e-7)
        self.assertAlmostEqual(qsca, 1.097071819088392, delta=1e-7)
        self.assertAlmostEqual(qback, 0.123586468179818, delta=1e-7)
        self.assertAlmostEqual(g, 0.748905978948507, delta=1e-7)


class PerfectlyReflecting(unittest.TestCase):

    def test_11_wiscombe_perfectly_conducting(self):

        m = -10000j
        # MIEV0 Test Case 0
        x = 0.001
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 3.3333e-12, delta=1e-13)

        # MIEV0 Test Case 1
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000321, delta=1e-4)
        self.assertAlmostEqual(g, -0.397357, delta=1e-3)

        # MIEV0 Test Case 2
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 0.000348, delta=1e-6)
        self.assertAlmostEqual(g, -0.397262, delta=1e-6)

        # MIEV0 Test Case 3
        x = 100
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.008102, delta=1e-6)
        self.assertAlmostEqual(g, 0.500926, delta=1e-6)

        # MIEV0 Test Case 4
        x = 10000
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qsca, 2.000289, delta=1e-6)
        self.assertAlmostEqual(g, 0.500070, delta=1e-6)


class Small(unittest.TestCase):

    def test_10_small_spheres(self):
        # MIEV0 Test Case 5
        m = 0.75
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000007, delta=1e-6)
        self.assertAlmostEqual(g, 0.001448, delta=1e-6)

        # MIEV0 Test Case 6
        m = 0.75
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000008, delta=1e-6)
        self.assertAlmostEqual(g, 0.001507, delta=1e-6)

        m = 1.5 - 1j
        x = 0.055
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.101491, delta=1e-6)
        self.assertAlmostEqual(g, 0.000491, delta=1e-6)
        x = 0.056
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.103347, delta=1e-6)
        self.assertAlmostEqual(g, 0.000509, delta=1e-6)

        m = 1e-10 - 1e10j
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000321, delta=1e-6)
        self.assertAlmostEqual(g, -0.397357, delta=1e-4)
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000348, delta=1e-6)
        self.assertAlmostEqual(g, -0.397262, delta=1e-6)

        m = 0 - 1e10j
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000321, delta=1e-6)
        self.assertAlmostEqual(g, -0.397357, delta=1e-4)
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        self.assertAlmostEqual(qext, 0.000348, delta=1e-6)
        self.assertAlmostEqual(g, -0.397262, delta=1e-4)


class AngleScattering(unittest.TestCase):

    def test_12_scatter_function(self):
        """Wiscombe MIEV0 Test Case 14."""
        x = 1.0
        m = 1.5 - 1.0j
        theta = np.arange(0, 181, 30)
        mu = np.cos(theta * np.pi / 180)

        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        S1, S2 = mie.S1_S2(m, x, mu, norm="wiscombe")
        #        S1 *= np.sqrt(np.pi * x**2 * qext)
        #        S2 *= np.sqrt(np.pi * x**2 * qext)

        self.assertAlmostEqual(S1[0].real, 0.584080, delta=1e-6)
        self.assertAlmostEqual(S1[0].imag, 0.190515, delta=1e-6)
        self.assertAlmostEqual(S2[0].real, 0.584080, delta=1e-6)
        self.assertAlmostEqual(S2[0].imag, 0.190515, delta=1e-6)

        self.assertAlmostEqual(S1[1].real, 0.565702, delta=1e-6)
        self.assertAlmostEqual(S1[1].imag, 0.187200, delta=1e-6)
        self.assertAlmostEqual(S2[1].real, 0.500161, delta=1e-6)
        self.assertAlmostEqual(S2[1].imag, 0.145611, delta=1e-6)

        self.assertAlmostEqual(S1[2].real, 0.517525, delta=1e-6)
        self.assertAlmostEqual(S1[2].imag, 0.178443, delta=1e-6)
        self.assertAlmostEqual(S2[2].real, 0.287964, delta=1e-6)
        self.assertAlmostEqual(S2[2].imag, 0.041054, delta=1e-6)

        self.assertAlmostEqual(S1[3].real, 0.456340, delta=1e-6)
        self.assertAlmostEqual(S1[3].imag, 0.167167, delta=1e-6)
        self.assertAlmostEqual(S2[3].real, 0.0362285, delta=1e-6)
        self.assertAlmostEqual(S2[3].imag, -0.0618265, delta=1e-6)

        self.assertAlmostEqual(S1[4].real, 0.400212, delta=1e-6)
        self.assertAlmostEqual(S1[4].imag, 0.156643, delta=1e-6)
        self.assertAlmostEqual(S2[4].real, -0.174875, delta=1e-6)
        self.assertAlmostEqual(S2[4].imag, -0.122959, delta=1e-6)

        self.assertAlmostEqual(S1[5].real, 0.362157, delta=1e-6)
        self.assertAlmostEqual(S1[5].imag, 0.149391, delta=1e-6)
        self.assertAlmostEqual(S2[5].real, -0.305682, delta=1e-6)
        self.assertAlmostEqual(S2[5].imag, -0.143846, delta=1e-6)

        self.assertAlmostEqual(S1[6].real, 0.348844, delta=1e-6)
        self.assertAlmostEqual(S1[6].imag, 0.146829, delta=1e-6)
        self.assertAlmostEqual(S2[6].real, -0.348844, delta=1e-6)
        self.assertAlmostEqual(S2[6].imag, -0.146829, delta=1e-6)

    def test_13_unity_normalization(self):
        """Wiscombe MIEV0 Test Case 14."""
        x = 1.0
        m = 1.5 - 1.0j
        theta = np.arange(0, 181, 30)
        mu = np.cos(theta * np.pi / 180)

        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        S1, S2 = mie.S1_S2(m, x, mu, norm="wiscombe")

        self.assertAlmostEqual(S1[0].real, 0.584080, delta=1e-6)
        self.assertAlmostEqual(S1[0].imag, 0.190515, delta=1e-6)
        self.assertAlmostEqual(S2[0].real, 0.584080, delta=1e-6)
        self.assertAlmostEqual(S2[0].imag, 0.190515, delta=1e-6)

        self.assertAlmostEqual(S1[1].real, 0.565702, delta=1e-6)
        self.assertAlmostEqual(S1[1].imag, 0.187200, delta=1e-6)
        self.assertAlmostEqual(S2[1].real, 0.500161, delta=1e-6)
        self.assertAlmostEqual(S2[1].imag, 0.145611, delta=1e-6)

        self.assertAlmostEqual(S1[2].real, 0.517525, delta=1e-6)
        self.assertAlmostEqual(S1[2].imag, 0.178443, delta=1e-6)
        self.assertAlmostEqual(S2[2].real, 0.287964, delta=1e-6)
        self.assertAlmostEqual(S2[2].imag, 0.041054, delta=1e-6)

        self.assertAlmostEqual(S1[3].real, 0.456340, delta=1e-6)
        self.assertAlmostEqual(S1[3].imag, 0.167167, delta=1e-6)
        self.assertAlmostEqual(S2[3].real, 0.0362285, delta=1e-6)
        self.assertAlmostEqual(S2[3].imag, -0.0618265, delta=1e-6)

        self.assertAlmostEqual(S1[4].real, 0.400212, delta=1e-6)
        self.assertAlmostEqual(S1[4].imag, 0.156643, delta=1e-6)
        self.assertAlmostEqual(S2[4].real, -0.174875, delta=1e-6)
        self.assertAlmostEqual(S2[4].imag, -0.122959, delta=1e-6)

        self.assertAlmostEqual(S1[5].real, 0.362157, delta=1e-6)
        self.assertAlmostEqual(S1[5].imag, 0.149391, delta=1e-6)
        self.assertAlmostEqual(S2[5].real, -0.305682, delta=1e-6)
        self.assertAlmostEqual(S2[5].imag, -0.143846, delta=1e-6)

        self.assertAlmostEqual(S1[6].real, 0.348844, delta=1e-6)
        self.assertAlmostEqual(S1[6].imag, 0.146829, delta=1e-6)
        self.assertAlmostEqual(S2[6].real, -0.348844, delta=1e-6)
        self.assertAlmostEqual(S2[6].imag, -0.146829, delta=1e-6)

    def test_i_unpolarized_01(self):
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 1000)
        qext, qsca, _, _ = mie.efficiencies_mx(m, x)
        expected = [
            qsca / qext,
            1.0,
            4 * np.pi,
            qsca,
            qext,
            qsca * 4 * np.pi * x**2,
            qsca * np.pi * x**2,
        ]

        for i, norm in enumerate(["albedo", "one", "4pi", "qsca", "qext", "bohren", "wiscombe"]):
            intensity = mie.i_unpolarized(m, x, mu, norm)
            total = 2 * np.pi * (mu[1] - mu[0]) * np.sum(intensity)
            self.assertAlmostEqual(total / expected[i], 1.0, delta=4e-3)

    def test_i_par_i_per_01(self):
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 10000)
        qext, qsca, _, _ = mie.efficiencies_mx(m, x)
        expected = [
            qsca / qext,
            1.0,
            4 * np.pi,
            qsca,
            qext,
            qsca * 4 * np.pi * x**2,
            qsca * np.pi * x**2,
        ]

        for i, norm in enumerate(["albedo", "one", "4pi", "qsca", "qext", "bohren", "wiscombe"]):
            iper = mie.i_per(m, x, mu, norm)
            total1 = 2 * np.pi * (mu[1] - mu[0]) * np.sum(iper)
            ipar = mie.i_par(m, x, mu, norm)
            total2 = 2 * np.pi * (mu[1] - mu[0]) * np.sum(ipar)
            total = (total1 + total2) / 2
            self.assertAlmostEqual(total / expected[i], 1, delta=1e-3)

    def test_molecular_hydrogen(self):
        m = 1.00013626
        x = 0.0006403246172921872
        mu = np.linspace(-1, 1, 100)
        ph = mie.i_unpolarized(m, x, mu)
        self.assertAlmostEqual(ph[1], 0.1169791, delta=1e-5)


class MiePhaseMatrix(unittest.TestCase):
    def test_phase_matrix_basic(self):
        """
        Test element (0,0).

        Element (0, 0) of array returned by phase_matrix should match output
        of i_unpolarized.
        """
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 1000)

        p = mie.phase_matrix(m, x, mu)  # result to be validated
        p00 = mie.i_unpolarized(m, x, mu)  # reference result

        assert np.allclose(p[0, 0], p00, rtol=1e-9)

    def test_phase_matrix_mu_scalar(self):
        """
        Test element (0,0).

        phase_matrix should return (4, 4) array when mu is scalar.
        """
        assert mie.phase_matrix(m=1.5, x=2.0, mu=0.0).shape == (4, 4)

    def test_phase_matrix_symmetry(self):
        """
        Check symmetry.

        Upper left 2X2 block is symmetric and lower right 2X2 block is
        antisymmetric.
        """
        p = mie.phase_matrix(m=1.5, x=2.0, mu=np.linspace(-1, 1, 10))
        assert np.allclose(p[0, 1], p[1, 0])
        assert np.allclose(p[2, 3], -p[3, 2])

    def test_phase_matrix_unity(self):
        """
        Ensure elements add up properly.

        Element p[0, 0, :]**2 = should equal the sum of squares of other elements.
        """
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 1000)

        p = mie.phase_matrix(m, x, mu)  # result to be validated

        assert np.allclose(p[0, 0] ** 2, p[0, 1] ** 2 + p[2, 2] ** 2 + p[2, 3] ** 2, rtol=1e-9)

    def test_phase_matrix_bohren(self):
        """
        Compare with output from Bohren's program.

        s33 and s34 elements are normalized by s11
        s11 is normalized to 1 in the forward direction
        pol is -s12 normalized by s11
        """
        mm = np.array(
            [
                [000.00, 0.100000e01, 0.000000e00, 0.100000e01, 0.000000e00],
                [009.00, 0.785390e00, -0.459811e-02, 0.999400e00, 0.343261e-01],
                [018.00, 0.356897e00, -0.458541e-01, 0.986022e00, 0.160184e00],
                [027.00, 0.766119e-01, -0.364744e00, 0.843603e00, 0.394076e00],
                [036.00, 0.355355e-01, -0.534997e00, 0.686967e00, -0.491787e00],
                [045.00, 0.701845e-01, 0.959953e-02, 0.959825e00, -0.280434e00],
                [054.00, 0.574313e-01, 0.477927e-01, 0.985371e00, 0.163584e00],
                [063.00, 0.219660e-01, -0.440604e00, 0.648043e00, 0.621216e00],
                [072.00, 0.125959e-01, -0.831996e00, 0.203255e00, -0.516208e00],
                [081.00, 0.173750e-01, 0.341670e-01, 0.795354e00, -0.605182e00],
                [090.00, 0.124601e-01, 0.230462e00, 0.937497e00, 0.260742e00],
                [099.00, 0.679093e-02, -0.713472e00, -0.717397e-02, 0.700647e00],
                [108.00, 0.954239e-02, -0.756255e00, -0.394748e-01, -0.653085e00],
                [117.00, 0.863419e-02, -0.281215e00, 0.536251e00, -0.795835e00],
                [126.00, 0.227421e-02, -0.239612e00, 0.967602e00, 0.795798e-01],
                [135.00, 0.543998e-02, -0.850804e00, 0.187531e00, -0.490882e00],
                [144.00, 0.160243e-01, -0.706334e00, 0.495254e00, -0.505781e00],
                [153.00, 0.188852e-01, -0.891081e00, 0.453277e00, -0.226817e-01],
                [162.00, 0.195254e-01, -0.783319e00, -0.391613e00, 0.482752e00],
                [171.00, 0.301676e-01, -0.196194e00, -0.962069e00, 0.189556e00],
                [180.00, 0.383189e-01, 0.000000e00, -0.100000e01, 0.000000e00],
            ]
        )

        m = 1.55
        x = 5.213
        theta = np.linspace(0, 180, 21)

        mu = np.cos(np.radians(theta))
        p = mie.phase_matrix(m, x, mu, norm="bohren")  # result to be validated

        assert np.allclose(theta, mm[:, 0], rtol=1e-9)
        assert np.allclose(p[0, 0, :] / p[0, 0, 0], mm[:, 1], atol=1e-3)
        assert np.allclose(-p[0, 1, :] / p[0, 0, :], mm[:, 2], atol=1e-3)
        assert np.allclose(p[2, 2, :] / p[0, 0, :], mm[:, 3], atol=1e-3)
        assert np.allclose(p[3, 2, :] / p[0, 0, :], mm[:, 4], atol=1e-3)


class NotebookTests(unittest.TestCase):
    def test_nb1_x(self):
        N = 500
        m = 1.5
        x = np.linspace(0.1, 20, N)  # also in microns
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        self.assertAlmostEqual(qsca[0], 2.3084093592198083e-05, delta=1e-6)
        self.assertAlmostEqual(qsca[100], 4.105960809066763, delta=1e-6)
        self.assertAlmostEqual(qsca[200], 1.9947867190110644, delta=1e-6)
        self.assertAlmostEqual(qsca[300], 2.4652591512196405, delta=1e-6)
        self.assertAlmostEqual(qsca[400], 2.472171798724846, delta=1e-6)
        self.assertAlmostEqual(qsca[499], 2.03583698038088, delta=1e-6)

    def test_nb1_rho(self):
        N = 500
        rho = np.linspace(0.1, 20, N)

        m = 1.5
        x15 = rho / 2 / (m - 1)
        qext, scal5, qback, g = mie.efficiencies_mx(m, x15)

        m = 1.1
        x11 = rho / 2 / (m - 1)
        qext, scal1, qback, g = mie.efficiencies_mx(m, x11)

        self.assertAlmostEqual(scal1[0], 0.0006616369953521216, delta=1e-6)
        self.assertAlmostEqual(scal1[99], 3.449616595439377, delta=1e-6)
        self.assertAlmostEqual(scal1[199], 1.6837703285684387, delta=1e-6)
        self.assertAlmostEqual(scal1[299], 2.3167184401740495, delta=1e-6)
        self.assertAlmostEqual(scal1[399], 2.218210809017406, delta=1e-6)
        self.assertAlmostEqual(scal1[499], 1.876467571615533, delta=1e-6)

        self.assertAlmostEqual(scal5[0], 2.3084093592198083e-05, delta=1e-6)
        self.assertAlmostEqual(scal5[99], 4.07295075914037, delta=1e-6)
        self.assertAlmostEqual(scal5[199], 1.8857586341949146, delta=1e-6)
        self.assertAlmostEqual(scal5[299], 2.464763930426085, delta=1e-6)
        self.assertAlmostEqual(scal5[399], 2.430569030744473, delta=1e-6)
        self.assertAlmostEqual(scal5[499], 2.03583698038088, delta=1e-6)

    def test_nb1_spheres(self):
        N = 500
        m = 1.0
        r = 500  # nm
        lambda0 = np.linspace(300, 800, N)  # also in nm

        mwater = 4 / 3  # rough approximation
        mm = m / mwater
        xx = 2 * np.pi * r * mwater / lambda0

        qext, qsca, qback, g = mie.efficiencies_mx(mm, xx)

        self.assertAlmostEqual(qsca[0], 1.5525047718022498, delta=1e-6)
        self.assertAlmostEqual(qsca[99], 2.1459528526672678, delta=1e-6)
        self.assertAlmostEqual(qsca[199], 2.365171370327149, delta=1e-6)
        self.assertAlmostEqual(qsca[299], 2.2039860928542128, delta=1e-6)
        self.assertAlmostEqual(qsca[399], 1.9261758931397088, delta=1e-6)
        self.assertAlmostEqual(qsca[499], 1.640006561518987, delta=1e-6)

    def test_nb1_mie(self):
        m_sphere = 1.0
        n_water = 4 / 3
        d = 1000
        lambda0 = np.linspace(300, 800)
        qext, qsca, qback, g = mie.efficiencies(m_sphere, d, lambda0, n_water)

        self.assertAlmostEqual(qsca[0], 1.5525047718022498, delta=1e-6)
        self.assertAlmostEqual(qsca[9], 2.107970892634116, delta=1e-6)
        self.assertAlmostEqual(qsca[19], 2.3654333205160074, delta=1e-6)
        self.assertAlmostEqual(qsca[29], 2.213262310704816, delta=1e-6)
        self.assertAlmostEqual(qsca[39], 1.9314911518355427, delta=1e-6)
        self.assertAlmostEqual(qsca[49], 1.640006561518987, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
