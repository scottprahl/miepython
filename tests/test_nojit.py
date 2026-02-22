#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=consider-using-f-string

import os
import pytest
import numpy as np

os.environ["MIEPYTHON_USE_JIT"] = "0"  # Set to "0" to disable JIT
import miepython as mie


class TestNonAbsorbing:

    def test_03_bh_dielectric(self):
        m = 1.55
        lambda0 = 0.6328
        radius = 0.525
        x = 2 * np.pi * radius / lambda0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        assert qext == pytest.approx(3.10543, abs=0.00001)
        assert qsca == pytest.approx(3.10543, abs=0.00001)
        assert qback == pytest.approx(2.92534, abs=0.00001)
        assert g == pytest.approx(0.63314, abs=0.00001)

    def test_05_wiscombe_non_absorbing(self):

        # MIEV0 Test Case 5
        m = complex(0.75, 0.0)
        x = 0.099
        s1 = 1.81756e-8 - 1.64810e-4 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000007, abs=1e-6)
        assert g == pytest.approx(0.001448, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 6
        m = complex(0.75, 0.0)
        x = 0.101
        s1 = 2.04875e-08 - 1.74965e-04 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000008, abs=1e-6)
        assert g == pytest.approx(0.001507, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 7
        m = complex(0.75, 0.0)
        x = 10.0
        s1 = -1.07857e00 - 3.60881e-02 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.232265, abs=1e-6)
        assert g == pytest.approx(0.896473, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 8
        m = complex(0.75, 0.0)
        x = 1000.0
        s1 = 1.70578e01 + 4.84251e02 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.997908, abs=1e-6)
        assert g == pytest.approx(0.844944, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)

    def test_05_old_wiscombe_non_absorbing(self):

        # OLD MIEV0 Test Case 1
        m = complex(1.5, 0.0)
        x = 10
        s1 = 4.322e00 + 4.868e00 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.8820, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-4)
        # OLD MIEV0 Test Case 2
        m = complex(1.5, 0.0)
        x = 100
        s1 = 4.077e01 + 5.175e01 * 1j
        G = abs(2 * s1 / x) ** 2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0944, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-4)
        # OLD MIEV0 Test Case 3
        m = complex(1.5, 0.0)
        x = 1000
        G = 4 * 2.576e06 / x**2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0139, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-3)
        # OLD MIEV0 Test Case 4
        m = complex(1.5, 0.0)
        x = 5000.0
        G = 4 * 2.378e08 / x**2
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0086, abs=1e-4)
        assert qback == pytest.approx(G, abs=3e-3)

    def test_04_non_dielectric(self):
        m = 1.55 - 0.1j
        lambda0 = 0.6328
        radius = 0.525
        x = 2 * np.pi * radius / lambda0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        assert qext == pytest.approx(2.86165188243, abs=1e-7)
        assert qsca == pytest.approx(1.66424911991, abs=1e-7)
        assert qback == pytest.approx(0.20599534080, abs=1e-7)
        assert g == pytest.approx(0.80128972639, abs=1e-7)


class TestAbsorbing:
    def test_06_wiscombe_water_absorbing(self):

        # MIEV0 Test Case 9
        m = complex(1.33, -0.00001)
        x = 1.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.093923, abs=1e-6)
        assert g == pytest.approx(0.184517, abs=1e-6)
        # MIEV0 Test Case 10
        m = complex(1.33, -0.00001)
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.096594, abs=1e-6)
        assert g == pytest.approx(0.868959, abs=1e-6)
        # MIEV0 Test Case 11
        m = complex(1.33, -0.00001)
        x = 10000.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert g == pytest.approx(0.907840, abs=1e-6)
        assert qsca == pytest.approx(1.723857, abs=1e-6)

    def test_07_wiscombe_absorbing(self):

        # MIEV0 Test Case 12
        m = 1.5 - 1j
        x = 0.055
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000011, abs=1e-6)
        assert g == pytest.approx(0.000491, abs=1e-6)
        # MIEV0 Test Case 13
        m = 1.5 - 1j
        x = 0.056
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000012, abs=1e-6)
        assert g == pytest.approx(0.000509, abs=1e-6)
        # MIEV0 Test Case 14
        m = 1.5 - 1j
        x = 1
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.6634538, abs=1e-6)
        assert g == pytest.approx(0.192136, abs=1e-6)
        # MIEV0 Test Case 15
        m = 1.5 - 1j
        x = 100
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.283697, abs=1e-3)
        assert qext == pytest.approx(2.097502, abs=1e-2)
        assert g == pytest.approx(0.850252, abs=1e-3)
        # MIEV0 Test Case 16
        m = 1.5 - 1j
        x = 10000
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.236575, abs=1e-6)
        assert qext == pytest.approx(2.004368, abs=1e-6)
        assert g == pytest.approx(0.846309, abs=1e-6)

    def test_08_wiscombe_more_absorbing(self):

        # MIEV0 Test Case 17
        m = 10.0 - 10.0j
        x = 1.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.049405, abs=1e-6)
        assert g == pytest.approx(-0.110664, abs=1e-6)
        # MIEV0 Test Case 18
        m = 10.0 - 10.0j
        x = 100.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.836785, abs=1e-6)
        assert g == pytest.approx(0.556215, abs=1e-6)
        # MIEV0 Test Case 19
        m = 10.0 - 10.0j
        x = 10000.0
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.795393, abs=1e-6)
        assert g == pytest.approx(0.548194, abs=1e-6)

    def test_09_single_nonmagnetic(self):
        m = 1.5 - 0.5j
        x = 2.5
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        assert qext == pytest.approx(2.562873497454734, abs=1e-7)
        assert qsca == pytest.approx(1.097071819088392, abs=1e-7)
        assert qback == pytest.approx(0.123586468179818, abs=1e-7)
        assert g == pytest.approx(0.748905978948507, abs=1e-7)


class TestPerfectlyReflecting:

    def test_11_wiscombe_perfectly_conducting(self):

        m = -10000j
        # MIEV0 Test Case 0
        x = 0.001
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(3.3333e-12, abs=1e-13)
        # MIEV0 Test Case 1
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000321, abs=1e-4)
        assert g == pytest.approx(-0.397357, abs=1e-3)
        # MIEV0 Test Case 2
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-6)
        # MIEV0 Test Case 3
        x = 100
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.008102, abs=1e-6)
        assert g == pytest.approx(0.500926, abs=1e-6)
        # MIEV0 Test Case 4
        x = 10000
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.000289, abs=1e-6)
        assert g == pytest.approx(0.500070, abs=1e-6)


class TestSmall:

    def test_10_small_spheres(self):
        # MIEV0 Test Case 5
        m = 0.75
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000007, abs=1e-6)
        assert g == pytest.approx(0.001448, abs=1e-6)
        # MIEV0 Test Case 6
        m = 0.75
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000008, abs=1e-6)
        assert g == pytest.approx(0.001507, abs=1e-6)
        m = 1.5 - 1j
        x = 0.055
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.101491, abs=1e-6)
        assert g == pytest.approx(0.000491, abs=1e-6)
        x = 0.056
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.103347, abs=1e-6)
        assert g == pytest.approx(0.000509, abs=1e-6)
        m = 1e-10 - 1e10j
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000321, abs=1e-6)
        assert g == pytest.approx(-0.397357, abs=1e-4)
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-6)
        m = 0 - 1e10j
        x = 0.099
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000321, abs=1e-6)
        assert g == pytest.approx(-0.397357, abs=1e-4)
        x = 0.101
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-4)


class TestAngleScattering:

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

        assert S1[0].real == pytest.approx(0.584080, abs=1e-6)
        assert S1[0].imag == pytest.approx(0.190515, abs=1e-6)
        assert S2[0].real == pytest.approx(0.584080, abs=1e-6)
        assert S2[0].imag == pytest.approx(0.190515, abs=1e-6)
        assert S1[1].real == pytest.approx(0.565702, abs=1e-6)
        assert S1[1].imag == pytest.approx(0.187200, abs=1e-6)
        assert S2[1].real == pytest.approx(0.500161, abs=1e-6)
        assert S2[1].imag == pytest.approx(0.145611, abs=1e-6)
        assert S1[2].real == pytest.approx(0.517525, abs=1e-6)
        assert S1[2].imag == pytest.approx(0.178443, abs=1e-6)
        assert S2[2].real == pytest.approx(0.287964, abs=1e-6)
        assert S2[2].imag == pytest.approx(0.041054, abs=1e-6)
        assert S1[3].real == pytest.approx(0.456340, abs=1e-6)
        assert S1[3].imag == pytest.approx(0.167167, abs=1e-6)
        assert S2[3].real == pytest.approx(0.0362285, abs=1e-6)
        assert S2[3].imag == pytest.approx(-0.0618265, abs=1e-6)
        assert S1[4].real == pytest.approx(0.400212, abs=1e-6)
        assert S1[4].imag == pytest.approx(0.156643, abs=1e-6)
        assert S2[4].real == pytest.approx(-0.174875, abs=1e-6)
        assert S2[4].imag == pytest.approx(-0.122959, abs=1e-6)
        assert S1[5].real == pytest.approx(0.362157, abs=1e-6)
        assert S1[5].imag == pytest.approx(0.149391, abs=1e-6)
        assert S2[5].real == pytest.approx(-0.305682, abs=1e-6)
        assert S2[5].imag == pytest.approx(-0.143846, abs=1e-6)
        assert S1[6].real == pytest.approx(0.348844, abs=1e-6)
        assert S1[6].imag == pytest.approx(0.146829, abs=1e-6)
        assert S2[6].real == pytest.approx(-0.348844, abs=1e-6)
        assert S2[6].imag == pytest.approx(-0.146829, abs=1e-6)

    def test_13_unity_normalization(self):
        """Wiscombe MIEV0 Test Case 14."""
        x = 1.0
        m = 1.5 - 1.0j
        theta = np.arange(0, 181, 30)
        mu = np.cos(theta * np.pi / 180)

        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        S1, S2 = mie.S1_S2(m, x, mu, norm="wiscombe")

        assert S1[0].real == pytest.approx(0.584080, abs=1e-6)
        assert S1[0].imag == pytest.approx(0.190515, abs=1e-6)
        assert S2[0].real == pytest.approx(0.584080, abs=1e-6)
        assert S2[0].imag == pytest.approx(0.190515, abs=1e-6)
        assert S1[1].real == pytest.approx(0.565702, abs=1e-6)
        assert S1[1].imag == pytest.approx(0.187200, abs=1e-6)
        assert S2[1].real == pytest.approx(0.500161, abs=1e-6)
        assert S2[1].imag == pytest.approx(0.145611, abs=1e-6)
        assert S1[2].real == pytest.approx(0.517525, abs=1e-6)
        assert S1[2].imag == pytest.approx(0.178443, abs=1e-6)
        assert S2[2].real == pytest.approx(0.287964, abs=1e-6)
        assert S2[2].imag == pytest.approx(0.041054, abs=1e-6)
        assert S1[3].real == pytest.approx(0.456340, abs=1e-6)
        assert S1[3].imag == pytest.approx(0.167167, abs=1e-6)
        assert S2[3].real == pytest.approx(0.0362285, abs=1e-6)
        assert S2[3].imag == pytest.approx(-0.0618265, abs=1e-6)
        assert S1[4].real == pytest.approx(0.400212, abs=1e-6)
        assert S1[4].imag == pytest.approx(0.156643, abs=1e-6)
        assert S2[4].real == pytest.approx(-0.174875, abs=1e-6)
        assert S2[4].imag == pytest.approx(-0.122959, abs=1e-6)
        assert S1[5].real == pytest.approx(0.362157, abs=1e-6)
        assert S1[5].imag == pytest.approx(0.149391, abs=1e-6)
        assert S2[5].real == pytest.approx(-0.305682, abs=1e-6)
        assert S2[5].imag == pytest.approx(-0.143846, abs=1e-6)
        assert S1[6].real == pytest.approx(0.348844, abs=1e-6)
        assert S1[6].imag == pytest.approx(0.146829, abs=1e-6)
        assert S2[6].real == pytest.approx(-0.348844, abs=1e-6)
        assert S2[6].imag == pytest.approx(-0.146829, abs=1e-6)

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
            assert total / expected[i] == pytest.approx(1.0, abs=4e-3)

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
            assert total / expected[i] == pytest.approx(1, abs=1e-3)

    def test_molecular_hydrogen(self):
        m = 1.00013626
        x = 0.0006403246172921872
        mu = np.linspace(-1, 1, 100)
        ph = mie.i_unpolarized(m, x, mu)
        assert ph[1] == pytest.approx(0.1169791, abs=1e-5)


class TestMiePhaseMatrix:
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


class TestNotebookTests:
    def test_nb1_x(self):
        N = 500
        m = 1.5
        x = np.linspace(0.1, 20, N)  # also in microns
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        assert qsca[0] == pytest.approx(2.3084093592198083e-05, abs=1e-6)
        assert qsca[100] == pytest.approx(4.105960809066763, abs=1e-6)
        assert qsca[200] == pytest.approx(1.9947867190110644, abs=1e-6)
        assert qsca[300] == pytest.approx(2.4652591512196405, abs=1e-6)
        assert qsca[400] == pytest.approx(2.472171798724846, abs=1e-6)
        assert qsca[499] == pytest.approx(2.03583698038088, abs=1e-6)

    def test_nb1_rho(self):
        N = 500
        rho = np.linspace(0.1, 20, N)

        m = 1.5
        x15 = rho / 2 / (m - 1)
        qext, scal5, qback, g = mie.efficiencies_mx(m, x15)

        m = 1.1
        x11 = rho / 2 / (m - 1)
        qext, scal1, qback, g = mie.efficiencies_mx(m, x11)

        assert scal1[0] == pytest.approx(0.0006616369953521216, abs=1e-6)
        assert scal1[99] == pytest.approx(3.449616595439377, abs=1e-6)
        assert scal1[199] == pytest.approx(1.6837703285684387, abs=1e-6)
        assert scal1[299] == pytest.approx(2.3167184401740495, abs=1e-6)
        assert scal1[399] == pytest.approx(2.218210809017406, abs=1e-6)
        assert scal1[499] == pytest.approx(1.876467571615533, abs=1e-6)
        assert scal5[0] == pytest.approx(2.3084093592198083e-05, abs=1e-6)
        assert scal5[99] == pytest.approx(4.07295075914037, abs=1e-6)
        assert scal5[199] == pytest.approx(1.8857586341949146, abs=1e-6)
        assert scal5[299] == pytest.approx(2.464763930426085, abs=1e-6)
        assert scal5[399] == pytest.approx(2.430569030744473, abs=1e-6)
        assert scal5[499] == pytest.approx(2.03583698038088, abs=1e-6)

    def test_nb1_spheres(self):
        N = 500
        m = 1.0
        r = 500  # nm
        lambda0 = np.linspace(300, 800, N)  # also in nm

        mwater = 4 / 3  # rough approximation
        mm = m / mwater
        xx = 2 * np.pi * r * mwater / lambda0

        qext, qsca, qback, g = mie.efficiencies_mx(mm, xx)

        assert qsca[0] == pytest.approx(1.5525047718022498, abs=1e-6)
        assert qsca[99] == pytest.approx(2.1459528526672678, abs=1e-6)
        assert qsca[199] == pytest.approx(2.365171370327149, abs=1e-6)
        assert qsca[299] == pytest.approx(2.2039860928542128, abs=1e-6)
        assert qsca[399] == pytest.approx(1.9261758931397088, abs=1e-6)
        assert qsca[499] == pytest.approx(1.640006561518987, abs=1e-6)

    def test_nb1_mie(self):
        m_sphere = 1.0
        n_water = 4 / 3
        d = 1000
        lambda0 = np.linspace(300, 800)
        qext, qsca, qback, g = mie.efficiencies(m_sphere, d, lambda0, n_water)

        assert qsca[0] == pytest.approx(1.5525047718022498, abs=1e-6)
        assert qsca[9] == pytest.approx(2.107970892634116, abs=1e-6)
        assert qsca[19] == pytest.approx(2.3654333205160074, abs=1e-6)
        assert qsca[29] == pytest.approx(2.213262310704816, abs=1e-6)
        assert qsca[39] == pytest.approx(1.9314911518355427, abs=1e-6)
        assert qsca[49] == pytest.approx(1.640006561518987, abs=1e-6)
