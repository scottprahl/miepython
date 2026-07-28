"""Regression tests for the high-level miepython API.

Everything here goes through ``miepython.core``, which holds no backend-specific
code, so one copy is enough: ``make test`` runs the file once per backend.  Tests
that address the swappable kernels directly live in ``test_kernels.py``, where a
fixture covers both backends in a single process.
"""

import numpy as np
import pytest

import miepython as mie


class TestNonAbsorbing:
    """Test cases for non absorbing behavior."""

    def test_02_wiscombe_conducting(self):

        # MIEV0 Test Case 2
        """Test 02 wiscombe conducting."""
        m = complex(0, 100)
        x = 0.101
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000348, abs=1e-6)
        assert qsca == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-6)

    def test_03_bh_dielectric(self):
        """Test 03 bh dielectric."""
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
        """Test 05 wiscombe non absorbing."""
        m = complex(0.75, 0.0)
        x = 0.099
        s1 = 1.81756e-8 - 1.64810e-4 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000007, abs=1e-6)
        assert g == pytest.approx(0.001448, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 6
        m = complex(0.75, 0.0)
        x = 0.101
        s1 = 2.04875e-08 - 1.74965e-04 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000008, abs=1e-6)
        assert g == pytest.approx(0.001507, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 7
        m = complex(0.75, 0.0)
        x = 10.0
        s1 = -1.07857e00 - 3.60881e-02 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.232265, abs=1e-6)
        assert g == pytest.approx(0.896473, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)
        # MIEV0 Test Case 8
        m = complex(0.75, 0.0)
        x = 1000.0
        s1 = 1.70578e01 + 4.84251e02 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.997908, abs=1e-6)
        assert g == pytest.approx(0.844944, abs=1e-6)
        assert qback == pytest.approx(G, abs=1e-6)

    def test_05_old_wiscombe_non_absorbing(self):

        # OLD MIEV0 Test Case 1
        """Test 05 old wiscombe non absorbing."""
        m = complex(1.5, 0.0)
        x = 10
        s1 = 4.322e00 + 4.868e00 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, _ = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.8820, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-4)
        # OLD MIEV0 Test Case 2
        m = complex(1.5, 0.0)
        x = 100
        s1 = 4.077e01 + 5.175e01 * 1j
        G = abs(2 * s1 / x) ** 2
        _, qsca, qback, _ = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0944, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-4)
        # OLD MIEV0 Test Case 3
        m = complex(1.5, 0.0)
        x = 1000
        G = 4 * 2.576e06 / x**2
        _, qsca, qback, _ = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0139, abs=1e-4)
        assert qback == pytest.approx(G, abs=1e-3)
        # OLD MIEV0 Test Case 4
        m = complex(1.5, 0.0)
        x = 5000.0
        G = 4 * 2.378e08 / x**2
        _, qsca, qback, _ = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.0086, abs=1e-4)
        assert qback == pytest.approx(G, abs=3e-3)

    def test_04_non_dielectric(self):
        """Test 04 non dielectric."""
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
    """Test cases for absorbing behavior."""

    def test_06_wiscombe_water_absorbing(self):

        # MIEV0 Test Case 9
        """Test 06 wiscombe water absorbing."""
        m = complex(1.33, -0.00001)
        x = 1.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.093923, abs=1e-6)
        assert g == pytest.approx(0.184517, abs=1e-6)
        # MIEV0 Test Case 10
        m = complex(1.33, -0.00001)
        x = 100.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.096594, abs=1e-6)
        assert g == pytest.approx(0.868959, abs=1e-6)
        # MIEV0 Test Case 11
        m = complex(1.33, -0.00001)
        x = 10000.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert g == pytest.approx(0.907840, abs=1e-6)
        assert qsca == pytest.approx(1.723857, abs=1e-6)

    def test_07_wiscombe_absorbing(self):

        # MIEV0 Test Case 12
        """Test 07 wiscombe absorbing."""
        m = 1.5 - 1j
        x = 0.055
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.101491, abs=1e-6)
        assert qsca == pytest.approx(0.000011, abs=1e-6)
        assert g == pytest.approx(0.000491, abs=1e-6)
        # MIEV0 Test Case 13
        m = 1.5 - 1j
        x = 0.056
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.1033467, abs=1e-6)
        assert qsca == pytest.approx(0.000012, abs=1e-6)
        assert g == pytest.approx(0.000509, abs=1e-6)
        # MIEV0 Test Case 14
        m = 1.5 - 1j
        x = 1
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(2.336321, abs=1e-6)
        assert qsca == pytest.approx(0.6634538, abs=1e-6)
        assert g == pytest.approx(0.192136, abs=1e-6)
        # MIEV0 Test Case 15
        m = 1.5 - 1j
        x = 100
        x = 100.0
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(2.097502, abs=1e-6)
        assert qsca == pytest.approx(1.283697, abs=1e-3)
        assert qext == pytest.approx(2.097502, abs=1e-2)
        assert g == pytest.approx(0.850252, abs=1e-3)
        # MIEV0 Test Case 16
        m = 1.5 - 1j
        x = 10000
        qext, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.236575, abs=1e-6)
        assert qext == pytest.approx(2.004368, abs=1e-6)
        assert g == pytest.approx(0.846309, abs=1e-6)

    def test_08_wiscombe_more_absorbing(self):

        # MIEV0 Test Case 17
        """Test 08 wiscombe more absorbing."""
        m = 10.0 - 10.0j
        x = 1.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.049405, abs=1e-6)
        assert g == pytest.approx(-0.110664, abs=1e-6)
        # MIEV0 Test Case 18
        m = 10.0 - 10.0j
        x = 100.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.836785, abs=1e-6)
        assert g == pytest.approx(0.556215, abs=1e-6)
        # MIEV0 Test Case 19
        m = 10.0 - 10.0j
        x = 10000.0
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(1.795393, abs=1e-6)
        assert g == pytest.approx(0.548194, abs=1e-6)

    def test_09_single_nonmagnetic(self):
        """Test 09 single nonmagnetic."""
        m = 1.5 - 0.5j
        x = 2.5
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)

        assert qext == pytest.approx(2.562873497454734, abs=1e-7)
        assert qsca == pytest.approx(1.097071819088392, abs=1e-7)
        assert qback == pytest.approx(0.123586468179818, abs=1e-7)
        assert g == pytest.approx(0.748905978948507, abs=1e-7)


class TestPerfectlyConducting:
    """Test cases for perfectly conducting behavior."""

    @pytest.mark.parametrize("m", [1000j, -10000j])
    def test_11_wiscombe_perfectly_conducting(self, m):
        """Test 11 wiscombe perfectly conducting.

        A zero real part is what signals a perfect conductor, so neither the sign
        nor the magnitude of the imaginary part should matter.  The two halves of
        the old JIT/no-JIT pair happened to use different values; both are kept.
        """
        # MIEV0 Test Case 0
        x = 0.001
        _, qsca, _, _ = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(3.3333e-12, abs=1e-13)
        # MIEV0 Test Case 1
        x = 0.099
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000321, abs=1e-4)
        assert g == pytest.approx(-0.397357, abs=1e-3)
        # MIEV0 Test Case 2
        x = 0.101
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-6)
        # MIEV0 Test Case 3
        x = 100
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.008102, abs=1e-6)
        assert g == pytest.approx(0.500926, abs=1e-6)
        # MIEV0 Test Case 4
        x = 10000
        _, qsca, _, g = mie.efficiencies_mx(m, x)
        assert qsca == pytest.approx(2.000289, abs=1e-6)
        assert g == pytest.approx(0.500070, abs=1e-6)


class TestSmall:
    """Test cases for small behavior."""

    def test_10_small_spheres(self):
        # MIEV0 Test Case 5
        """Test 10 small spheres."""
        m = 0.75
        x = 0.099
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000007, abs=1e-6)
        assert g == pytest.approx(0.001448, abs=1e-6)
        # MIEV0 Test Case 6
        m = 0.75
        x = 0.101
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000008, abs=1e-6)
        assert g == pytest.approx(0.001507, abs=1e-6)
        m = 1.5 - 1j
        x = 0.055
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.101491, abs=1e-6)
        assert g == pytest.approx(0.000491, abs=1e-6)
        x = 0.056
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.103347, abs=1e-6)
        assert g == pytest.approx(0.000509, abs=1e-6)
        m = 1e-10 - 1e10j
        x = 0.099
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000321, abs=1e-6)
        assert g == pytest.approx(-0.397357, abs=1e-4)
        x = 0.101
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-6)
        m = 0 - 1e10j
        x = 0.099
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000321, abs=1e-6)
        assert g == pytest.approx(-0.397357, abs=1e-4)
        x = 0.101
        qext, _, _, g = mie.efficiencies_mx(m, x)
        assert qext == pytest.approx(0.000348, abs=1e-6)
        assert g == pytest.approx(-0.397262, abs=1e-4)


class TestAngleScattering:
    """Test cases for angle scattering behavior."""

    def test_12_scatter_function(self):
        """Test 12 scatter function."""
        x = 1.0
        m = 1.5 - 1.0j
        theta = np.arange(0, 181, 30)
        mu = np.cos(theta * np.pi / 180)

        _, _, _, _ = mie.efficiencies_mx(m, x)
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

    def test_13_unity_normalization(self):
        """Test 13 unity normalization."""
        x = 1.0
        m = 1.5 - 1.0j
        theta = np.arange(0, 181, 30)
        mu = np.cos(theta * np.pi / 180)

        _, _, _, _ = mie.efficiencies_mx(m, x)
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
        """Test i unpolarized 01."""
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
        """Test i par i per 01."""
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
        """Test molecular hydrogen."""
        m = 1.00013626
        x = 0.0006403246172921872
        mu = np.linspace(-1, 1, 100)
        ph = mie.i_unpolarized(m, x, mu)
        assert ph[1] == pytest.approx(0.1169791, abs=1e-5)


class TestMiePhaseMatrix:
    """Test cases for mie phase matrix behavior."""

    def test_phase_matrix_basic(self):
        """Element (0, 0) of array returned by phase_matrix should match i_unpolarized."""
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 1000)

        p = mie.phase_matrix(m, x, mu)  # result to be validated
        p00 = mie.i_unpolarized(m, x, mu)  # reference result

        assert np.allclose(p[0, 0], p00, rtol=1e-9)

    def test_phase_matrix_mu_scalar(self):
        """phase_matrix returns (4, 4) array when mu is scalar."""
        assert mie.phase_matrix(m=1.5, x=2.0, mu=0.0).shape == (4, 4)

    def test_phase_matrix_symmetry(self):
        """Upper left 2X2 block is symmetric and lower right 2X2 block is antisymmetric."""
        p = mie.phase_matrix(m=1.5, x=2.0, mu=np.linspace(-1, 1, 10))
        assert np.allclose(p[0, 1], p[1, 0])
        assert np.allclose(p[2, 3], -p[3, 2])

    def test_phase_matrix_unity(self):
        """Element p[0, 0, :]**2 = sum of squares of other three."""
        m = 1.5 - 1.5j
        x = 2
        mu = np.linspace(-1, 1, 1000)

        p = mie.phase_matrix(m, x, mu)  # result to be validated

        assert np.allclose(p[0, 0] ** 2, p[0, 1] ** 2 + p[2, 2] ** 2 + p[2, 3] ** 2, rtol=1e-9)

    def test_phase_matrix_bohren(self):
        """Compare with output from Bohren's program.

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
    """Test cases for notebook tests."""

    def test_nb1_x(self):
        """Test nb1 x."""
        N = 500
        m = 1.5
        x = np.linspace(0.1, 20, N)  # also in microns
        _, qsca, _, _ = mie.efficiencies_mx(m, x)

        assert qsca[0] == pytest.approx(2.3084093592198083e-05, abs=1e-6)
        assert qsca[100] == pytest.approx(4.105960809066763, abs=1e-6)
        assert qsca[200] == pytest.approx(1.9947867190110644, abs=1e-6)
        assert qsca[300] == pytest.approx(2.4652591512196405, abs=1e-6)
        assert qsca[400] == pytest.approx(2.472171798724846, abs=1e-6)
        assert qsca[499] == pytest.approx(2.03583698038088, abs=1e-6)

    def test_nb1_rho(self):
        """Test nb1 rho."""
        N = 500
        rho = np.linspace(0.1, 20, N)

        m = 1.5
        x15 = rho / 2 / (m - 1)
        _, scal5, _, _ = mie.efficiencies_mx(m, x15)

        m = 1.1
        x11 = rho / 2 / (m - 1)
        _, scal1, _, _ = mie.efficiencies_mx(m, x11)
        print(x11)

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
        """Test nb1 spheres."""
        N = 500
        m = 1.0
        r = 500  # nm
        lambda0 = np.linspace(300, 800, N)  # also in nm

        mwater = 4 / 3  # rough approximation
        mm = m / mwater
        xx = 2 * np.pi * r * mwater / lambda0
        _, qsca, _, _ = mie.efficiencies_mx(mm, xx)

        assert qsca[0] == pytest.approx(1.5525047718022498, abs=1e-6)
        assert qsca[99] == pytest.approx(2.1459528526672678, abs=1e-6)
        assert qsca[199] == pytest.approx(2.365171370327149, abs=1e-6)
        assert qsca[299] == pytest.approx(2.2039860928542128, abs=1e-6)
        assert qsca[399] == pytest.approx(1.9261758931397088, abs=1e-6)
        assert qsca[499] == pytest.approx(1.640006561518987, abs=1e-6)

    def test_nb1_mie(self):
        """Test nb1 mie."""
        m_sphere = 1.0
        n_water = 4 / 3
        d = 1000
        lambda0 = np.linspace(300, 800)
        _, qsca, _, _ = mie.efficiencies(m_sphere, d, lambda0, n_water)

        assert qsca[0] == pytest.approx(1.5525047718022498, abs=1e-6)
        assert qsca[9] == pytest.approx(2.107970892634116, abs=1e-6)
        assert qsca[19] == pytest.approx(2.3654333205160074, abs=1e-6)
        assert qsca[29] == pytest.approx(2.213262310704816, abs=1e-6)
        assert qsca[39] == pytest.approx(1.9314911518355427, abs=1e-6)
        assert qsca[49] == pytest.approx(1.640006561518987, abs=1e-6)


class TestMultipoles:
    """Test that n_pole isolates the requested multipole order."""

    def test_multipole_sum_matches_full_series(self):
        """Summing every multipole reproduces the complete Mie series."""
        m = 1.5 - 0.01j
        x = 3.0
        mu = np.linspace(-1, 1, 11)
        n_max = len(mie.an_bn(complex(m.real, -abs(m.imag)), x, 0)[0])

        S1_sum = np.zeros_like(mu, dtype=complex)
        S2_sum = np.zeros_like(mu, dtype=complex)
        for n in range(1, n_max + 1):
            s1, s2 = mie.S1_S2(m, x, mu, norm="wiscombe", n_pole=n)
            S1_sum += s1
            S2_sum += s2

        S1, S2 = mie.S1_S2(m, x, mu, norm="wiscombe", n_pole=0)
        np.testing.assert_allclose(S1_sum, S1, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(S2_sum, S2, rtol=1e-10, atol=1e-12)

    def test_multipole_closed_form(self):
        """n_pole=n pairs the nth Mie coefficient with pi_n and tau_n."""
        m = 1.5 - 0.01j
        x = 3.0
        mu = np.array([1.0, 0.3, -0.7])
        a, b = mie.an_bn(complex(m.real, -abs(m.imag)), x, 0)

        # pi_1 = 1, tau_1 = mu, pi_2 = 3 mu, tau_2 = 3 (2 mu^2 - 1)
        cases = ((1, 1 + 0 * mu, mu), (2, 3 * mu, 3 * (2 * mu**2 - 1)))
        for n, pi_n, tau_n in cases:
            scale = (2 * n + 1) / (n * (n + 1))
            S1, S2 = mie.S1_S2(m, x, mu, norm="wiscombe", n_pole=n)
            # an_bn returns conjugated coefficients, S1_S2 does not
            expected_S1 = np.conjugate(scale * (a[n - 1] * pi_n + b[n - 1] * tau_n))
            expected_S2 = np.conjugate(scale * (a[n - 1] * tau_n + b[n - 1] * pi_n))
            np.testing.assert_allclose(S1, expected_S1, rtol=1e-12)
            np.testing.assert_allclose(S2, expected_S2, rtol=1e-12)

    def test_multipole_optical_theorem(self):
        """S1_S2 and efficiencies_mx agree on which order n_pole selects."""
        m = 1.5 - 0.01j
        x = 3.0
        for n in range(1, 5):
            qext_e, _, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=True)
            qext_m, _, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=False)
            S1, _ = mie.S1_S2(m, x, np.array([1.0]), norm="wiscombe", n_pole=n)
            assert qext_e + qext_m == pytest.approx(4 * S1[0].real / x**2, abs=1e-12)

    def test_multipole_out_of_range(self):
        """Orders outside the truncated series are rejected."""
        m = 1.5 - 0.01j
        x = 3.0
        mu = np.array([0.5])
        n_terms = len(mie.an_bn(complex(m.real, -abs(m.imag)), x, 0)[0])

        with pytest.raises(ValueError):
            mie.S1_S2(m, x, mu, n_pole=-1)
        with pytest.raises(ValueError):
            mie.S1_S2(m, x, mu, n_pole=n_terms + 1)


class TestElectricMagneticMultipoles:
    """Test the e_field selector between electric (a_n) and magnetic (b_n) multipoles."""

    def test_electric_and_magnetic_closed_form(self):
        """e_field picks a_n when True and b_n when False."""
        m = 1.5 - 0.01j
        x = 3.0
        a, b = mie.an_bn(complex(m.real, -abs(m.imag)), x, 0)

        for n in range(1, 4):
            cn = 2 * n + 1
            qext_e, qsca_e, qback_e, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=True)
            qext_m, qsca_m, qback_m, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=False)

            assert qext_e == pytest.approx(2 * cn * a[n - 1].real / x**2, abs=1e-12)
            assert qext_m == pytest.approx(2 * cn * b[n - 1].real / x**2, abs=1e-12)
            assert qsca_e == pytest.approx(2 * cn * abs(a[n - 1]) ** 2 / x**2, abs=1e-12)
            assert qsca_m == pytest.approx(2 * cn * abs(b[n - 1]) ** 2 / x**2, abs=1e-12)
            assert qback_e == pytest.approx((cn * abs(a[n - 1])) ** 2 / x**2, abs=1e-12)
            assert qback_m == pytest.approx((cn * abs(b[n - 1])) ** 2 / x**2, abs=1e-12)

    def test_electric_and_magnetic_differ(self):
        """The e_field flag changes the result instead of being ignored."""
        m = 1.5 - 0.01j
        x = 3.0
        for n in range(1, 4):
            qext_e, qsca_e, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=True)
            qext_m, qsca_m, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=False)
            assert qext_e != pytest.approx(qext_m, abs=1e-6)
            assert qsca_e != pytest.approx(qsca_m, abs=1e-6)

    def test_multipoles_sum_to_full_efficiencies(self):
        """Every order and both parities add up to the complete result."""
        for m, x in ((1.5 - 0.01j, 3.0), (1.33, 5.0), (1.5 - 0.5j, 2.0)):
            mm = complex(np.real(m), -abs(np.imag(m)))
            n_terms = len(mie.an_bn(mm, x, 0)[0])

            qext_sum = 0.0
            qsca_sum = 0.0
            for n in range(1, n_terms):
                for e_field in (True, False):
                    qext, qsca, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=e_field)
                    qext_sum += qext
                    qsca_sum += qsca

            qext, qsca, _, _ = mie.efficiencies_mx(m, x)
            assert qext_sum == pytest.approx(qext, rel=1e-10)
            assert qsca_sum == pytest.approx(qsca, rel=1e-10)

    def test_lossless_qsca_matches_modulus_form(self):
        """For a lossless sphere the qsca=qext shortcut equals 2(2n+1)|a_n|^2/x^2."""
        m = 1.33
        x = 5.0
        a, b = mie.an_bn(complex(m, 0), x, 0)
        for n in range(1, 4):
            cn = 2 * n + 1
            _, qsca_e, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=True)
            _, qsca_m, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=False)
            assert qsca_e == pytest.approx(2 * cn * abs(a[n - 1]) ** 2 / x**2, abs=1e-12)
            assert qsca_m == pytest.approx(2 * cn * abs(b[n - 1]) ** 2 / x**2, abs=1e-12)

    def test_single_multipole_asymmetry_is_zero(self):
        """Asymmetry is exactly zero for an isolated multipole, on both code paths."""
        m = 1.5 - 0.01j
        x = 3.0
        for n in range(1, 5):
            for e_field in (True, False):
                _, _, _, g = mie.efficiencies_mx(m, x, n_pole=n, e_field=e_field)
                assert g == 0.0

        # the array path must not turn g into nan
        _, _, _, g = mie.efficiencies_mx(np.array([1.5 - 0.01j, 1.33]), np.array([3.0, 2.0]), n_pole=1)
        assert np.all(np.isfinite(g))
        np.testing.assert_array_equal(g, np.zeros(2))

    def test_e_field_ignored_for_full_series(self):
        """e_field has no effect when every multipole is included."""
        m = 1.5 - 0.01j
        x = 3.0
        assert mie.efficiencies_mx(m, x, n_pole=0, e_field=True) == mie.efficiencies_mx(m, x, n_pole=0, e_field=False)

    def test_lossless_qext_equals_qsca(self):
        """A lossless sphere cannot absorb, so qext and qsca must agree."""
        for m in (1.05, 1.2, 1.33, 1.5, 2.0, 3.0, 0.75):
            for x in (0.1, 1.0, 5.0, 20.0, 62.0, 200.0):
                qext, qsca, _, _ = mie.efficiencies_mx(m, x)
                assert qext == pytest.approx(qsca, rel=1e-13)

    def test_per_multipole_extinction_is_positive(self):
        """Every isolated multipole must extinguish a non-negative amount."""
        for m in (1.05, 1.33, 1.5):
            for x in (0.1, 0.5, 2.0):
                for n in range(1, 4):
                    for e_field in (True, False):
                        qext, _, _, _ = mie.efficiencies_mx(m, x, n_pole=n, e_field=e_field)
                        assert qext >= 0.0, f"m={m} x={x} n_pole={n} e_field={e_field}"


class TestConductingInternalCoefficients:
    """Test that cn_dn returns no internal field for a perfectly conducting sphere."""

    def test_conducting_coefficients_via_public_api(self):
        """coefficients(internal=True) is zero for the internal terms only."""
        a, b, c, d = mie.coefficients(complex(0.0, 100.0), 1.0, internal=True)
        assert np.any(a != 0) or np.any(b != 0)
        np.testing.assert_array_equal(c, np.zeros_like(c))
        np.testing.assert_array_equal(d, np.zeros_like(d))


class TestSeriesTruncation:
    """Test that every returned coefficient is a real term, with no padding."""

    def test_highest_multipole_is_usable(self):
        """n_pole may address the highest order now that nothing is padding."""
        m, x = complex(1.5, -0.01), 3.0
        n_max = len(mie.an_bn(complex(m.real, -abs(m.imag)), x, 0)[0])
        mu = np.array([0.4])
        s1, s2 = mie.S1_S2(m, x, mu, norm="wiscombe", n_pole=n_max)
        assert np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))
        assert s1[0] != 0
        with pytest.raises(ValueError):
            mie.S1_S2(m, x, mu, n_pole=n_max + 1)


class TestDegenerateSpheres:
    """Test spheres that do not scatter: zero size or index matched."""

    def test_zero_size_efficiencies_are_zero(self):
        """Every efficiency tends to zero as x goes to zero, qback included."""
        for m in (1.5 - 0.1j, 1.33, complex(0.0, 100.0)):
            q = mie.efficiencies_mx(m, 0.0)
            assert np.all(np.isfinite(q)), m
            assert all(v == 0 for v in q), m

    def test_zero_size_inside_an_array(self):
        """One zero size parameter must not poison the rest of the sweep."""
        qext, qsca, qback, g = mie.efficiencies_mx(1.5 - 0.1j, np.array([0.0, 1.0, 2.0]))
        for arr in (qext, qsca, qback, g):
            assert np.all(np.isfinite(arr))
        assert qback[0] == 0.0
        assert qback[1] > 0.0

    def test_index_matched_normalization_raises(self):
        """A sphere matching its surroundings has nothing to normalize against."""
        mu = np.array([0.5])
        for norm in ("albedo", "one", "4pi", "qext"):
            with pytest.raises(ValueError, match="scatters"):
                mie.i_unpolarized(1.0, 2.0, mu, norm=norm)

    def test_zero_size_normalization_raises(self):
        """Same for a sphere of zero size."""
        mu = np.array([0.5])
        for norm in ("albedo", "one", "4pi", "qext"):
            with pytest.raises(ValueError, match="scatters"):
                mie.i_unpolarized(1.5, 0.0, mu, norm=norm)

    def test_unnormalized_choices_still_work(self):
        """The norms that do not divide by an efficiency stay usable."""
        mu = np.array([0.5])
        for norm in ("qsca", "wiscombe", "bohren"):
            value = mie.i_unpolarized(1.0, 2.0, mu, norm=norm)
            assert np.all(np.isfinite(value))
            assert value[0] == pytest.approx(0.0, abs=1e-20)

    def test_ordinary_sphere_unaffected(self):
        """The guard must not fire for a sphere that does scatter."""
        mu = np.linspace(-1, 1, 5)
        for norm in ("albedo", "one", "4pi", "qext", "qsca", "wiscombe", "bohren"):
            value = mie.i_unpolarized(1.5 - 0.01j, 2.0, mu, norm=norm)
            assert np.all(np.isfinite(value))
            assert np.all(value > 0)


class TestArrayDispatchAndErrors:
    """Test core.py's array handling and the errors it is supposed to raise."""

    def test_wiscombe_terms_matches_the_criterion(self):
        """The named helper must agree with the expression the kernels inline."""
        for x in (0.1, 1.0, 6.2832, 20.0, 200.0):
            assert mie.core.wiscombe_terms(x) == int(x + 4.05 * x**0.33333 + 2.0)
        # and with the number of coefficients actually returned
        for x in (0.5, 5.0, 50.0):
            assert len(mie.an_bn(complex(1.5, -0.01), x, 0)[0]) == mie.core.wiscombe_terms(x)

    def test_coefficients_rejects_mismatched_array_lengths(self):
        """Two arrays of different length cannot be paired up."""
        with pytest.raises(RuntimeError, match="same length"):
            mie.coefficients(np.array([1.5, 1.4, 1.3]), np.array([2.0, 3.0]), n_pole=2)

    def test_coefficients_requires_n_pole_for_arrays(self):
        """Array input has no single term count to fall back on."""
        with pytest.raises(RuntimeError, match="n_pole must be"):
            mie.coefficients(np.array([1.5, 1.4]), np.array([2.0, 3.0]))

    @pytest.mark.parametrize("internal,expected_rows", [(False, 2), (True, 4)])
    def test_coefficients_array_shapes(self, internal, expected_rows):
        """The internal flag decides whether c and d come back with a and b."""
        m_arr = np.array([1.5 - 0.01j, 1.4 - 0.02j])
        x_arr = np.array([2.0, 3.0])
        got = mie.coefficients(m_arr, x_arr, n_pole=3, internal=internal)
        assert np.shape(got) == (expected_rows, 2, 3)

    @pytest.mark.parametrize("internal", [False, True])
    def test_coefficients_array_matches_scalar_calls(self, internal):
        """Each row must be what the scalar call gives for that pair."""
        m_arr = np.array([1.5 - 0.01j, 1.4 - 0.02j])
        x_arr = np.array([2.0, 3.0])
        stacked = mie.coefficients(m_arr, x_arr, n_pole=3, internal=internal)
        for i, (m_one, x_one) in enumerate(zip(m_arr, x_arr)):
            single = mie.coefficients(m_one, x_one, n_pole=3, internal=internal)
            for row in range(stacked.shape[0]):
                np.testing.assert_allclose(stacked[row][i], single[row], rtol=1e-13)

    def test_coefficients_broadcasts_a_scalar_against_an_array(self):
        """One scalar and one array is allowed; the scalar is reused."""
        x_arr = np.array([2.0, 3.0])
        by_array = mie.coefficients(1.5, x_arr, n_pole=3)
        for i, x_one in enumerate(x_arr):
            np.testing.assert_allclose(by_array[0][i], mie.coefficients(1.5, x_one, n_pole=3)[0], rtol=1e-13)

        m_arr = np.array([1.5 - 0.01j, 1.4 - 0.02j])
        by_index = mie.coefficients(m_arr, 2.0, n_pole=3)
        for i, m_one in enumerate(m_arr):
            np.testing.assert_allclose(by_index[0][i], mie.coefficients(m_one, 2.0, n_pole=3)[0], rtol=1e-13)

    def test_efficiencies_mx_rejects_mismatched_array_lengths(self):
        """Same rule for the efficiencies."""
        with pytest.raises(RuntimeError, match="same length"):
            mie.efficiencies_mx(np.array([1.5, 1.4, 1.3]), np.array([2.0, 3.0]))

    @pytest.mark.parametrize(
        "m_in,x_in",
        [
            (np.array([1.5 - 0.01j, 1.4 - 0.02j]), 2.0),
            (1.5 - 0.01j, np.array([2.0, 3.0])),
            (np.array([1.5 - 0.01j, 1.4 - 0.02j]), np.array([2.0, 3.0])),
        ],
        ids=["array m", "array x", "both arrays"],
    )
    def test_efficiencies_mx_array_matches_scalar_calls(self, m_in, x_in):
        """Every mix of scalar and array must agree with the scalar calls."""
        qext, qsca, qback, g = mie.efficiencies_mx(m_in, x_in)
        assert len(qext) == len(qsca) == len(qback) == len(g) == 2
        for i in range(2):
            m_one = m_in[i] if np.ndim(m_in) else m_in
            x_one = x_in[i] if np.ndim(x_in) else x_in
            want = mie.efficiencies_mx(m_one, x_one)
            for got, expected in zip((qext[i], qsca[i], qback[i], g[i]), want):
                assert got == pytest.approx(expected, rel=1e-13)

    def test_unknown_normalization_is_rejected(self):
        """A misspelt normalization must list the valid choices."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            mie.S1_S2(1.5, 2.0, np.array([0.5]), norm="nonsense")

    @pytest.mark.parametrize("norm", ["albedo", "one", "4pi", "qext", "qsca", "wiscombe", "bohren"])
    def test_normalization_aliases_and_case_are_accepted(self, norm):
        """The lookup lowercases, so capitalised spellings must work too."""
        mu = np.linspace(-1, 1, 5)
        lower = mie.i_unpolarized(1.5 - 0.01j, 2.0, mu, norm=norm)
        upper = mie.i_unpolarized(1.5 - 0.01j, 2.0, mu, norm=norm.upper())
        np.testing.assert_allclose(upper, lower, rtol=1e-13)

    @pytest.mark.parametrize("norm", ["albedo", "wiscombe"])
    def test_sign_of_the_imaginary_index_does_not_matter(self, norm):
        """The convention is m = n - ik, so a positive imaginary part is conjugated."""
        mu = np.linspace(-1, 1, 5)
        physical, flipped = 1.5 - 0.1j, 1.5 + 0.1j
        np.testing.assert_allclose(
            mie.S1_S2(flipped, 2.0, mu, norm=norm), mie.S1_S2(physical, 2.0, mu, norm=norm), rtol=1e-13
        )
        np.testing.assert_allclose(mie.efficiencies_mx(flipped, 2.0), mie.efficiencies_mx(physical, 2.0), rtol=1e-13)
