"""Regression tests for the Mie coefficients through the public API.

The kernel-level coefficient tests are parametrized over both backends in
``test_kernels.py``; these go through ``miepython.core`` and need only one copy.
"""

import numpy as np
import pytest

from reference import basic_abcd, bohren_ab, pyscatt_ab

import miepython as mie
from miepython.bessel import (
    d_riccati_bessel_h1,
    d_riccati_bessel_jn,
    riccati_bessel_h1,
    riccati_bessel_jn,
)
from miepython.util import cs


class Test1:
    """Regression test cases."""

    def test_09an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        """Test 09an bn."""
        m = 8.9 - 0.69j
        xx = np.array([0.385, 0.390, 0.395])
        a1 = np.array([0.0024 + 0.0410j, 0.0027 + 0.0428j, 0.0030 + 0.0447j])
        b1 = np.array([0.0258 - 0.0359j, 0.0231 - 0.0361j, 0.0208 - 0.0361j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        for n, x in enumerate(xx):
            a, b = mie.an_bn(m, x, 0)
            ab, bb, qsca_b = bohren_ab(m, x)
            ap, bp, _ = pyscatt_ab(m, x)
            at, bt, _, _ = basic_abcd(m, x, 4)
            _, qsca, _, _ = mie.efficiencies_mx(m, x)

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

            assert a[0].real == pytest.approx(a1[n].real, abs=1e-3)
            assert a[0].imag == pytest.approx(a1[n].imag, abs=1e-3)
            assert b[0].real == pytest.approx(b1[n].real, abs=1e-3)
            assert b[0].imag == pytest.approx(b1[n].imag, abs=1e-3)
            assert at[0].real == pytest.approx(a1[n].real, abs=1e-3)
            assert at[0].imag == pytest.approx(a1[n].imag, abs=1e-3)
            assert bt[0].real == pytest.approx(b1[n].real, abs=1e-3)
            assert bt[0].imag == pytest.approx(b1[n].imag, abs=1e-3)

    def test_08an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        """Test 08an bn."""
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

        qext, qsca, _, _ = mie.efficiencies_mx(m, x)

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

        assert qsca == pytest.approx(qsca_lowan, abs=1e-3)
        assert qext == pytest.approx(qext_lowan, abs=1e-3)
        assert at[0].real == pytest.approx(a1[0].real, abs=1e-3)
        assert at[0].imag == pytest.approx(a1[0].imag, abs=1e-3)
        assert bt[0].real == pytest.approx(b1[0].real, abs=1e-3)
        assert bt[0].imag == pytest.approx(b1[0].imag, abs=1e-3)


class Test_Weak_Absorbing_An_and_Bn:
    """Test cases for weak absorbing an and bn behavior."""

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
            assert ab[n].real == pytest.approx(a[n].real, abs=1e-5)
            assert ab[n].imag == pytest.approx(a[n].imag, abs=1e-5)
            assert bb[n].real == pytest.approx(b[n].real, abs=1e-5)
            assert bb[n].imag == pytest.approx(b[n].imag, abs=1e-5)

    def test_01an_bn(self):
        # Wiscombe MIEV0 Test Case 9
        """Test 01an bn."""
        m = 1.330 - 0.00001j
        x = 1
        qsca_wisc = 0.093923
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        _, qsca, _, _ = mie.efficiencies_mx(m, x)

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

        assert qsca == pytest.approx(qsca_wisc, abs=1e-5)
        for n, a_n in enumerate(a):
            assert a_n.real == pytest.approx(at[n].real, abs=0.00001)
            assert a_n.imag == pytest.approx(at[n].imag, abs=0.00001)
            assert b[n].real == pytest.approx(bt[n].real, abs=0.00001)
            assert b[n].imag == pytest.approx(bt[n].imag, abs=0.00001)


class Test_Medium_Absorbing_An_and_Bn:
    """Test cases for medium absorbing an and bn behavior."""

    def test_01an_bn(self):
        # Wiscombe MIEV0 Test Case 12
        """Test 01an bn."""
        m = 1.5 - 1.0j
        x = 0.055
        qext_wisc = 0.101491
        qsca_wisc = 0.000011
        a, b = mie.an_bn(m, x, 0)
        ab, bb, qsca_b = bohren_ab(m, x)
        ap, bp, qsca_p = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        qext, qsca, _, _ = mie.efficiencies_mx(m, x)

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

        assert qsca == pytest.approx(qsca_wisc, abs=1e-5)
        assert qext == pytest.approx(qext_wisc, abs=1e-5)
        for n, a_n in enumerate(a):
            assert a_n.real == pytest.approx(at[n].real, abs=0.00000001)
            assert a_n.imag == pytest.approx(at[n].imag, abs=0.00000001)
            assert b[n].real == pytest.approx(bt[n].real, abs=0.00000001)
            assert b[n].imag == pytest.approx(bt[n].imag, abs=0.00000001)

    def test_09an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        """Test 09an bn."""
        m = 8.9 - 0.69j
        xx = np.array([0.385, 0.390, 0.395])
        a1 = np.array([0.0024 + 0.0410j, 0.0027 + 0.0428j, 0.0030 + 0.0447j])
        b1 = np.array([0.0258 - 0.0359j, 0.0231 - 0.0361j, 0.0208 - 0.0361j])
        a1 = np.conjugate(a1)
        b1 = np.conjugate(b1)

        for n, x in enumerate(xx):
            a, b = mie.an_bn(m, x, 0)
            ab, bb, qsca_b = bohren_ab(m, x)
            ap, bp, _ = pyscatt_ab(m, x)
            at, bt, _, _ = basic_abcd(m, x, len(a))
            _, qsca, _, _ = mie.efficiencies_mx(m, x)

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

            assert a[0].real == pytest.approx(a1[n].real, abs=1e-3)
            assert a[0].imag == pytest.approx(a1[n].imag, abs=1e-3)
            assert b[0].real == pytest.approx(b1[n].real, abs=1e-3)
            assert b[0].imag == pytest.approx(b1[n].imag, abs=1e-3)
            assert at[0].real == pytest.approx(a1[n].real, abs=1e-3)
            assert at[0].imag == pytest.approx(a1[n].imag, abs=1e-3)
            assert bt[0].real == pytest.approx(b1[n].real, abs=1e-3)
            assert bt[0].imag == pytest.approx(b1[n].imag, abs=1e-3)

    def test_08an_bn(self):
        # From Lowan "Tables of Scattering Functions for Spherical Particles"
        """Test 08an bn."""
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

        qext, qsca, _, _ = mie.efficiencies_mx(m, x)

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

        assert qsca == pytest.approx(qsca_lowan, abs=1e-3)
        assert qext == pytest.approx(qext_lowan, abs=1e-3)
        assert a[0].real == pytest.approx(a1[0].real, abs=1e-3)
        assert a[0].imag == pytest.approx(a1[0].imag, abs=1e-3)
        assert b[0].real == pytest.approx(b1[0].real, abs=1e-3)
        assert b[0].imag == pytest.approx(b1[0].imag, abs=1e-3)

    def test_10an_bn(self):
        # From van de Hulst Table 28
        """Test 10an bn."""
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
        ap, bp, _ = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))
        qext, qsca, _, _ = mie.efficiencies_mx(m, x)

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
            assert a[n].real == pytest.approx(a1[n].real, abs=1e-3)
            assert a[n].imag == pytest.approx(a1[n].imag, abs=1e-3)
            assert b[n].real == pytest.approx(b1[n].real, abs=1e-3)
            assert b[n].imag == pytest.approx(b1[n].imag, abs=1e-3)


class TestAnBnCnDnLengths:
    """Unit tests for Mie coefficients cn and dn."""

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

            expected = int(x + 4.05 * x**0.33333 + 2.0)

            assert len(a) == expected, "Length mismatch for a_n with n_pole=0"
            assert len(b) == expected, "Length mismatch for b_n with n_pole=0"
            assert len(c) == expected, "Length mismatch for c_n with n_pole=0"
            assert len(d) == expected, "Length mismatch for d_n with n_pole=0"
        for m, x, n_pole in test_cases:
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)

            expected = int(x + 4.05 * x**0.33333 + 2.0)

            assert len(a) == expected, "Length mismatch for a_n with n_pole=0"
            assert len(b) == expected, "Length mismatch for b_n with n_pole=0"
            assert len(c) == expected, "Length mismatch for c_n with n_pole=0"
            assert len(d) == expected, "Length mismatch for d_n with n_pole=0"

    def test_lengths_npole(self):
        """
        Test that scalars returned when we pass scalars.
        """
        test_cases = [
            (1.5, 2.5, 1),
            (2.0 - 0.2j, 5.0, 2),
        ]

        for m, x, n_pole in test_cases:
            a, b = mie.an_bn(m, x, n_pole)
            c, d = mie.cn_dn(m, x, n_pole)
            assert len(a) == n_pole, "Length wrong for a with n_pole=%d" % n_pole
            assert len(b) == n_pole, "Length wrong for b with n_pole=%d" % n_pole
            assert len(c) == n_pole, "Length wrong for c with n_pole=%d" % n_pole
            assert len(d) == n_pole, "Length wrong for d with n_pole=%d" % n_pole
        for m, x, n_pole in test_cases:
            a, b, c, d = mie.coefficients(m, x, n_pole, internal=True)
            assert len(a) == n_pole, "Length wrong for a with n_pole=%d" % n_pole
            assert len(b) == n_pole, "Length wrong for b with n_pole=%d" % n_pole
            assert len(c) == n_pole, "Length wrong for c with n_pole=%d" % n_pole
            assert len(d) == n_pole, "Length wrong for d with n_pole=%d" % n_pole

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

            assert len(a) == expected, "Length wrong for a with n_pole=%d" % n_pole
            assert len(b) == expected, "Length wrong for b with n_pole=%d" % n_pole
            assert len(c) == expected, "Length wrong for c with n_pole=%d" % n_pole
            assert len(d) == expected, "Length wrong for d with n_pole=%d" % n_pole


class TestBoundaryConditions:
    """Test cases for boundary conditions behavior."""

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
            assert max_error < tolerance, f"Boundary condition test failed with max error {max_error:.2e}"
