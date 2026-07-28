"""Regression tests for the low-level Mie kernels.

The ``kernels`` fixture from conftest.py supplies one backend's kernels under
backend-free names and is parametrized over both, so every test here runs twice
in one process and this file replaces what used to be a
``test_jit*``/``test_nojit*`` pair.
"""

import numpy as np
import pytest
from scipy.special import lpmv, spherical_jn

from reference import basic_abcd, bohren_ab, pyscatt_ab, pyscatt_cd, slow_cn_dn

from miepython.util import cs


class TestElectricMagneticMultipoles:
    """Test the e_field selector between electric (a_n) and magnetic (b_n) multipoles."""

    def test_single_multipole_asymmetry_matches_angular_average(self, kernels):
        """The zero g agrees with <cos(theta)> integrated over the isolated pattern."""
        m = 1.5 - 0.01j
        x = 3.0
        a, b = kernels.an_bn(complex(m.real, -abs(m.imag)), x, 0)
        n_terms = len(a)
        mu = np.linspace(-1, 1, 2001)

        for n in range(1, 5):
            scale = (2 * n + 1) / (n * (n + 1))
            i_elec = np.zeros_like(mu)
            i_magn = np.zeros_like(mu)
            for k, mu_k in enumerate(mu):
                pi = np.zeros(n_terms)
                tau = np.zeros(n_terms)
                kernels.pi_tau(mu_k, pi, tau)
                i_elec[k] = abs(scale * a[n - 1] * pi[n - 1]) ** 2 + abs(scale * a[n - 1] * tau[n - 1]) ** 2
                i_magn[k] = abs(scale * b[n - 1] * tau[n - 1]) ** 2 + abs(scale * b[n - 1] * pi[n - 1]) ** 2

            assert np.sum(i_elec * mu) / np.sum(i_elec) == pytest.approx(0.0, abs=1e-12)
            assert np.sum(i_magn * mu) / np.sum(i_magn) == pytest.approx(0.0, abs=1e-12)

    def test_lossless_unitarity(self, kernels):
        """For a lossless sphere Re(a_n) must equal |a_n|**2 for every order."""
        for m in (1.05, 1.2, 1.33, 1.5, 2.0, 3.0, 0.75):
            for x in (0.1, 1.0, 5.0, 20.0, 62.0, 200.0):
                a, b = kernels.an_bn(complex(m, 0.0), x, 0)
                for coeff in (a, b):
                    nz = np.abs(coeff) > 0
                    resid = np.abs(coeff[nz].real - np.abs(coeff[nz]) ** 2)
                    assert np.max(resid / np.abs(coeff[nz])) < 1e-15, f"m={m} x={x}"


class TestConductingInternalCoefficients:
    """Test that cn_dn returns no internal field for a perfectly conducting sphere."""

    def test_conducting_sphere_has_no_internal_field(self, kernels):
        """Internal coefficients are zero when the index signals a perfect conductor."""
        for m in (complex(0.0, 0.0), complex(0.0, 100.0), complex(0.0, -100.0)):
            c, d = kernels.cn_dn(m, 1.0, 0)
            np.testing.assert_array_equal(c, np.zeros_like(c))
            np.testing.assert_array_equal(d, np.zeros_like(d))

    def test_infinite_index_has_no_internal_field(self, kernels):
        """An infinite index is guarded instead of producing nan."""
        for m in (complex(np.inf, 0.0), complex(1.5, -np.inf), complex(np.inf, -np.inf)):
            c, d = kernels.cn_dn(m, 1.0, 0)
            assert np.all(np.isfinite(c.view(float)))
            assert np.all(np.isfinite(d.view(float)))
            np.testing.assert_array_equal(c, np.zeros_like(c))
            np.testing.assert_array_equal(d, np.zeros_like(d))

    def test_dielectric_sphere_still_computed(self, kernels):
        """The guard does not suppress ordinary spheres."""
        for m in (complex(1.5, 0.0), complex(1.5, -0.1), complex(1.33, -1e-8)):
            c, d = kernels.cn_dn(m, 1.0, 0)
            assert np.any(c != 0)
            assert np.any(d != 0)

    def test_internal_coefficients_match_reference(self, kernels):
        """cn_dn agrees with a direct scipy evaluation, including m < 1."""
        for m, x in [
            (complex(1.5, -0.01), 5.0),
            (complex(1.5, -0.01), 20.0),
            (complex(1.5, -0.5), 10.0),
            (complex(0.75, 0.0), 1.0),
            (complex(0.75, 0.0), 20.0),
            (complex(0.75, -0.01), 20.0),
            (complex(1.05, 0.0), 15.0),
            (complex(1.33, 0.0), 0.1),
        ]:
            c, d = kernels.cn_dn(m, x, 0)
            c_ref, d_ref = slow_cn_dn(m, x, 0)
            k = min(len(c), len(c_ref))
            keep = np.abs(c_ref[:k]) > 1e-200
            np.testing.assert_allclose(c[:k][keep], c_ref[:k][keep], rtol=1e-8)
            keep = np.abs(d_ref[:k]) > 1e-200
            np.testing.assert_allclose(d[:k][keep], d_ref[:k][keep], rtol=1e-8)

    def test_internal_coefficients_stable_below_unit_index(self, kernels):
        """The psi ratio stays accurate when the order runs past |mx|."""
        # m < 1 puts every order above |mx|, which the old three-term
        # recurrence for psi_n(mx) could not survive
        m = complex(0.75, 0.0)
        x = 20.0
        c, _ = kernels.cn_dn(m, x, 0)
        c_ref, _ = slow_cn_dn(m, x, 0)
        k = min(len(c), len(c_ref))
        keep = np.abs(c_ref[:k]) > 1e-200
        rel = np.max(np.abs(c[:k][keep] - c_ref[:k][keep]) / np.abs(c_ref[:k][keep]))
        assert rel < 1e-10

    def test_internal_coefficients_at_sin_zeros(self, kernels):
        """Accuracy survives the orders where sin(x) or sin(mx) vanishes."""
        # x = n*pi happens whenever the diameter is a multiple of the wavelength,
        # and m*x = n*pi is the matching hazard on the internal argument
        cases = [
            (complex(1.5, 0.0), np.pi),
            (complex(1.5, 0.0), 2 * np.pi),
            (complex(2.5, 0.0), 2 * np.pi),  # m*x = 5*pi
            (complex(2.0, 0.0), 1.5 * np.pi),  # m*x = 3*pi
            (complex(0.5, 0.0), 4 * np.pi),  # m*x = 2*pi and m < 1
        ]
        for m, x in cases:
            c, d = kernels.cn_dn(m, x, 0)
            c_ref, d_ref = slow_cn_dn(m, x, 0)
            k = min(len(c), len(c_ref))
            keep = np.abs(c_ref[:k]) > 1e-200
            np.testing.assert_allclose(c[:k][keep], c_ref[:k][keep], rtol=1e-7)
            keep = np.abs(d_ref[:k]) > 1e-200
            np.testing.assert_allclose(d[:k][keep], d_ref[:k][keep], rtol=1e-7)

    def test_psi_downwards_matches_scipy(self, kernels):
        """The Riccati-Bessel helper agrees with scipy for real and complex z."""
        for z in (0.1 + 0j, np.pi + 0j, 20.0 + 0j, 15.0 - 3.0j, 2.0 - 8.0j, 0.5 + 0j):
            nstop = 30
            got = kernels.psi_downwards(np.complex128(z), nstop)
            n = np.arange(0, nstop + 1)
            ref = z * spherical_jn(n, z)

            # orders 1.. are tested relatively, right down into the underflowing
            # tail, because the downward recurrence carries relative accuracy
            keep = np.abs(ref[1:]) > 1e-250
            np.testing.assert_allclose(got[1:][keep], ref[1:][keep], rtol=1e-9)

            # psi_0 = sin(z) is a cancellation artifact at the zeros of sine, where
            # scipy and sin(z) disagree at the 1e-16 level, so it gets an absolute
            # bound instead
            assert abs(got[0] - ref[0]) <= 1e-13 * np.max(np.abs(ref))


class TestSeriesTruncation:
    """Test that every returned coefficient is a real term, with no padding."""

    def test_no_trailing_zero(self, kernels):
        """The last a_n and b_n are genuine coefficients, not padding."""
        for m, x in ((complex(1.5, -0.1), 5.0), (complex(1.33, 0.0), 1.0), (complex(2.0, -0.5), 20.0)):
            a, b = kernels.an_bn(m, x, 0)
            assert a[-1] != 0
            assert b[-1] != 0

    def test_internal_and_external_series_same_length(self, kernels):
        """a_n/b_n and c_n/d_n must be truncated at the same order."""
        for m, x in ((complex(1.5, -0.1), 5.0), (complex(1.33, 0.0), 1.0), (complex(2.0, -0.5), 20.0)):
            a, b = kernels.an_bn(m, x, 0)
            c, d = kernels.cn_dn(m, x, 0)
            assert len(a) == len(b) == len(c) == len(d)
            assert len(a) == int(x + 4.05 * x**0.33333 + 2.0)

    def test_pi_tau_fills_every_order(self, kernels):
        """pi_tau must set tau for the highest order too."""
        mu = 0.3
        for n_terms in (1, 2, 5, 12):
            pi = np.zeros(n_terms)
            tau = np.zeros(n_terms)
            kernels.pi_tau(mu, pi, tau)
            assert tau[-1] != 0
            # closed forms: tau_1 = mu, tau_2 = 3(2mu^2 - 1), pi_1 = 1, pi_2 = 3mu
            assert pi[0] == pytest.approx(1.0)
            assert tau[0] == pytest.approx(mu)
            if n_terms >= 2:
                assert pi[1] == pytest.approx(3 * mu)
                assert tau[1] == pytest.approx(3 * (2 * mu**2 - 1))

    def test_pi_tau_matches_legendre(self, kernels):
        """pi_n and tau_n agree with the Legendre function and its derivative."""
        n_terms = 10
        n = np.arange(1, n_terms + 1)
        for theta in (0.3, 1.1, np.pi / 2, 2.2, 2.9):
            mu = np.cos(theta)
            pi = np.zeros(n_terms)
            tau = np.zeros(n_terms)
            kernels.pi_tau(mu, pi, tau)

            # pi_n = P_n^1(cos theta) / sin theta
            # atol matters at theta=pi/2 where some pi_n vanish
            np.testing.assert_allclose(pi, -lpmv(1, n, mu) / np.sin(theta), rtol=1e-10, atol=1e-12)

            # tau_n = d/dtheta P_n^1(cos theta), by central difference
            h = 1e-6
            plus = -lpmv(1, n, np.cos(theta + h))
            minus = -lpmv(1, n, np.cos(theta - h))
            np.testing.assert_allclose(tau, (plus - minus) / (2 * h), rtol=1e-6, atol=1e-6)


class Test_NonAbsorbing_An_and_Bn:
    """Test cases for non absorbing an and bn behavior."""

    def test_01an_bn(self, kernels):
        """Test 01an bn."""
        m = 1.5
        x = 1
        a, b = kernels.an_bn(m, x, 0)
        ab, bb, _ = bohren_ab(m, x)
        ap, bp, _ = pyscatt_ab(m, x)
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
        for _ in range(3):
            assert a[0].real == pytest.approx(at[0].real, abs=0.00000001)
            assert a[0].imag == pytest.approx(at[0].imag, abs=0.00000001)
            assert b[0].real == pytest.approx(bt[0].real, abs=0.00000001)
            assert b[0].imag == pytest.approx(bt[0].imag, abs=0.00000001)

    def test_02an_bn(self, kernels):
        """Test 02an bn."""
        m = 1.5
        x = 0.01
        a, b = kernels.an_bn(m, x, 0)
        ab, bb, _ = bohren_ab(m, x)
        ap, bp, _ = pyscatt_ab(m, x)
        at, bt, _, _ = basic_abcd(m, x, len(a))

        print("a m=%s, x=%.4f" % (cs(m), x))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))
        for _ in range(3):
            assert a[0].real == pytest.approx(at[0].real, abs=0.00000001)
            assert a[0].imag == pytest.approx(at[0].imag, abs=0.00000001)
            assert b[0].real == pytest.approx(bt[0].real, abs=0.00000001)
            assert b[0].imag == pytest.approx(bt[0].imag, abs=0.00000001)

    def test_03an_bn(self, kernels):
        """Test 03an bn."""
        m = 1.5
        x = 10
        a, b = kernels.an_bn(m, x, 0)
        ab, bb, _ = bohren_ab(m, x)
        ap, bp, _ = pyscatt_ab(m, x)

        print("a m=%s, x=%.4f" % (cs(m), x))
        print(cs(a))
        print(cs(ab))
        print(cs(ap))
        print("b")
        print(cs(b))
        print(cs(bb))
        print(cs(bp))

        for _ in range(3):
            assert a[0].real == pytest.approx(ab[0].real, abs=0.00000001)
            assert a[0].imag == pytest.approx(ab[0].imag, abs=0.00000001)
            assert b[0].real == pytest.approx(bb[0].real, abs=0.00000001)
            assert b[0].imag == pytest.approx(bb[0].imag, abs=0.00000001)

    def test_05an_bn(self, kernels):
        """Test 05an bn."""
        m = 4.0 / 3.0
        x = 50
        a, b = kernels.an_bn(m, x, 0)
        ab, bb, _ = bohren_ab(m, x)
        ap, bp, _ = pyscatt_ab(m, x)
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
        assert a[0].real == pytest.approx(0.5311058892948411929, abs=0.00000001)
        assert a[0].imag == pytest.approx(-0.4990314856310943073, abs=0.00000001)


class Test_Strong_Absorbing_An_and_Bn:
    """Test cases for strong absorbing an and bn behavior."""

    def test_07an_bn(self, kernels):
        """Test 07an bn."""
        m = 1.1 - 25j
        x = 2
        a, b = kernels.an_bn(m, x, 0)
        assert a[1].real == pytest.approx(0.324433578437, abs=0.0001)
        assert a[1].imag == pytest.approx(-0.465627763266, abs=0.0001)
        assert b[1].real == pytest.approx(0.060464399088, abs=0.0001)
        assert b[1].imag == pytest.approx(0.236805417045, abs=0.0001)


class TestCnDn:
    """Test cn and dn against the results from pyscatt routine.."""

    def test_00_cn_dn(self, kernels):
        """Test 00 cn dn."""
        m = 1.5 - 1j
        x = 2.5
        c, d = kernels.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        _, _, ct, dt = basic_abcd(m, x)
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
            assert c[i] == pytest.approx(cx[i], abs=1e-6)
            assert d[i] == pytest.approx(dx[i], abs=1e-6)

    def test_02_cn_dn(self, kernels):
        """Test 02 cn dn."""
        m = 1.5 - 0.1j
        x = 1
        c, d = kernels.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        _, _, ct, dt = basic_abcd(m, x)
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
            assert c[i] == pytest.approx(cx[i], abs=1e-6)
            assert d[i] == pytest.approx(dx[i], abs=1e-6)

    def test_03_cn_dn(self, kernels):
        """Test 03 cn dn."""
        m = 1.5 - 1j
        x = 1
        c, d = kernels.cn_dn(m, x, 0)
        cx, dx = slow_cn_dn(m, x)
        _, _, ct, dt = basic_abcd(m, x)
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
            assert c[i] == pytest.approx(cx[i], abs=1e-6)
            assert d[i] == pytest.approx(dx[i], abs=1e-6)


class TestKernelEdges:
    """Test the kernel paths the ordinary Mie calls never reach."""

    def test_d_upwards_is_still_correct_where_it_is_valid(self, kernels):
        """``_D_upwards`` is retained for comparison, so it must still work.

        ``D_calc`` no longer selects it, because it loses accuracy once the order
        passes |z|.  Below that it is fine, and this pins that so the function is
        not left to rot.
        """
        z = np.complex128(40.0)
        n_max = 20  # comfortably under |z|
        upward = np.zeros(n_max + 1, dtype=np.complex128)
        kernels.D_upwards(z, n_max, upward)

        n = np.arange(1, n_max)
        exact = spherical_jn(n - 1, z) / spherical_jn(n, z) - n / z
        np.testing.assert_allclose(upward[1:n_max], exact, rtol=1e-11)

    def test_d_upwards_and_d_downwards_agree_below_the_argument(self, kernels):
        """The two recurrences must give the same answer where both are valid."""
        z = np.complex128(40.0)
        n_max = 20
        upward = np.zeros(n_max + 1, dtype=np.complex128)
        downward = np.zeros(n_max + 1, dtype=np.complex128)
        kernels.D_upwards(z, n_max, upward)
        kernels.D_downwards(z, n_max, downward)
        np.testing.assert_allclose(upward[1:n_max], downward[1:n_max], rtol=1e-10)

    def test_lentz_seeds_the_downward_recurrence(self, kernels):
        """Lentz's continued fraction must match the ratio it stands in for."""
        for z in (np.complex128(12.0), np.complex128(8.0 - 3.0j)):
            for order in (5, 15):
                got = kernels.Lentz_Dn(z, order)
                want = spherical_jn(order - 1, z) / spherical_jn(order, z) - order / z
                assert got == pytest.approx(want, rel=1e-9), (z, order)

    @pytest.mark.parametrize("z,nstop", [(1e-3, 100), (1e-2, 60)])
    def test_psi_downwards_rescales_without_losing_accuracy(self, kernels, z, nstop):
        """A small argument makes the downward recurrence overflow unless rescaled.

        psi grows on the way down from its tiny seed; for these inputs it passes
        1e200 and the guard divides the tail back into range.  The normalisation
        afterwards has to undo that exactly.
        """
        psi = kernels.psi_downwards(np.complex128(z), nstop)
        assert np.all(np.isfinite(psi))

        orders = np.arange(nstop + 1)
        reference = z * spherical_jn(orders, z)
        live = np.abs(reference) > 1e-300
        rel = np.abs(psi[live] - reference[live]) / np.abs(reference[live])
        assert np.max(rel) < 1e-11

    def test_cn_dn_returns_zeros_for_a_sphere_of_no_size(self, kernels):
        """There is no interior to have a field in."""
        c, d = kernels.cn_dn(complex(1.5, -0.1), 0.0, 0)
        assert len(c) == len(d) > 0
        np.testing.assert_array_equal(c, np.zeros_like(c))
        np.testing.assert_array_equal(d, np.zeros_like(d))

    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0])
    def test_zero_index_means_perfectly_conducting(self, kernels, x):
        """m=0 is a shorthand the kernel turns into a very large imaginary index."""
        as_zero = kernels.single_sphere(complex(0.0, 0.0), x, 0, True)
        as_conductor = kernels.single_sphere(complex(1.0, -10000.0), x, 0, True)
        np.testing.assert_allclose(as_zero, as_conductor, rtol=1e-12)

        # 1-10000j is a very good conductor rather than a perfect one, so a little
        # absorption remains.  It is roughly constant in absolute terms, which is
        # why this is not phrased as a relative tolerance: at x=0.1 qext itself is
        # only 3e-4 and the same 6e-8 of absorption is 2e-4 of it.
        qext, qsca = as_zero[0], as_zero[1]
        assert 0 <= qext - qsca < 1e-6
