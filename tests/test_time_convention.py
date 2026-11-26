"""The time convention: its API, and that switching it only conjugates complex results.

For a non-magnetic sphere, exp(+jωt) and exp(-iωt) describe the same physics and
every complex result in one is the complex conjugate of the same result in the
other, while real results do not change at all.  The tests here hold every public
function to exactly that, so the reference comparisons elsewhere only need to run
in one convention each: Wiscombe's MIEV0 in ``test_mie.py`` in exp(+jωt), and
Bohren & Huffman and scattnlay (``test_abcd.py``, ``test_field.py``,
``test_vsh.py``, ``test_scattnlay_reference.py``) in exp(-iωt).
"""

import numpy as np
import pytest

import miepython as mie
from miepython import field, rayleigh, vsh

M_ABSORBING = 1.5 - 0.1j
X = 2.3
MU = np.cos(np.radians([0.0, 30.0, 75.0, 120.0, 180.0]))

# lambda0, d_sphere, m_sphere, n_env for the field functions
FIELD_ARGS = (0.62, 0.81, 1.62 - 0.05j, 1.18)
# points inside the sphere, near the surface outside, and far away
R = np.array([0.2, 0.5, 3.0, 777.3])
THETA = np.array([0.4, 1.1, 1.9, 2.8])
PHI = np.array([0.27, 2.5, 0.6, 4.0])


def both(func, *args, **kwargs):
    """Return ``func(*args, **kwargs)`` under exp(+jωt) and then exp(-iωt)."""
    with mie.time_convention("+"):
        plus = func(*args, **kwargs)
    with mie.time_convention("-"):
        minus = func(*args, **kwargs)
    return plus, minus


def assert_conjugates(plus, minus):
    """Compare nested results element by element as complex conjugates."""
    if isinstance(plus, tuple):
        assert len(plus) == len(minus)
        for p, m in zip(plus, minus):
            assert_conjugates(p, m)
        return
    plus = np.asarray(plus)
    minus = np.asarray(minus)
    np.testing.assert_allclose(plus, np.conjugate(minus), rtol=1e-13, atol=1e-300)
    # make sure the comparison is not trivially passing on real numbers
    assert np.any(np.abs(np.imag(plus)) > 1e-12)


class TestSetting:
    """The get/set/context-manager API."""

    def test_default_is_plus(self):
        """exp(+jωt) is the default."""
        assert mie.get_time_convention() == "+"

    def test_set_returns_previous(self):
        """The setter hands back what it replaced."""
        assert mie.set_time_convention("-") == "+"
        assert mie.get_time_convention() == "-"
        assert mie.set_time_convention("+") == "-"

    def test_context_manager_restores(self):
        """The block sees its own setting and the caller's comes back afterwards."""
        mie.set_time_convention("-")
        with mie.time_convention("+") as inside:
            assert inside == "+"
            assert mie.get_time_convention() == "+"
        assert mie.get_time_convention() == "-"

    def test_context_manager_restores_after_an_exception(self):
        """An error inside the block does not leave the setting changed."""
        with pytest.raises(RuntimeError):
            with mie.time_convention("-"):
                raise RuntimeError("boom")
        assert mie.get_time_convention() == "+"

    @pytest.mark.parametrize("bad", ["", "plus", "+-", "exp(-iwt)", None, 1, -1])
    def test_rejects_anything_else(self, bad):
        """Only the two strings are accepted, and a bad value changes nothing."""
        with pytest.raises(ValueError, match="time convention"):
            mie.set_time_convention(bad)
        with pytest.raises(ValueError, match="time convention"):
            with mie.time_convention(bad):
                pass
        assert mie.get_time_convention() == "+"


class TestComplexResultsAreConjugated:
    """Every complex public result flips to its conjugate, and nothing else changes."""

    @pytest.mark.parametrize("internal", [False, True])
    def test_coefficients(self, internal):
        """a, b (and c, d) are conjugated."""
        assert_conjugates(*both(mie.coefficients, M_ABSORBING, X, internal=internal))

    def test_coefficients_for_an_array_of_spheres(self):
        """The array path converts too."""
        m = np.array([1.5 - 0.1j, 1.33 - 0.01j])
        x = np.array([0.7, 3.1])
        assert_conjugates(*both(mie.coefficients, m, x, n_pole=3, internal=True))

    @pytest.mark.parametrize("m, x", [(M_ABSORBING, X), (1.5, 0.05), (-1j, 0.05), (-1j, 1.2)])
    @pytest.mark.parametrize("n_pole", [0, 1])
    def test_S1_S2(self, m, x, n_pole):
        """Including the small-sphere and conducting routes, and a single multipole."""
        assert_conjugates(*both(mie.S1_S2, m, x, MU, n_pole=n_pole))

    @pytest.mark.parametrize("m", [M_ABSORBING, 1.5, -1j])
    def test_rayleigh_S1_S2(self, m):
        """The small-particle amplitudes follow the same rule."""
        assert_conjugates(*both(rayleigh.S1_S2, m, 0.05, MU))

    @pytest.mark.parametrize("module", [mie, rayleigh], ids=["mie", "rayleigh"])
    def test_phase_matrix_only_S34_changes_sign(self, module):
        """S11, S12 and S33 are identical; S34 and -S34 swap."""
        x = X if module is mie else 0.05
        plus, minus = both(module.phase_matrix, M_ABSORBING, x, MU)
        sign = np.ones((4, 4))
        sign[2, 3] = sign[3, 2] = -1
        np.testing.assert_allclose(plus, sign[:, :, np.newaxis] * minus, rtol=1e-13, atol=1e-30)

    @pytest.mark.parametrize("name", ["e_near", "h_near", "e_near_cartesian", "h_near_cartesian"])
    @pytest.mark.parametrize("include_incident", [True, False])
    def test_near_fields(self, name, include_incident):
        """Inside, near and far from the sphere, with and without the incident wave."""
        func = getattr(field, name)
        if name.endswith("cartesian"):
            x, y, z = R * np.sin(THETA) * np.cos(PHI), R * np.sin(THETA) * np.sin(PHI), R * np.cos(THETA)
            args = (*FIELD_ARGS, x, y, z)
        else:
            args = (*FIELD_ARGS, R, THETA, PHI)
        assert_conjugates(*both(func, *args, include_incident=include_incident))

    @pytest.mark.parametrize("name", ["eh_near", "eh_near_cartesian"])
    def test_combined_near_fields(self, name):
        """Both halves of the (E, H) pair."""
        func = getattr(field, name)
        if name.endswith("cartesian"):
            x, y, z = R * np.sin(THETA) * np.cos(PHI), R * np.sin(THETA) * np.sin(PHI), R * np.cos(THETA)
            args = (*FIELD_ARGS, x, y, z)
        else:
            args = (*FIELD_ARGS, R, THETA, PHI)
        assert_conjugates(*both(func, *args))

    def test_far_field(self):
        """e_far, whose radial part was wrong in both senses before #36."""
        plus, minus = both(field.e_far, *FIELD_ARGS, R[-1], THETA, PHI)
        # E_r is identically zero, which assert_conjugates would reject as trivial
        assert_conjugates(plus[1:], minus[1:])
        assert np.all(plus[0] == 0) and np.all(minus[0] == 0)

    @pytest.mark.parametrize("name", ["M_odd", "M_even", "N_odd", "N_even"])
    def test_vsh_outside(self, name):
        """Outgoing harmonics switch between h_n^(2) and h_n^(1).

        Outside the sphere the index is real, so the conjugate is exactly the
        other Hankel function.  M has no radial part, so only the angular
        components are compared for it.
        """
        func = getattr(vsh, name)
        lambda0, d_sphere = 0.5, 1.0
        plus, minus = both(func, 2, lambda0, d_sphere, 1.0, 0.9, 1.1, 0.4)
        keep = slice(1, None) if name.startswith("M") else slice(None)
        assert_conjugates(plus[keep], minus[keep])

    @pytest.mark.parametrize("name", ["M_odd", "N_even"])
    def test_vsh_inside_takes_the_index_for_its_convention(self, name):
        """Inside, n - ik under exp(+jωt) is the conjugate of n + ik under exp(-iωt)."""
        func = getattr(vsh, name)
        lambda0, d_sphere, r = 0.5, 1.0, 0.3
        with mie.time_convention("+"):
            plus = func(2, lambda0, d_sphere, M_ABSORBING, r, 1.1, 0.4)
        with mie.time_convention("-"):
            minus = func(2, lambda0, d_sphere, np.conjugate(M_ABSORBING), r, 1.1, 0.4)
        keep = slice(1, None) if name.startswith("M") else slice(None)
        assert_conjugates(plus[keep], minus[keep])


class TestRealResultsDoNotChange:
    """Results that carry no phase are identical in both conventions."""

    @pytest.mark.parametrize("func", [mie.efficiencies_mx, rayleigh.efficiencies_mx])
    def test_efficiencies(self, func):
        """qext, qsca, qback and g."""
        plus, minus = both(func, M_ABSORBING, 0.3)
        np.testing.assert_array_equal(plus, minus)

    @pytest.mark.parametrize("name", ["i_par", "i_per", "i_unpolarized"])
    def test_intensities(self, name):
        """Each polarized and unpolarized intensity."""
        plus, minus = both(getattr(mie, name), M_ABSORBING, X, MU)
        np.testing.assert_array_equal(plus, minus)

    def test_physical_units(self):
        """The wavelength-and-diameter entry points."""
        plus, minus = both(mie.efficiencies, M_ABSORBING, 0.5, 0.6, 1.33)
        np.testing.assert_array_equal(plus, minus)
        plus, minus = both(mie.intensities, M_ABSORBING, 0.5, 0.6, MU, 1.33)
        np.testing.assert_array_equal(plus, minus)


class TestKernelsIgnoreTheSetting:
    """The raw kernels always work in exp(-iωt)."""

    def test_an_bn_and_S1_S2(self):
        """Switching the convention leaves the kernels untouched."""
        mu = np.array(MU)
        plus = both(mie.an_bn, M_ABSORBING, X, 0)
        np.testing.assert_array_equal(plus[0], plus[1])
        plus = both(mie._S1_S2, M_ABSORBING, X, mu, 0)  # pylint: disable=protected-access
        np.testing.assert_array_equal(plus[0], plus[1])

    def test_kernels_are_bohren_and_huffman(self):
        """A small sphere's a_1 is B&H's -i(2/3)x³(m²-1)/(m²+2), with m = n + ik."""
        x = 0.01
        a, _ = mie.an_bn(M_ABSORBING, x, 1)
        m_bh = np.conjugate(M_ABSORBING)
        expected = -1j * 2 * x**3 / 3 * (m_bh**2 - 1) / (m_bh**2 + 2)
        np.testing.assert_allclose(a[0], expected, rtol=1e-3)


class TestRelationsBetweenFunctions:
    """The public functions agree with each other in either convention."""

    @pytest.mark.parametrize("sign", ["+", "-"])
    def test_S1_S2_is_the_series_of_coefficients(self, sign):
        """S1 = sum (2n+1)/(n(n+1)) (a_n pi_n + b_n tau_n), Bohren & Huffman eq. 4.74."""
        with mie.time_convention(sign):
            a, b = mie.coefficients(M_ABSORBING, X)
            S1, S2 = mie.S1_S2(M_ABSORBING, X, MU, norm="wiscombe")
        n = np.arange(1, len(a) + 1)
        scale = (2 * n + 1) / (n * (n + 1))
        for k, mu in enumerate(MU):
            pi = np.zeros(len(a))
            tau = np.zeros(len(a))
            mie.pi_tau(mu, pi, tau)
            np.testing.assert_allclose(S1[k], np.sum(scale * (a * pi + b * tau)), rtol=1e-12)
            np.testing.assert_allclose(S2[k], np.sum(scale * (a * tau + b * pi)), rtol=1e-12)

    @pytest.mark.parametrize("sign", ["+", "-"])
    def test_user_abcd_round_trips(self, sign):
        """Fields from a caller's ``coefficients`` equal fields computed internally."""
        lambda0, d_sphere, m_sphere, n_env = FIELD_ARGS
        x = np.pi * d_sphere * n_env / lambda0
        n_pole = mie.core.wiscombe_terms(x) + 2
        with mie.time_convention(sign):
            abcd = np.array(mie.coefficients(m_sphere / n_env, x, n_pole=n_pole, internal=True))
            given = field.eh_near(*FIELD_ARGS, R, THETA, PHI, abcd=abcd)
            computed = field.eh_near(*FIELD_ARGS, R, THETA, PHI)
        np.testing.assert_allclose(given[0], computed[0], rtol=1e-14)
        np.testing.assert_allclose(given[1], computed[1], rtol=1e-14)

    @pytest.mark.parametrize("sign, phase", [("+", -1j), ("-", 1j)])
    def test_incident_wave_runs_the_right_way(self, sign, phase):
        """A quarter wave along +z multiplies the incident E_x by exp(∓jπ/2)."""
        lambda0, d_sphere, m_sphere, n_env = FIELD_ARGS
        quarter = lambda0 / n_env / 4
        z0 = 50.0
        with mie.time_convention(sign):
            e_total = field.e_near_cartesian(lambda0, d_sphere, m_sphere, n_env, 0.0, 0.0, [z0, z0 + quarter])
            e_scat = field.e_near_cartesian(
                lambda0, d_sphere, m_sphere, n_env, 0.0, 0.0, [z0, z0 + quarter], include_incident=False
            )
        ex = e_total[0] - e_scat[0]
        np.testing.assert_allclose(ex[1], phase * ex[0], atol=1e-12)

    @pytest.mark.parametrize("sign, phase", [("+", -1j), ("-", 1j)])
    def test_far_field_runs_outward(self, sign, phase):
        """A quarter wave further out multiplies r·E by exp(∓jπ/2), as in issue #36."""
        lambda0, d_sphere, m_sphere, n_env = 0.62, 0.81, 1.62, 1.18
        period = lambda0 / n_env
        r1, r2 = 256 * period, 256.25 * period
        with mie.time_convention(sign):
            e1 = field.e_far(lambda0, d_sphere, m_sphere, n_env, r1, 0.44, 0.27)
            e2 = field.e_far(lambda0, d_sphere, m_sphere, n_env, r2, 0.44, 0.27)
        np.testing.assert_allclose(r2 * e2, phase * r1 * e1, atol=1e-12)
