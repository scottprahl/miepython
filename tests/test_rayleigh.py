"""Tests for the rayleigh scattering approximation routines."""

import warnings

import numpy as np
import pytest
import miepython as mie
from miepython import rayleigh

# Set up test parameters
x = 0.01  # Small size parameter for Rayleigh approximation
mu = np.linspace(-1, 1, 5)  # Cosine of scattering angles

# Refractive indices to test
test_cases = [
    (1.5, "Non-absorbing"),
    (1.5 - 0.1j, "Weakly absorbing"),
    (1.5 - 1j, "Strongly absorbing"),
]

# Allowable tolerance
TOLERANCE = 1e-2  # 1% relative error


def relative_error(a, b):
    """Calculate relative error safely, avoiding division by zero."""
    return np.abs(a - b) / (np.abs(b) + 1e-12)  # Small value to prevent div by zero


@pytest.mark.parametrize("m, case", test_cases)
def test_efficiencies(m, case):
    """Compare efficiencies from Rayleigh approximation to Mie theory."""
    qext_ray, qsca_ray, qback_ray, g_ray = rayleigh.efficiencies_mx(m, x)
    qext_mie, qsca_mie, qback_mie, g_mie = mie.efficiencies_mx(m, x)

    assert relative_error(qext_ray, qext_mie) < TOLERANCE, f"Qext mismatch for {case}"
    assert relative_error(qsca_ray, qsca_mie) < TOLERANCE, f"Qsca mismatch for {case}"
    assert relative_error(qback_ray, qback_mie) < TOLERANCE, f"Qback mismatch for {case}"
    assert np.isclose(g_ray, g_mie, atol=2e-5), f"g mismatch for {case}"  # g should be 0 for Rayleigh


@pytest.mark.parametrize("index", [1.33, 1.5])
@pytest.mark.parametrize("absorption", [1e-7, 1e-6])
@pytest.mark.parametrize("size", [0.01, 0.05])
def test_weak_absorption_is_positive(index, absorption, size):
    """Extinction must include positive absorption under either index convention."""
    m = index - 1j * absorption
    negative = rayleigh.efficiencies_mx(m, size)
    positive = rayleigh.efficiencies_mx(np.conj(m), size)
    np.testing.assert_allclose(negative, positive, rtol=1e-13)
    qabs = negative[0] - negative[1]
    assert qabs > 0
    # Leading Rayleigh absorption: 4*x*Im((m**2-1)/(m**2+2)), with
    # a positive imaginary index. Finite-size corrections are O(x**2).
    polarizability = (np.conj(m) ** 2 - 1) / (np.conj(m) ** 2 + 2)
    np.testing.assert_allclose(qabs, 4 * size * polarizability.imag, rtol=0.003)


@pytest.mark.parametrize(
    "m, size, rtol",
    [
        (1.5 - 1e-6j, 0.01, 1e-5),  # the sign error made qext 21% low here
        (1.5 - 0.1j, 0.05, 1e-5),
        (1.5 - 1j, 0.1, 1e-4),  # |ratio|**2 in the x**4 term is 6e-4 high here
        (0.05 - 1.43j, 0.05, 2e-3),  # silver-like, near the Frohlich resonance
    ],
)
def test_extinction_matches_mie_for_absorbing_spheres(m, size, rtol):
    """The 1% tolerance of test_efficiencies hides errors this size, so compare tightly."""
    qext_ray = rayleigh.efficiencies_mx(m, size)[0]
    qext_mie = mie.efficiencies_mx(m, size)[0]
    assert qext_ray == pytest.approx(qext_mie, rel=rtol)


@pytest.mark.parametrize("m, case", test_cases)
def test_S1_S2(m, case):
    """Compare scattering amplitudes S1 and S2 to Mie theory."""
    S1_ray, S2_ray = rayleigh.S1_S2(m, x, mu)
    S1_mie, S2_mie = mie.S1_S2(m, x, mu)

    assert np.allclose(S1_ray, S1_mie, atol=TOLERANCE), f"S1 mismatch for {case}"
    assert np.allclose(S2_ray, S2_mie, atol=TOLERANCE), f"S2 mismatch for {case}"


@pytest.mark.parametrize("m, case", test_cases)
def test_i_par(m, case):
    """Compare parallel intensity to Mie theory."""
    i_par_ray = rayleigh.i_par(m, x, mu)
    i_par_mie = mie.i_par(m, x, mu)

    assert np.allclose(i_par_ray, i_par_mie, atol=TOLERANCE), f"i_par mismatch for {case}"


@pytest.mark.parametrize("m, case", test_cases)
def test_i_per(m, case):
    """Compare perpendicular intensity to Mie theory."""
    i_per_ray = rayleigh.i_per(m, x, mu)
    i_per_mie = mie.i_per(m, x, mu)

    assert np.allclose(i_per_ray, i_per_mie, atol=TOLERANCE), f"i_per mismatch for {case}"


@pytest.mark.parametrize("m, case", test_cases)
def test_i_unpolarized(m, case):
    """Compare unpolarized intensity to Mie theory."""
    i_unpol_ray = rayleigh.i_unpolarized(m, x, mu)
    i_unpol_mie = (mie.i_per(m, x, mu) + mie.i_par(m, x, mu)) / 2

    assert np.allclose(i_unpol_ray, i_unpol_mie, atol=TOLERANCE), f"i_unpolarized mismatch for {case}"


@pytest.mark.parametrize("m, case", test_cases)
def test_phase_matrix(m, case):
    """Compare phase matrix from Rayleigh approximation to Mie theory."""
    phase_ray = rayleigh.phase_matrix(m, x, mu)
    phase_mie = mie.phase_matrix(m, x, mu)

    assert np.allclose(phase_ray, phase_mie, atol=TOLERANCE), f"Phase matrix mismatch for {case}"


# ---------------------------------------------------------------------------
# The physical-units wrappers, every normalization, and the error paths
# ---------------------------------------------------------------------------

NORMS_ALL = [
    "albedo",
    "a",
    "one",
    "1",
    "unity",
    "4pi",
    "four_pi",
    "qext",
    "extinction_efficiency",
    "qsca",
    "scattering_efficiency",
    "bohren",
    "wiscombe",
]


@pytest.mark.parametrize("n_env", [1.0, 1.333])
def test_efficiencies_wraps_efficiencies_mx(n_env):
    """The diameter/wavelength form must reduce to the size-parameter form."""
    m_sphere, d, lambda0 = 1.5 - 0.01j, 0.01, 0.5
    got = rayleigh.efficiencies(m_sphere, d, lambda0, n_env)
    x_env = np.pi * d / (lambda0 / n_env)
    want = rayleigh.efficiencies_mx(m_sphere / n_env, x_env)
    np.testing.assert_allclose(got, want, rtol=1e-13)


@pytest.mark.parametrize("n_env", [1.0, 1.333])
def test_intensities_wraps_i_par_and_i_per(n_env):
    """The wrapper must return exactly i_par and i_per for the same sphere."""
    m_sphere, d, lambda0 = 1.5 - 0.01j, 0.01, 0.5
    ipar, iper = rayleigh.intensities(m_sphere, d, lambda0, mu, n_env=n_env)
    x_env = np.pi * d / (lambda0 / n_env)
    m_rel = m_sphere / n_env
    np.testing.assert_allclose(ipar, rayleigh.i_par(m_rel, x_env, mu), rtol=1e-13)
    np.testing.assert_allclose(iper, rayleigh.i_per(m_rel, x_env, mu), rtol=1e-13)


@pytest.mark.parametrize("norm", NORMS_ALL)
def test_every_normalization_is_accepted(norm):
    """Each documented spelling, including the aliases, must work."""
    values = rayleigh.i_unpolarized(1.5 - 0.01j, x, mu, norm=norm)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)


@pytest.mark.parametrize(
    "norm,expected",
    [("one", 1.0), ("4pi", 4 * np.pi), ("qsca", "qsca"), ("qext", "qext"), ("albedo", "albedo")],
)
def test_normalization_integrates_to_its_name(norm, expected):
    """The integral of the unpolarized intensity over 4pi must match the label."""
    m_sphere = 1.5 - 0.01j
    fine = np.linspace(-1, 1, 20001)
    qext, qsca, _, _ = rayleigh.efficiencies_mx(m_sphere, x)
    want = {"qsca": qsca, "qext": qext, "albedo": qsca / qext}.get(expected, expected)

    intensity = rayleigh.i_unpolarized(m_sphere, x, fine, norm=norm)
    total = 2 * np.pi * np.trapezoid(intensity, fine)
    # only exact to O(x**2): see test_normalization_self_consistency_is_second_order
    assert total == pytest.approx(want, rel=1e-4)


def test_normalization_self_consistency_is_second_order():
    """The normalization closes only to O(x**2), and that is worth pinning.

    ``efficiencies_mx`` truncates qsca at x**4 while the ``a1`` used by ``S1_S2``
    also carries an x**5 term, so the integral of the 'one' normalization misses
    unity by roughly 0.07 x**2.  Anyone extending one expansion without the other
    should see this move.
    """
    m_sphere = 1.5 - 0.01j
    fine = np.linspace(-1, 1, 40001)
    residuals = {}
    for size in (0.003, 0.01, 0.03):
        intensity = rayleigh.i_unpolarized(m_sphere, size, fine, norm="one")
        residuals[size] = 2 * np.pi * np.trapezoid(intensity, fine) - 1.0

    for size, residual in residuals.items():
        assert residual == pytest.approx(0.0706 * size**2, rel=0.05), size

    # a decade in x moves the residual by two, confirming the quadratic order
    assert residuals[0.03] / residuals[0.003] == pytest.approx(100.0, rel=0.05)


def test_unnormalized_choices_are_self_consistent():
    """The bohren choice divides by a half where wiscombe divides by one."""
    m_sphere = 1.5 - 0.01j
    wis = rayleigh.i_unpolarized(m_sphere, x, mu, norm="wiscombe")
    boh = rayleigh.i_unpolarized(m_sphere, x, mu, norm="bohren")
    np.testing.assert_allclose(boh, 4 * wis, rtol=1e-12)


def test_unknown_normalization_is_rejected():
    """A misspelt normalization must say what the choices are."""
    with pytest.raises(ValueError, match="normalization must be one of"):
        rayleigh.i_unpolarized(1.5, x, mu, norm="nonsense")


@pytest.mark.parametrize("norm", ["albedo", "one", "4pi", "qext"])
def test_index_matched_sphere_is_rejected(norm):
    """A sphere that does not scatter has nothing to normalize against."""
    with pytest.raises(ValueError, match="scatters"):
        rayleigh.i_unpolarized(1.0, x, mu, norm=norm)


@pytest.mark.parametrize("norm", ["wiscombe", "bohren", "qsca"])
def test_index_matched_sphere_allowed_without_normalizing(norm):
    """The choices that do not divide by an efficiency still work."""
    values = rayleigh.i_unpolarized(1.0, x, mu, norm=norm)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values, 0.0, atol=1e-30)


def test_phase_matrix_matches_intensities():
    """Element (0,0) of the Mueller matrix is the unpolarized intensity."""
    m_sphere = 1.5 - 0.01j
    phase = rayleigh.phase_matrix(m_sphere, x, mu)
    np.testing.assert_allclose(phase[0, 0], rayleigh.i_unpolarized(m_sphere, x, mu), rtol=1e-12)


def test_phase_matrix_shape_for_scalar_mu():
    """A scalar angle gives a bare 4x4 matrix."""
    assert rayleigh.phase_matrix(1.5, x, 0.5).shape == (4, 4)
    assert rayleigh.phase_matrix(1.5, x, mu).shape == (4, 4, mu.size)


@pytest.mark.parametrize("norm", ["wiscombe", "albedo"])
def test_sign_of_the_imaginary_index_does_not_matter(norm):
    """The convention is m = n - ik, so a positive imaginary part is conjugated."""
    physical = 1.5 - 0.1j
    flipped = 1.5 + 0.1j
    np.testing.assert_allclose(
        rayleigh.S1_S2(flipped, x, mu, norm=norm),
        rayleigh.S1_S2(physical, x, mu, norm=norm),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        rayleigh.i_unpolarized(flipped, x, mu, norm=norm),
        rayleigh.i_unpolarized(physical, x, mu, norm=norm),
        rtol=1e-13,
    )


class TestArrayIndex:
    """Efficiencies handle an array of spheres entry by entry; angular functions take one."""

    M = np.array([1.5 - 0.1j, 1.5 + 0.1j, 1.33 + 1e-6j])

    def test_efficiencies_conjugate_each_entry(self):
        """Either sign of the imaginary part gives the same efficiencies."""
        qext, qsca, qback, _ = rayleigh.efficiencies_mx(self.M, 0.05)
        for i, m in enumerate(self.M):
            expected = rayleigh.efficiencies_mx(np.conj(m) if m.imag > 0 else m, 0.05)
            np.testing.assert_allclose([qext[i], qsca[i], qback[i]], expected[:3], rtol=1e-14)

    @pytest.mark.parametrize("module", [rayleigh, mie])
    @pytest.mark.parametrize("m, size", [(M, 0.05), (1.5, np.array([0.01, 0.05]))])
    def test_amplitudes_take_one_sphere(self, module, m, size):
        """S1_S2 used to fail with 'truth value of an array is ambiguous'; now it says why."""
        with pytest.raises(ValueError, match="takes one sphere"):
            module.S1_S2(m, size, 0.5)
        with pytest.raises(ValueError, match="takes one sphere"):
            module.i_unpolarized(m, size, 0.5)

    def test_g_has_the_shape_of_the_other_efficiencies(self):
        """The asymmetry parameter used to be the int 0 even when the others were arrays."""
        qext, qsca, qback, g = rayleigh.efficiencies_mx(self.M, 0.05)
        assert np.shape(g) == np.shape(qext) == np.shape(qsca) == np.shape(qback) == (3,)
        np.testing.assert_array_equal(g, 0)
        _, _, _, g = rayleigh.efficiencies_mx(1.5, np.array([0.01, 0.05]))
        assert np.shape(g) == (2,)

    def test_scalar_g_is_a_float(self):
        """A single sphere still gets a scalar g, as mie.efficiencies_mx returns."""
        g = rayleigh.efficiencies_mx(1.5 - 0.1j, 0.05)[3]
        assert np.ndim(g) == 0 and g == 0.0


class TestInvalidInput:
    """Reject the same inputs as the full Mie functions, rather than returning plausible numbers."""

    NAN = float("nan")
    ENTRY_POINTS = {
        "efficiencies_mx": rayleigh.efficiencies_mx,
        "efficiencies": lambda m, x: rayleigh.efficiencies(m, x * 0.5 / np.pi, 0.5),
        "S1_S2": lambda m, x: rayleigh.S1_S2(m, x, 0.5),
        "i_unpolarized": lambda m, x: rayleigh.i_unpolarized(m, x, 0.5),
        "intensities": lambda m, x: rayleigh.intensities(m, x * 0.5 / np.pi, 0.5, 0.5),
        "phase_matrix": lambda m, x: rayleigh.phase_matrix(m, x, 0.5),
    }

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    @pytest.mark.parametrize("m", [NAN, complex(1.5, NAN), complex(np.inf, 0)])
    def test_non_finite_index_is_an_error(self, name, m):
        """A NaN index used to return NaN efficiencies silently."""
        with pytest.raises(ValueError, match="refractive index must be finite"):
            self.ENTRY_POINTS[name](m, 0.05)

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_negative_real_index_is_an_error(self, name):
        """A negative real part used to give normal-looking efficiencies."""
        with pytest.raises(ValueError, match="must not be negative"):
            self.ENTRY_POINTS[name](-1.5 - 0.1j, 0.05)

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    @pytest.mark.parametrize("size", [NAN, np.inf])
    def test_non_finite_size_is_an_error(self, name, size):
        """A non-finite size parameter has no small-particle limit."""
        with pytest.raises(ValueError, match="size parameter must be finite"):
            self.ENTRY_POINTS[name](1.5, size)

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_negative_size_is_an_error(self, name):
        """A negative size parameter used to give normal-looking efficiencies."""
        with pytest.raises(ValueError, match="size parameter must not be negative"):
            self.ENTRY_POINTS[name](1.5, -0.05)

    def test_rayleigh_and_mie_agree_on_what_is_valid(self):
        """The two APIs accept the edge cases the conventions allow."""
        for m, size in [(1.5 + 0.1j, 0.05), (1.5, 0.0)]:
            ray = rayleigh.efficiencies_mx(m, size)
            ref = mie.efficiencies_mx(m, size)
            assert np.all(np.isfinite(ray)) and np.all(np.isfinite(ref))


class TestPerfectConductor:
    """m.real == 0 is a perfect conductor, with an electric and a magnetic dipole."""

    @pytest.mark.parametrize("m", [0, -1j, -5j, -1j * np.sqrt(2)])
    @pytest.mark.parametrize("size, rtol", [(0.01, 1e-4), (0.05, 1e-3)])
    def test_efficiencies_match_mie(self, m, size, rtol):
        """The electric dipole alone used to make qext three times too large."""
        qext, qsca, qback, g = rayleigh.efficiencies_mx(m, size)
        ref = mie.efficiencies_mx(m, size)
        np.testing.assert_allclose([qext, qsca, qback], ref[:3], rtol=rtol)
        assert g == pytest.approx(ref[3], abs=1e-3)

    def test_closed_forms(self):
        """No absorption, qsca = 10/3 x**4, qback = 9 x**4 and g = -2/5."""
        size = 0.05
        qext, qsca, qback, g = rayleigh.efficiencies_mx(-1j, size)
        assert qext == qsca == pytest.approx(10 / 3 * size**4, rel=1e-14)
        assert qback == pytest.approx(9 * size**4, rel=1e-14)
        assert g == -0.4

    @pytest.mark.parametrize("m", [0, -1j])
    def test_amplitudes_match_mie(self, m):
        """S1 and S2 carry the magnetic dipole, so the pattern is no longer symmetric."""
        size = 0.01
        s1, s2 = rayleigh.S1_S2(m, size, mu, norm="wiscombe")
        t1, t2 = mie.S1_S2(m, size, mu, norm="wiscombe")
        np.testing.assert_allclose(s1, t1, atol=1e-4 * size**3)
        np.testing.assert_allclose(s2, t2, atol=1e-4 * size**3)

    def test_backscattering_is_nine_times_forward(self):
        """The two dipoles add backwards and partly cancel forwards."""
        back, forward = rayleigh.i_unpolarized(-1j, 0.05, np.array([-1.0, 1.0]))
        assert back / forward == pytest.approx(9, rel=1e-14)

    def test_array_mixing_conductors_and_dielectrics(self):
        """Each entry uses its own expressions, with no warning from a vanishing denominator."""
        m = np.array([1.5 - 0.1j, -1j * np.sqrt(2), 0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rayleigh.efficiencies_mx(m, 0.05)
        for i, mm in enumerate(m):
            np.testing.assert_allclose([q[i] for q in result], rayleigh.efficiencies_mx(mm, 0.05), rtol=1e-14)
