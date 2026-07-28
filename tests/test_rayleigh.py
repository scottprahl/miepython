"""Tests for the rayleigh scattering approximation routines."""

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
