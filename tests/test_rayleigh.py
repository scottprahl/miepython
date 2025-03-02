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


if __name__ == "__main__":
    pytest.main()
