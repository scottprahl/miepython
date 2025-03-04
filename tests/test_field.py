"""Test for electric field calculations."""

import pytest
import numpy as np
import miepython as mie
from miepython.field import e_near, e_far


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.1j])
@pytest.mark.parametrize("r", [1000, 10000])
@pytest.mark.parametrize("d_sphere", [0.1, 1])
def test_e_near_vs_e_far(m_sphere, r, d_sphere):
    """Confirm that e_near matches far field approximation for large r."""
    lambda0 = 1
    n_env = 1
    theta = np.radians(45)
    phi = np.radians(45)

    x = np.pi * d_sphere / lambda0

    # Mie coefficients
    a, b, c, d = mie.coefficients(m_sphere / n_env, x, internal=True)
    abcd = np.array([a, b, c, d])

    # Compute near and far fields
    Er, Etheta, Ephi = e_near(abcd, lambda0, d_sphere, n_env, r, theta, phi)
    Fr, Ftheta, Fphi = e_far(lambda0, d_sphere, m_sphere, r, theta, phi)[:, 0]
    #    print(abs(Etheta), abs(Ftheta))

    assert np.isclose(abs(Etheta), abs(Ftheta), rtol=1e-3), f"θ {Etheta}, far = {Ftheta}"
    assert np.isclose(abs(Ephi), abs(Fphi), rtol=1e-3), f"ɸ {Ephi}, far= {Fphi}"
