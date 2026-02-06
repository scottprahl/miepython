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

    # Compute near and far fields
    Er, Etheta, Ephi = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        r,
        theta,
        phi,
        include_incident=False,
    )
    Fr, Ftheta, Fphi = e_far(lambda0, d_sphere, m_sphere, n_env, r, theta, phi)[:, 0]
    #    print(abs(Etheta), abs(Ftheta))

    assert np.isclose(abs(Etheta), abs(Ftheta), rtol=1e-3), f"θ {Etheta}, far = {Ftheta}"
    assert np.isclose(abs(Ephi), abs(Fphi), rtol=1e-3), f"ɸ {Ephi}, far= {Fphi}"


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
@pytest.mark.parametrize("theta", [np.radians(25), np.radians(70), np.radians(130)])
def test_e_boundary_tangential_continuity(m_sphere, n_env, theta):
    """Tangential components of total E should be continuous across the boundary."""
    lambda0 = 0.6328
    d_sphere = 1.2
    phi = np.radians(37)
    radius = d_sphere / 2
    delta = d_sphere * 1e-6

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    e_in = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius - delta,
        theta,
        phi,
        abcd=abcd,
    )
    e_out = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius + delta,
        theta,
        phi,
        include_incident=True,
        abcd=abcd,
    )

    np.testing.assert_allclose(e_in[1], e_out[1], rtol=2e-3, atol=2e-6)
    np.testing.assert_allclose(e_in[2], e_out[2], rtol=2e-3, atol=2e-6)


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
@pytest.mark.parametrize("theta", [np.radians(25), np.radians(70), np.radians(130)])
def test_e_boundary_normal_displacement_continuity(m_sphere, n_env, theta):
    """Normal component of D should be continuous across the boundary."""
    lambda0 = 0.6328
    d_sphere = 1.2
    phi = np.radians(37)
    radius = d_sphere / 2
    delta = d_sphere * 1e-6

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    e_in = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius - delta,
        theta,
        phi,
        abcd=abcd,
    )
    e_out = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius + delta,
        theta,
        phi,
        include_incident=True,
        abcd=abcd,
    )

    # Internal fields are evaluated with conjugated m to match miepython's
    # n-ik coefficient convention.
    eps_in = np.conjugate(m_sphere) ** 2
    eps_out = n_env**2

    np.testing.assert_allclose(eps_in * e_in[0], eps_out * e_out[0], rtol=3e-3, atol=2e-6)
