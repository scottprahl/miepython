"""Test for electric field calculations."""

import pytest
import numpy as np
import miepython as mie
from miepython.field import (
    e_near,
    e_far,
    h_near,
    eh_near,
    e_near_cartesian,
    h_near_cartesian,
    eh_near_cartesian,
)
from miepython.util import spherical_vector_to_cartesian


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.1j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
@pytest.mark.parametrize("r", [1000, 10000])
@pytest.mark.parametrize("d_sphere", [0.1, 1])
def test_e_near_vs_e_far(m_sphere, n_env, r, d_sphere):
    """Confirm that e_near matches far field approximation for large r."""
    lambda0 = 1
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


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
@pytest.mark.parametrize("theta", [np.radians(25), np.radians(70), np.radians(130)])
def test_h_boundary_tangential_continuity(m_sphere, n_env, theta):
    """Tangential components of total H should be continuous across the boundary."""
    lambda0 = 0.6328
    d_sphere = 1.2
    phi = np.radians(37)
    radius = d_sphere / 2
    delta = d_sphere * 1e-6

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    h_in = h_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius - delta,
        theta,
        phi,
        abcd=abcd,
    )
    h_out = h_near(
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

    np.testing.assert_allclose(h_in[1], h_out[1], rtol=3e-3, atol=2e-6)
    np.testing.assert_allclose(h_in[2], h_out[2], rtol=3e-3, atol=2e-6)


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
@pytest.mark.parametrize("theta", [np.radians(25), np.radians(70), np.radians(130)])
def test_h_boundary_normal_flux_continuity(m_sphere, n_env, theta):
    """Normal component of B should be continuous; here mu_r=1 so B_r is H_r."""
    lambda0 = 0.6328
    d_sphere = 1.2
    phi = np.radians(37)
    radius = d_sphere / 2
    delta = d_sphere * 1e-6

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    h_in = h_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        radius - delta,
        theta,
        phi,
        abcd=abcd,
    )
    h_out = h_near(
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

    np.testing.assert_allclose(h_in[0], h_out[0], rtol=3e-3, atol=2e-6)


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
def test_eh_near_matches_individual_calls(m_sphere, n_env):
    """eh_near should be equivalent to separate e_near and h_near calls."""
    lambda0 = 0.6328
    d_sphere = 1.2
    theta = np.radians(70)
    phi = np.radians(37)
    r = 0.8 * d_sphere

    E, H = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi)
    E_ref = e_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi)
    H_ref = h_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi)

    np.testing.assert_allclose(E, E_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(H, H_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
def test_eh_near_vectorized_matches_scalar_loop(m_sphere, n_env):
    """Vectorized spherical input should match scalar evaluations."""
    lambda0 = 0.6328
    d_sphere = 1.2
    r = d_sphere * np.array([[0.25, 0.55, 0.85], [0.35, 0.75, 1.05]])
    theta = np.array([np.radians(20), np.radians(80), np.radians(140)])
    phi = np.array([[np.radians(15)], [np.radians(75)]])

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    e_vec, h_vec = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, abcd=abcd)
    rr, tt, pp = np.broadcast_arrays(r, theta, phi)

    e_ref = np.empty((3,) + rr.shape, dtype=complex)
    h_ref = np.empty((3,) + rr.shape, dtype=complex)
    for idx in np.ndindex(rr.shape):
        e_ref[(slice(None),) + idx], h_ref[(slice(None),) + idx] = eh_near(
            lambda0,
            d_sphere,
            m_sphere,
            n_env,
            float(rr[idx]),
            float(tt[idx]),
            float(pp[idx]),
            abcd=abcd,
        )

    np.testing.assert_allclose(e_vec, e_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_vec, h_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("m_sphere", [1.5, 1.5 - 0.05j])
@pytest.mark.parametrize("n_env", [1.0, 1.33])
def test_cartesian_wrappers_match_spherical_transform(m_sphere, n_env):
    """Cartesian wrappers should match explicit spherical->Cartesian conversion."""
    lambda0 = 0.6328
    d_sphere = 1.2
    x = np.array([0.2, -0.35, 0.5])
    y = np.array([0.1, 0.4, -0.25])
    z = np.array([0.6, 0.2, -0.45])

    rr = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / rr, -1.0, 1.0))
    phi = np.arctan2(y, x)

    e_sph, h_sph = eh_near(lambda0, d_sphere, m_sphere, n_env, rr, theta, phi)
    ex_ref, ey_ref, ez_ref = spherical_vector_to_cartesian(e_sph[0], e_sph[1], e_sph[2], rr, theta, phi)
    hx_ref, hy_ref, hz_ref = spherical_vector_to_cartesian(h_sph[0], h_sph[1], h_sph[2], rr, theta, phi)
    e_ref = np.array([ex_ref, ey_ref, ez_ref])
    h_ref = np.array([hx_ref, hy_ref, hz_ref])

    e_xyz = e_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z)
    h_xyz = h_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z)
    e_xyz2, h_xyz2 = eh_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z)

    np.testing.assert_allclose(e_xyz, e_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_xyz, h_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_xyz2, e_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_xyz2, h_ref, rtol=1e-12, atol=1e-12)
