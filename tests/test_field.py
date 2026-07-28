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
    _coefficients_abcd,
)
from miepython.core import wiscombe_terms
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
    _Er, Etheta, Ephi = e_near(
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        r,
        theta,
        phi,
        include_incident=False,
    )
    _Fr, Ftheta, Fphi = e_far(lambda0, d_sphere, m_sphere, n_env, r, theta, phi)[:, 0]
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


def test_no_contrast_cartesian_fields_match_incident_plane_wave():
    """No-contrast sphere should reduce to the incident plane wave everywhere."""
    lambda0 = 1.0
    n_env = 1.0
    d_sphere = 0.002
    m_sphere = 1.0 + 0.0j

    x = np.array([3.0, 0.0, 0.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 3.0, 2.0, 2.0])
    z = np.array([0.0, 3.0, 0.0, 0.0, 0.5])

    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * z)
    zeros = np.zeros_like(phase)
    e_expected = np.array([phase, zeros, zeros])
    h_expected = np.array([zeros, phase, zeros])

    e_xyz = e_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z, include_incident=True)
    h_xyz = h_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z, include_incident=True)
    e_xyz2, h_xyz2 = eh_near_cartesian(lambda0, d_sphere, m_sphere, n_env, x, y, z, include_incident=True)

    np.testing.assert_allclose(e_xyz, e_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_xyz, h_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_xyz2, e_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_xyz2, h_expected, rtol=1e-12, atol=1e-12)


def test_no_contrast_spherical_fields_match_incident_components():
    """No-contrast spherical fields should follow standard spherical components."""
    lambda0 = 1.0
    n_env = 1.0
    d_sphere = 0.002
    m_sphere = 1.0 + 0.0j
    x, y, z = 2.0, 2.0, 0.5

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * z)

    e_expected = np.array(
        [
            phase * np.sin(theta) * np.cos(phi),
            phase * np.cos(theta) * np.cos(phi),
            -phase * np.sin(phi),
        ]
    )
    h_expected = np.array(
        [
            phase * np.sin(theta) * np.sin(phi),
            phase * np.cos(theta) * np.sin(phi),
            phase * np.cos(phi),
        ]
    )

    e_sph = e_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True)
    h_sph = h_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True)
    e_sph2, h_sph2 = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True)

    np.testing.assert_allclose(e_sph, e_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_sph, h_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_sph2, e_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_sph2, h_expected, rtol=1e-12, atol=1e-12)


EXTRA_FIELD_ORDERS = 2


def test_default_truncation_keeps_extra_orders():
    """The near field keeps more terms than Wiscombe's scattering criterion.

    The criterion is tuned for the scattered series; the field at the surface
    converges more slowly, and past two extra orders it stops improving.
    """
    lambda0, d_sphere, n_env = 1.0, 1.0, 1.0
    x = np.pi * d_sphere * n_env / lambda0
    abcd = _coefficients_abcd(lambda0, d_sphere, 1.5 + 0j, n_env, 0)
    for coeff in abcd:
        assert len(coeff) == wiscombe_terms(x) + EXTRA_FIELD_ORDERS


@pytest.mark.parametrize("m_sphere", [1.5 + 0j, 0.75 + 0j, 1.33 + 0j, 1.5 - 0.1j])
def test_default_path_boundary_continuity(m_sphere):
    """Tangential E is continuous to better than 1e-5 on the default abcd path.

    Truncating at Wiscombe's criterion instead leaves a mismatch near 1.5e-5, so
    this also guards the extra orders from being dropped again.  The exact count
    is pinned by test_default_truncation_keeps_extra_orders; this checks the
    outcome the count is chosen for.
    """
    lambda0, d_sphere, n_env = 1.0, 1.0, 1.0
    radius = d_sphere / 2
    delta = 1e-7
    theta = np.linspace(0.05, np.pi - 0.05, 25)
    phi = np.linspace(0.0, 2 * np.pi, 25)

    e_in = e_near(lambda0, d_sphere, m_sphere, n_env, np.full_like(theta, radius - delta), theta, phi)
    e_out = e_near(lambda0, d_sphere, m_sphere, n_env, np.full_like(theta, radius + delta), theta, phi)

    for comp in (1, 2):  # theta and phi are the tangential components
        scale = np.maximum(np.abs(e_out[comp]), 1e-12)
        assert np.max(np.abs(e_in[comp] - e_out[comp]) / scale) < 5e-6


# ---------------------------------------------------------------------------
# The scattered-only fields and an explicitly requested term count
# ---------------------------------------------------------------------------

FIELD_LAMBDA0 = 1.0
FIELD_D = 1.0
FIELD_N_ENV = 1.0
FIELD_M = 1.5 - 0.05j


def _angles():
    """A ring of points that avoids the poles and the azimuth seam."""
    theta = np.linspace(0.2, np.pi - 0.2, 7)
    phi = np.linspace(0.1, 2 * np.pi - 0.1, 7)
    return theta, phi


def _incident_plane_wave(theta, phi, r):
    """The incident wave written out independently of miepython."""
    k = 2 * np.pi * FIELD_N_ENV / FIELD_LAMBDA0
    amp = np.exp(1j * k * r * np.cos(theta))
    e_inc = np.array([amp * np.sin(theta) * np.cos(phi), amp * np.cos(theta) * np.cos(phi), -amp * np.sin(phi)])
    h_inc = np.array([amp * np.sin(theta) * np.sin(phi), amp * np.cos(theta) * np.sin(phi), amp * np.cos(phi)])
    return e_inc, h_inc


@pytest.mark.parametrize("which", ["E", "H"])
def test_total_minus_scattered_is_the_incident_wave(which):
    """Outside the sphere the total field is the scattered plus incident field."""
    theta, phi = _angles()
    r = np.full_like(theta, 1.4)  # comfortably outside
    fn = e_near if which == "E" else h_near
    total = fn(FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi, include_incident=True)
    scattered = fn(FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi, include_incident=False)

    e_inc, h_inc = _incident_plane_wave(theta, phi, r)
    expected = e_inc if which == "E" else h_inc
    np.testing.assert_allclose(total - scattered, expected, rtol=1e-10, atol=1e-14)


def test_eh_near_scattered_matches_the_separate_calls():
    """The combined call must agree with the individual ones when incident is off."""
    theta, phi = _angles()
    r = np.full_like(theta, 1.4)
    args = (FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi)
    e_both, h_both = eh_near(*args, include_incident=False)
    np.testing.assert_allclose(e_both, e_near(*args, include_incident=False), rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(h_both, h_near(*args, include_incident=False), rtol=1e-12, atol=1e-300)


def test_include_incident_is_irrelevant_inside_the_sphere():
    """There is no incident field to add inside; the internal field stands alone."""
    theta, phi = _angles()
    r = np.full_like(theta, 0.3)  # inside
    args = (FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi)
    for fn in (e_near, h_near):
        np.testing.assert_array_equal(fn(*args, include_incident=True), fn(*args, include_incident=False))
    e_on, h_on = eh_near(*args, include_incident=True)
    e_off, h_off = eh_near(*args, include_incident=False)
    np.testing.assert_array_equal(e_on, e_off)
    np.testing.assert_array_equal(h_on, h_off)


def test_explicit_term_count_reproduces_the_default():
    """Asking for the number of terms the default picks must change nothing."""
    theta, phi = _angles()
    r = np.full_like(theta, 1.4)
    args = (FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi)
    x = np.pi * FIELD_D * FIELD_N_ENV / FIELD_LAMBDA0
    explicit = wiscombe_terms(x) + EXTRA_FIELD_ORDERS

    for fn in (e_near, h_near):
        np.testing.assert_array_equal(fn(*args, n_pole=explicit), fn(*args))
    e_exp, h_exp = eh_near(*args, n_pole=explicit)
    e_def, h_def = eh_near(*args)
    np.testing.assert_array_equal(e_exp, e_def)
    np.testing.assert_array_equal(h_exp, h_def)


def test_too_few_terms_gives_a_visibly_different_field():
    """A deliberately short series must not silently match the converged one."""
    theta, phi = _angles()
    r = np.full_like(theta, 1.4)
    args = (FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, r, theta, phi)
    truncated = e_near(*args, n_pole=2)
    converged = e_near(*args)
    assert np.max(np.abs(truncated - converged)) > 1e-3


def test_cartesian_wrappers_take_a_term_count_too():
    """The Cartesian entry points must pass n_pole through as well."""
    u = np.linspace(-1.3, 1.3, 5)
    grid_x, grid_z = np.meshgrid(u, u)
    grid_y = np.zeros_like(grid_x)
    x = np.pi * FIELD_D * FIELD_N_ENV / FIELD_LAMBDA0
    explicit = wiscombe_terms(x) + EXTRA_FIELD_ORDERS
    args = (FIELD_LAMBDA0, FIELD_D, FIELD_M, FIELD_N_ENV, grid_x, grid_y, grid_z)

    np.testing.assert_array_equal(e_near_cartesian(*args, n_pole=explicit), e_near_cartesian(*args))
    np.testing.assert_array_equal(h_near_cartesian(*args, n_pole=explicit), h_near_cartesian(*args))
    e_exp, h_exp = eh_near_cartesian(*args, n_pole=explicit)
    e_def, h_def = eh_near_cartesian(*args)
    np.testing.assert_array_equal(e_exp, e_def)
    np.testing.assert_array_equal(h_exp, h_def)
