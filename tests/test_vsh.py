"""
Test Suite for Vector Spherical Harmonics in Mie Scattering.

This module contains pytest-based unit tests for the computation of vector
spherical harmonics (VSH) used in Mie scattering. The tests compare numerical
implementations of the magnetic (`M_odd`, `M_even`) and electric (`N_odd`, `N_even`)
vector spherical harmonics against their corresponding analytical expressions.

Currently, only the n=1 mode is tested.

Tested Functions:
-----------------
All four share the signature `(n, lambda0, d_sphere, m_index, r, theta, phi)`.

- `M_odd`: Computes the nth odd magnetic VSH.
- `M_even`: Computes the nth even magnetic VSH.
- `N_odd`: Computes the nth odd electric VSH.
- `N_even`: Computes the nth even electric VSH.

Conventions:
------------
- The sign conventions used in these tests follow the Wikipedia page on
  "Vector Spherical Harmonics" as well as Ladutenko's paper:
  DOI: https://doi.org/10.1016/j.cpc.2017.01.017.
- The spherical Bessel functions (`spherical_jn`) are used inside the sphere
  (r < d_sphere/2), while spherical Hankel functions of the first kind
  (`spherical_h1`) are used outside the sphere (r ≥ d_sphere/2).

Testing Strategy:
-----------------
- The tests are parameterized for two different regions:
  1. **Inside the sphere**: r = 0.3
  2. **Outside the sphere**: r = 0.75
- The tests are run for multiple polar angles (theta = π/6, π/3, π/2.5).
- The relative tolerance (`rtol=1e-6`) is used to ensure accurate comparisons.
"""

import numpy as np
import pytest
from scipy.special import spherical_jn
from miepython.bessel import spherical_h1, d_spherical_jn, d_spherical_h1
from miepython.vsh import M_odd, M_even, N_odd, N_even
from miepython import vsh
from miepython.field import _vsh_components_base  # the duplicate implementation

# Constants for testing
LAMBDA0_DEFAULT = 500e-9  # wavelength in meters
D_SPHERE_DEFAULT = 1e-6  # sphere diameter in meters
r_boundary = D_SPHERE_DEFAULT / 2
m_sphere = 1.5  # refractive index inside sphere
m_env = 1.0  # refractive index of environment
theta_test = np.pi / 4
phi_test = np.pi / 4

# Import the vector spherical harmonic functions


def analytic_M1_odd(k, d_sphere, r, theta, phi):
    """
    Calculate the first odd magnetic vector spherical harmonic.

    This is M_{omn}(rho) with n=1 and m=1

    This matches the convention in wikipedia for vector spherical harmonics
    and that found in the paper by Ladutenko.  It differs from Bohren & Huffman
    which has the odd and even M modes switched.

    Args:
        k (float): Wave number of the incident wave.
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis)
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    """
    rho = k * r
    if r < d_sphere / 2:
        zn = spherical_jn(1, rho)
    else:
        zn = spherical_h1(1, rho)
    return (0.0, -np.sin(phi) * zn, -np.cos(phi) * np.cos(theta) * zn)


def analytic_M1_even(k, d_sphere, r, theta, phi):
    """
    Calculate the first even magnetic vector spherical harmonic.

    This is M_{emn}(rho) with n=1 and m=1

    Args:
        k (float): Wave number of the incident wave.
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis)
        phi (float): Azimuthal angle in radians. (angle from x-axis).
    """
    rho = k * r
    if r < d_sphere / 2:
        zn = spherical_jn(1, rho)
    else:
        zn = spherical_h1(1, rho)
    return (0.0, np.cos(phi) * zn, -np.sin(phi) * np.cos(theta) * zn)


def analytic_N1_odd(k, d_sphere, r, theta, phi):
    """
    Calculate the first odd electric vector spherical harmonic.

    This is N_{omn}(rho) with n=1 and m=1

    Args:
        k (float): Wave number of the incident wave.
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis)
        phi (float): Azimuthal angle in radians. (angle from x-axis).
    """
    rho = k * r
    n = 1  # order of the spherical harmonic
    if r < d_sphere / 2:
        zn = spherical_jn(n, rho)
        zn_prime_over_rho = zn / rho + d_spherical_jn(n, rho)
    else:
        zn = spherical_h1(n, rho)
        zn_prime_over_rho = zn / rho + d_spherical_h1(n, rho)

    nr = np.sin(phi) * 2 * np.sin(theta) * zn / rho
    nth = np.sin(phi) * np.cos(theta) * zn_prime_over_rho
    nph = np.cos(phi) * zn_prime_over_rho
    return (nr, nth, nph)


def analytic_N1_even(k, d_sphere, r, theta, phi):
    """
    Calculate the first even electric vector spherical harmonic.

    This is N_{emn}(rho) with n=1 and m=1

    Args:
        k (float): Wave number of the incident wave.
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis)
        phi (float): Azimuthal angle in radians. (angle from x-axis).
    """
    n = 1  # order of the spherical harmonic
    rho = k * r
    if r < d_sphere / 2:
        zn = spherical_jn(n, rho)
        zn_prime_over_rho = zn / rho + d_spherical_jn(n, rho)
    else:
        zn = spherical_h1(n, rho)
        zn_prime_over_rho = zn / rho + d_spherical_h1(n, rho)

    nr = np.cos(phi) * 2 * np.sin(theta) * zn / rho
    nth = np.cos(phi) * np.cos(theta) * zn_prime_over_rho
    nph = -np.sin(phi) * zn_prime_over_rho
    return (nr, nth, nph)


# Parameterize tests for two regions: inside sphere (r = 0.3) and outside sphere (r = 0.75)
# and for a set of theta values.
@pytest.mark.parametrize("r, region", [(0.3, "inside sphere"), (0.75, "outside sphere")])
@pytest.mark.parametrize("theta", [np.pi / 6, np.pi / 3, np.pi / 2.5])
def test_vector_spherical_harmonics(r, theta, region):
    """Test n=1, m=1 vector spherical harmonics."""
    n = 1
    d_sphere = 1.0
    lambda0 = 1
    phi = np.pi / 6

    if r < d_sphere / 2:
        m_index = 1.5
    else:
        m_index = 1.0

    v_me = M_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_mo = M_even(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_no = N_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_ne = N_even(n, lambda0, d_sphere, m_index, r, theta, phi)

    k = m_index * 2 * np.pi / lambda0
    a_me = analytic_M1_even(k, d_sphere, r, theta, phi)
    a_mo = analytic_M1_odd(k, d_sphere, r, theta, phi)
    a_ne = analytic_N1_even(k, d_sphere, r, theta, phi)
    a_no = analytic_N1_odd(k, d_sphere, r, theta, phi)

    theta = np.degrees(theta)
    phi = np.degrees(phi)
    np.testing.assert_allclose(
        v_me, a_me, rtol=1e-6, err_msg=f"M_even wrong when m={m_index}, 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_mo, a_mo, rtol=1e-6, err_msg=f"M_odd wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_ne, a_ne, rtol=1e-6, err_msg=f"N_even wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_no, a_no, rtol=1e-6, err_msg=f"N_odd wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )


@pytest.mark.parametrize("m_index", [1.0, 1.5, 1.5 - 0.1j])
def test_vector_spherical_harmonics2(m_index):
    """Test n=1, m=1 vector spherical harmonics."""
    n = 1
    d_sphere = 1.0
    lambda0 = 1

    r = 0.3
    theta = np.pi / 6
    phi = np.pi / 6
    region = "inside sphere"

    v_me = M_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_mo = M_even(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_no = N_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
    v_ne = N_even(n, lambda0, d_sphere, m_index, r, theta, phi)

    k = m_index * 2 * np.pi / lambda0
    a_me = analytic_M1_even(k, d_sphere, r, theta, phi)
    a_mo = analytic_M1_odd(k, d_sphere, r, theta, phi)
    a_ne = analytic_N1_even(k, d_sphere, r, theta, phi)
    a_no = analytic_N1_odd(k, d_sphere, r, theta, phi)

    theta = np.degrees(theta)
    phi = np.degrees(phi)
    np.testing.assert_allclose(
        v_me, a_me, rtol=1e-6, err_msg=f"M_even wrong when m={m_index}, 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_mo, a_mo, rtol=1e-6, err_msg=f"M_odd wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_ne, a_ne, rtol=1e-6, err_msg=f"N_even wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )
    np.testing.assert_allclose(
        v_no, a_no, rtol=1e-6, err_msg=f"N_odd wrong when m={m_index} 𝜃={theta}° ɸ={phi}° {region}"
    )


# @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
# def test_boundary_conditions_M(n):
#     """Test magnetic field continuity at the sphere surface for harmonics M."""
#     # Inside
#     M_inside_odd = M_odd(n, lambda0, d_sphere, m_sphere, r_boundary - 1e-12, theta_test, phi_test)
#     M_inside_even = M_even(n, lambda0, d_sphere, m_sphere, r_boundary - 1e-12, theta_test, phi_test)
#
#     # Outside
#     M_outside_odd = M_odd(n, lambda0, d_sphere, 1.0, r_boundary + 1e-12, theta_test, phi_test)
#     M_outside_even = M_even(n, lambda0, d_sphere, 1.0, r_boundary + 1e-12, theta_test, phi_test)
#
#     np.testing.assert_allclose(M_inside_odd, M_outside_odd, atol=1e-9)
#     np.testing.assert_allclose(M_inside_even, M_outside_even, atol=1e-9)
#
# @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
# def test_boundary_conditions_N(n):
#     """Test electric field continuity at the sphere surface for harmonics N."""
#     theta_test = np.pi / 4
#     phi_test = np.pi / 3
#
#     # Inside
#     N_inside_odd = N_odd(n, lambda0, d_sphere, m_sphere, r_boundary - 1e-12, theta_test, phi_test)
#     N_inside_even = N_even(n, lambda0, d_sphere, m_sphere, r_boundary - 1e-12, theta_test, phi_test)
#
#     # Outside
#     N_outside_odd = N_odd(n, lambda0, d_sphere, 1.0, r_boundary + 1e-12, theta_test, phi_test)
#     N_outside_even = N_even(n, lambda0, d_sphere, 1.0, r_boundary + 1e-12, theta_test, phi_test)
#
#     # Tangential electric fields continuity
#     np.testing.assert_allclose(N_inside_odd, N_outside_odd, atol=1e-9)
#     np.testing.assert_allclose(N_inside_even, N_outside_even, atol=1e-9)
#
# @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
# def test_radial_derivative_continuity(n):
#     """Test continuity of radial derivatives at the boundary for N and M harmonics."""
#     delta = 1e-12
#     theta_test = np.pi / 4
#     phi_test = np.pi / 3
#
#     def radial_derivative(func, r, delta, *args):
#         field_out = func(n, lambda0, d_sphere, args[-1], r + delta, *args[:-1])
#         field_in = func(n, lambda0, d_sphere, args[-1], r - delta, *args[:-1])
#         return (field_out - field_in) / (2 * delta)
#
#     # Radial derivative continuity for M
#     dM_dr_inside = radial_derivative(M_odd, r_boundary - delta, [theta_test, phi_test, m_sphere])
#     dM_dr_outside = radial_derivative(M_odd, r_boundary + delta, [theta_test, phi_test, m_sphere])
#
#     np.testing.assert_allclose(dM_dr_inside, dM_dr_outside, atol=1e-9)
#
#     # For N harmonics
#     dN_dr_inside = radial_derivative(N_even, r_boundary - delta, [theta_test, phi_test, m_sphere])
#     dN_dr_outside = radial_derivative(N_even, r_boundary + delta, [theta_test, phi_test, m_sphere])
#
#     np.testing.assert_allclose(dN_dr_inside, dN_dr_outside, atol=1e-9)


# ---------------------------------------------------------------------------
# The array helpers, the degree-valued angles, and the small-argument branch
# ---------------------------------------------------------------------------

LAMBDA0 = 0.8
D_SPHERE = 1.0
N_TERMS = 4


@pytest.mark.parametrize("r,label", [(0.3, "inside"), (0.9, "inside"), (1.7, "outside")])
@pytest.mark.parametrize("m_index", [1.5 + 0j, 1.33 - 0.1j])
def test_array_helpers_match_the_scalar_ones(r, label, m_index):
    """M_*_array and N_*_array must stack exactly what the scalar versions give."""
    theta, phi = np.radians(37.0), np.radians(58.0)
    pairs = (
        (vsh.M_odd_array, vsh.M_odd),
        (vsh.M_even_array, vsh.M_even),
        (vsh.N_odd_array, vsh.N_odd),
        (vsh.N_even_array, vsh.N_even),
    )
    for array_fn, scalar_fn in pairs:
        got = array_fn(N_TERMS, LAMBDA0, D_SPHERE, m_index, r, theta, phi)
        want = np.array([scalar_fn(n, LAMBDA0, D_SPHERE, m_index, r, theta, phi) for n in range(1, N_TERMS + 1)]).T
        assert got.shape == (3, N_TERMS), f"{array_fn.__name__} {label}"
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=1e-300, err_msg=f"{array_fn.__name__} disagrees with {scalar_fn.__name__}"
        )


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_degrees_and_radians_agree(n):
    """deg=True must be the same curve, just labelled differently."""
    degrees = np.array([5.0, 37.0, 90.0, 143.0, 175.0])
    radians = np.radians(degrees)
    np.testing.assert_allclose(vsh.mie_pi(n, degrees, deg=True), vsh.mie_pi(n, radians), rtol=1e-12)
    np.testing.assert_allclose(vsh.mie_tau(n, degrees, deg=True), vsh.mie_tau(n, radians), rtol=1e-12)


def test_pi_and_tau_avoid_the_poles():
    """The +-1 guard keeps the division by sin(theta) finite at the poles."""
    for n in (1, 2, 4):
        for theta in (0.0, np.pi):
            assert np.isfinite(vsh.mie_pi(n, theta))
            assert np.isfinite(vsh.mie_tau(n, theta))
        both = vsh.mie_pi(n, np.array([0.0, np.pi]))
        assert np.all(np.isfinite(both))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_argument_branch_is_continuous(n):
    """N_base switches formula at |rho| = 0.01; the two must agree there."""
    m_index = 1.5 + 0j
    # kr chosen so |rho| lands just either side of the threshold
    kr_below = 0.0099 / abs(m_index)
    kr_above = 0.0101 / abs(m_index)
    theta = np.radians(50.0)
    below = vsh.N_base(n, m_index, kr_below, theta, inside=True)
    above = vsh.N_base(n, m_index, kr_above, theta, inside=True)
    assert np.all(np.isfinite(below)) and np.all(np.isfinite(above))
    # a 2% step in argument must not move the result by more than a few percent
    for lo, hi in zip(below, above):
        if abs(hi) > 1e-250:
            assert abs(lo - hi) / abs(hi) < 0.05, f"discontinuity at n={n}"


@pytest.mark.parametrize("r", [0.0002, 0.0005])  # |m*kr| stays under 0.01
def test_small_argument_branch_is_actually_taken(r):
    """A tiny radius must reach the series expansion and stay finite."""
    m_index = 1.5 + 0j
    kr = 2 * np.pi * r / LAMBDA0
    assert abs(m_index * kr) < 0.01, "test no longer exercises the small-argument path"
    for n in (1, 2, 3):
        out = vsh.N_base(n, m_index, kr, np.radians(40.0), inside=True)
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("r,inside", [(0.3, True), (1.7, False)])
def test_vsh_module_agrees_with_the_field_module(r, inside):
    """vsh.py and field.py hold two copies of this maths; they must not drift."""
    sphere_index, env_index = 1.5 + 0j, 1.0
    theta = np.radians(63.0)
    m_index = np.conjugate(sphere_index) if inside else env_index

    # field.py evaluates a batch of points at once, so hand it a batch of one and
    # read row 0 back out
    m_the, m_phi, n_rad, n_the, n_phi = _vsh_components_base(
        N_TERMS, LAMBDA0, m_index, np.array([r]), np.array([theta]), inside
    )

    kr = 2 * np.pi * r / LAMBDA0
    rho = m_index * kr
    for i, n in enumerate(range(1, N_TERMS + 1)):
        _, mb_the, mb_phi = vsh.M_base(n, rho, theta, inside)
        nb_rad, nb_the, nb_phi = vsh.N_base(n, m_index, kr, theta, inside)
        # field.py builds pi and tau from the kernel recurrence, vsh.py from lpmv,
        # so allow a little more room than pure round-off
        for label, want, got in (
            ("M_theta", mb_the, m_the[0, i]),
            ("M_phi", mb_phi, m_phi[0, i]),
            ("N_r", nb_rad, n_rad[0, i]),
            ("N_theta", nb_the, n_the[0, i]),
            ("N_phi", nb_phi, n_phi[0, i]),
        ):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300, err_msg=f"{label} n={n} r={r}")
