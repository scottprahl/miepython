"""
Test Suite for Vector Spherical Harmonics in Mie Scattering.

This module contains pytest-based unit tests for the computation of vector
spherical harmonics (VSH) used in Mie scattering. The tests compare numerical
implementations of the magnetic (`M_odd`, `M_even`) and electric (`N_odd`, `N_even`)
vector spherical harmonics against their corresponding analytical expressions.

Currently, only the n=1 mode is tested.

Tested Functions:
-----------------
- `M_odd(n, k, d_sphere, r, theta, phi)`: Computes the nth odd magnetic VSH.
- `M_even(n, k, d_sphere, r, theta, phi)`: Computes the nth even magnetic VSH.
- `N_odd(n, k, d_sphere, r, theta, phi)`: Computes the nth odd electric VSH.
- `N_even(n, k, d_sphere, r, theta, phi)`: Computes the nth even electric VSH.

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

# Constants for testing
lambda0 = 500e-9  # wavelength in meters
d_sphere = 1e-6  # sphere diameter in meters
r_boundary = d_sphere / 2
m_sphere = 1.5   # refractive index inside sphere
m_env = 1.0      # refractive index of environment
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
