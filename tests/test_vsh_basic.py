"""
Test Suite for Vector Spherical Harmonics in Mie Scattering.

This module contains pytest-based unit tests for the computation of vector
spherical harmonics (VSH) used in Mie scattering. The tests compare numerical
implementations of the magnetic (`M_odd`, `M_even`) and electric (`N_odd`, `N_even`)
vector spherical harmonics against their corresponding analytical expressions.

Currently, only the n=1 mode is tested.

Tested Functions:
-----------------
- `M_odd(n, d_sphere, r, theta, phi, k)`: Computes the nth odd magnetic VSH.
- `M_even(n, d_sphere, r, theta, phi, k)`: Computes the nth even magnetic VSH.
- `N_odd(n, d_sphere, r, theta, phi, k)`: Computes the nth odd electric VSH.
- `N_even(n, d_sphere, r, theta, phi, k)`: Computes the nth even electric VSH.

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


def analytic_M1_odd(d_sphere, r, theta, phi, k):
    """
    Calculate the first odd magnetic vector spherical harmonic.

    This is M_{omn}(rho) with n=1 and m=1

    This matches the convention in wikipedia for vector spherical harmonics
    and that found in the paper by Ladutenko.  It differs from Bohren & Huffman
    which has the odd and even M modes switched.
    """
    rho = k * r
    if r < d_sphere / 2:
        zn = spherical_jn(1, rho)
    else:
        zn = spherical_h1(1, rho)
    return (0.0, -np.sin(phi) * zn, -np.cos(phi) * np.cos(theta) * zn)


def analytic_M1_even(d_sphere, r, theta, phi, k):
    """
    Calculate the first even magnetic vector spherical harmonic.

    This is M_{emn}(rho) with n=1 and m=1
    """
    rho = k * r
    if r < d_sphere / 2:
        zn = spherical_jn(1, rho)
    else:
        zn = spherical_h1(1, rho)
    return (0.0, np.cos(phi) * zn, -np.sin(phi) * np.cos(theta) * zn)


def analytic_N1_odd(d_sphere, r, theta, phi, k):
    """
    Calculate the first odd electric vector spherical harmonic.

    This is N_{omn}(rho) with n=1 and m=1
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


def analytic_N1_even(d_sphere, r, theta, phi, k):
    """
    Calculate the first even electric vector spherical harmonic.

    This is N_{emn}(rho) with n=1 and m=1
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
    d_sphere = 1.0
    phi = np.pi / 6
    k = 2 * np.pi
    n = 1

    v_me = M_odd(n, d_sphere, r, theta, phi, k)
    v_mo = M_even(n, d_sphere, r, theta, phi, k)
    v_no = N_odd(n, d_sphere, r, theta, phi, k)
    v_ne = N_even(n, d_sphere, r, theta, phi, k)

    a_me = analytic_M1_even(d_sphere, r, theta, phi, k)
    a_mo = analytic_M1_odd(d_sphere, r, theta, phi, k)
    a_ne = analytic_N1_even(d_sphere, r, theta, phi, k)
    a_no = analytic_N1_odd(d_sphere, r, theta, phi, k)

    np.testing.assert_allclose(
        v_me, a_me, rtol=1e-6, err_msg=f"Mismatch in M_even for {region}, theta={theta}"
    )
    np.testing.assert_allclose(
        v_mo, a_mo, rtol=1e-6, err_msg=f"Mismatch in M_odd for {region}, theta={theta}"
    )
    np.testing.assert_allclose(
        v_ne, a_ne, rtol=1e-6, err_msg=f"Mismatch in N_even for {region}, theta={theta}"
    )
    np.testing.assert_allclose(
        v_no, a_no, rtol=1e-6, err_msg=f"Mismatch in N_odd for {region}, theta={theta}"
    )
