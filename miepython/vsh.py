"""
Vector Spherical Harmonics for Mie Scattering.

This module calculates the electric and magnetic vector spherical harmonics (VSH)
required for Mie scattering computations. The functions in this module compute
the VSH corresponding to the m=1 mode, which is commonly used in scattering
problems.

Sign Conventions:

    The sign conventions used in this module follow those presented on the
    "Vector Spherical Harmonics" webpage on Wikipedia as well as those described
    in the paper by Ladutenko (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).
    In particular, the odd and even modes are consistent with these references.

    The even and odd magnetic modes of Bohren & Huffman are reversed from those
    used here.

Bessel Function Selection:

    The appropriate spherical Bessel function is chosen based on the spatial
    region where the field is evaluated:

        - For points inside the sphere (r < d_sphere/2), the regular spherical
          Bessel functions (``spherical_jn``) are used.
        - For points outside the sphere (r >= d_sphere/2), the spherical Hankel
          functions of the first kind (``spherical_h1``) are employed to represent
          the radiating field.

Functions:

    M_odd(n, d_sphere, r, theta, phi, k)
        Compute the nth odd magnetic vector spherical harmonic (m=1).

    M_even(n, d_sphere, r, theta, phi, k)
        Compute the nth even magnetic vector spherical harmonic (m=1).

    N_odd(n, d_sphere, r, theta, phi, k)
        Compute the nth odd electric vector spherical harmonic (m=1).

    N_even(n, d_sphere, r, theta, phi, k)
        Compute the nth even electric vector spherical harmonic (m=1).

Additional Utility Functions:

    mie_pi(n, theta, deg=False)
        Computes the angular variation (𝜋ₙ(𝜃)) for the electric multipole field,
        based on the associated Legendre polynomial.

    mie_tau(n, theta, deg=False)
        Computes the angular variation (𝛕ₙ(𝜃)) for the magnetic multipole field,
        corresponding to the derivative of the associated Legendre polynomial.
"""

import numpy as np
from scipy.special import spherical_jn, lpmv, factorial2
from miepython.bessel import spherical_h1, d_riccati_bessel_h1
from miepython.core import _D_calc

__all__ = ("mie_tau", "mie_pi", "M_even", "M_odd", "N_even", "N_odd")


def mie_pi(n, theta, deg=False):
    """
    Computes the angular variation of the electric field arising from the nth electric multipole 𝜋_n(𝜃).

    The variation is given by associated nth Legendre polynomial P¹_n(cos(𝜃))/sin(𝜃)

    Care is taken to avoid division by zero and for this function to work with
    both scalars and arrays

    Parameters:
        n : Degree of multipole 1=dipole, 2=quadrupole, 3=octopole
        theta: angle in radians
        deg: if True then angle is in degrees

    Returns:
        array-like: Values of dP_n^m(x)/dx.
    """
    if deg:
        x = np.cos(np.radians(theta))
    else:
        x = np.cos(theta)

    # Ensure x is not ±1, whether a scalar or an element of an array
    x = 0.999999 if np.isscalar(x) and x == 1 else np.where(x == 1, 0.999999, x)
    x = -0.999999 if np.isscalar(x) and x == -1 else np.where(x == -1, -0.999999, x)

    return -lpmv(1, n, x) / np.sqrt(1 - x * x)


def mie_tau(n, theta, deg=False):
    """
    Computes the angular variation of the electric field arising from the nth magnetic multipole 𝛕_n(𝜃).

    The variation is given by the derivative of the nth associated Legendre polynomial d/d𝜃 P¹_n(cos𝜃)

    Care is taken to avoid division by zero and for this function to work with
    both scalars and arrays.

    Parameters:
        n : Degree of multipole 1=dipole, 2=quadrupole, 3=octopole
        theta: angle
        deg: if True then angle is in degrees

    Returns:
        array-like: Values of dP_n^m(x)/dx.
    """
    if deg:
        x = np.cos(np.radians(theta))
    else:
        x = np.cos(theta)

    # Ensure x is not ±1, whether a scalar or an element of an array
    x = 0.999999 if np.isscalar(x) and x == 1 else np.where(x == 1, 0.999999, x)
    x = -0.999999 if np.isscalar(x) and x == -1 else np.where(x == -1, -0.999999, x)

    return ((n + 1) * x * lpmv(1, n, x) - n * lpmv(1, n + 1, x)) / np.sqrt(1 - x * x)


def M_odd(n, d_sphere, r, theta, phi, k):
    """
    Compute the nth odd magnetic vector spherical harmonic (m=1).

    This function calculates the odd-parity magnetic vector spherical harmonic,
    denoted as M_{omn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Multipole order (1 for dipole, 2 for quadrupole, etc.).
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.
        k (float): Wave number of the incident wave.

    Returns:
        tuple: A tuple (Mr, Mtheta, Mphi) representing the radial, polar,
        and azimuthal components of the odd magnetic vector spherical harmonic.
    """
    rho = k * r
    if r < d_sphere / 2:
        factor = spherical_jn(n, rho)
    else:
        factor = spherical_h1(n, rho)

    Mr = 0
    Mtheta = np.cos(phi) * mie_pi(n, theta) * factor
    Mphi = -np.sin(phi) * mie_tau(n, theta) * factor
    return (Mr, Mtheta, Mphi)


def M_even(n, d_sphere, r, theta, phi, k):
    """
    Compute the nth even magnetic vector spherical harmonic (m=1).

    This function calculates the even-parity magnetic vector spherical harmonic,
    denoted as M_{emn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Multipole order (1 for dipole, 2 for quadrupole, etc.).
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.
        k (float): Wave number of the incident wave.

    Returns:
        tuple: A tuple (Mr, Mtheta, Mphi) representing the radial, polar,
        and azimuthal components of the even magnetic vector spherical harmonic.
    """
    rho = k * r
    if r < d_sphere / 2:
        factor = spherical_jn(n, rho)
    else:
        factor = spherical_h1(n, rho)

    Mr = 0
    Mtheta = -np.sin(phi) * mie_pi(n, theta) * factor
    Mphi = -np.cos(phi) * mie_tau(n, theta) * factor
    return (Mr, Mtheta, Mphi)


def N_odd(n, d_sphere, r, theta, phi, k):
    """
    Compute the nth odd electric vector spherical harmonic (m=1).

    This function calculates the odd-parity electric vector spherical harmonic,
    denoted as N_{omn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Multipole order (1 for dipole, 2 for quadrupole, etc.).
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.
        k (float): Wave number of the incident wave.

    Returns:
        tuple: A tuple (Nr, Ntheta, Nphi) representing the radial, polar,
        and azimuthal components of the odd electric vector spherical harmonic.
    """
    rho = k * r
    if r < d_sphere / 2:
        if rho < 0.01:
            factor2 = (n+1)/factorial2(2*n+1)*rho**(n-1)
            factor1 = rho**(n-1)/factorial2(2*n+1)
        else:
            factor1 = spherical_jn(n, rho)
            factor2 = factor1 * _D_calc(1, rho, n)[-1]
            factor1 /= rho
    else:
        factor1 = spherical_h1(n, rho) / rho
        factor2 = d_riccati_bessel_h1(n, rho) / rho

    Nr = np.sin(phi) * n * (n + 1) * np.sin(theta) * mie_pi(n, theta) * factor1
    Ntheta = np.sin(phi) * mie_tau(n, theta) * factor2
    Nphi = np.cos(phi) * mie_pi(n, theta) * factor2
    return (Nr, Ntheta, Nphi)


def N_even(n, d_sphere, r, theta, phi, k):
    """
    Compute the nth even electric vector spherical harmonic (m=1).

    This function calculates the even-parity electric vector spherical harmonic,
    denoted as N_{emn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Multipole order (1 for dipole, 2 for quadrupole, etc.).
        d_sphere (float): Diameter of the sphere.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.
        k (float): Wave number of the incident wave.

    Returns:
        tuple: A tuple (Nr, Ntheta, Nphi) representing the radial, polar,
        and azimuthal components of the even electric vector spherical harmonic.
    """
    rho = k * r
    if r < d_sphere / 2:
        if rho < 0.01:
            factor2 = (n+1)/factorial2(2*n+1)*rho**(n-1)
            factor1 = rho**(n-1)/factorial2(2*n+1)
        else:
            factor1 = spherical_jn(n, rho)
            factor2 = factor1 * _D_calc(1, rho, n)[-1]
            factor1 /= rho
    else:
        factor1 = spherical_h1(n, rho) / rho
        factor2 = d_riccati_bessel_h1(n, rho) / rho

    Nr = np.cos(phi) * n * (n + 1) * np.sin(theta) * mie_pi(n, theta) * factor1
    Ntheta = np.cos(phi) * mie_tau(n, theta) * factor2
    Nphi = -np.sin(phi) * mie_pi(n, theta) * factor2
    return (Nr, Ntheta, Nphi)
