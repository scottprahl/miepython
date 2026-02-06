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

    M_odd(n, k, d_sphere, r, theta, phi)
        Compute the nth odd magnetic vector spherical harmonic (m=1).

    M_even(n, k, d_sphere, r, theta, phi)
        Compute the nth even magnetic vector spherical harmonic (m=1).

    N_odd(n, k, d_sphere, r, theta, phi)
        Compute the nth odd electric vector spherical harmonic (m=1).

    N_even(n, k, d_sphere, r, theta, phi)
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
import miepython as mie

__all__ = (
    "mie_tau",
    "mie_pi",
    "M_even",
    "M_odd",
    "N_even",
    "N_odd",
    "M_even_array",
    "M_odd_array",
    "N_even_array",
    "N_odd_array",
)


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


def M_base(n, rho, theta, inside):
    """
    Compute the non-phi part of magnetic vector spherical harmonic (m=1).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        rho (float): radius * k * m
        theta (float): Polar angle in radians (angle from z-axis).
        inside (bool): True if rho is inside sphere

    Returns:
        M_r, M_theta, M_phi without phi dependence
    """
    if inside:
        factor = spherical_jn(n, rho)
    else:
        factor = spherical_h1(n, rho)

    M_r = 0
    M_theta = mie_pi(n, theta) * factor
    M_phi = mie_tau(n, theta) * factor
    return np.array([M_r, M_theta, M_phi])


def M_odd(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the nth odd magnetic vector spherical harmonic (m=1).

    This function calculates the odd-parity magnetic vector spherical harmonic,
    denoted as M_{omn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: A tuple (M_r, M_theta, M_phi) representing the radial, polar,
        and azimuthal components of the odd magnetic vector spherical harmonic.
    """
    rho = 2 * np.pi * m_index * r / lambda0
    inside = r < d_sphere / 2

    M_r, M_theta, M_phi = M_base(n, rho, theta, inside)
    return np.array([M_r, np.cos(phi) * M_theta, -np.sin(phi) * M_phi])


def M_even(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the nth even magnetic vector spherical harmonic (m=1).

    This function calculates the even-parity magnetic vector spherical harmonic,
    denoted as M_{emn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: A tuple (M_r, M_theta, M_phi) representing the radial, polar,
        and azimuthal components of the even magnetic vector spherical harmonic.
    """
    rho = 2 * np.pi * m_index * r / lambda0
    inside = r < d_sphere / 2

    M_r, M_theta, M_phi = M_base(n, rho, theta, inside)
    return np.array([M_r, -np.sin(phi) * M_theta, -np.cos(phi) * M_phi])


def M_odd_array(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Generate the first n odd magnetic vector spherical harmonics (m=1).

    The wavenumber k=2𝜋m/λ₀ where m is the index of refraction of the
    sphere or medium at a distance r from the center of the sphere. λ₀
    is the wavelength in a vacuum.

    The units on d_sphere, r and are should be the same (e.g., microns)
    and those on k should be the reciprocal (e.g. 1/microns)

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: (M_r, M_theta, M_phi) where each is an array
    """
    M_r = np.zeros(n, dtype=complex)
    M_theta = np.zeros(n, dtype=complex)
    M_phi = np.zeros(n, dtype=complex)

    rho = 2 * np.pi * m_index * r / lambda0
    inside = r < d_sphere / 2
    for i in range(n):
        M_r[i], M_theta[i], M_phi[i] = M_base(i + 1, rho, theta, inside)

    return np.array([M_r, np.cos(phi) * M_theta, -np.sin(phi) * M_phi])


def M_even_array(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the nth even magnetic vector spherical harmonic (m=1).

    The wavenumber k=2𝜋m/λ₀ where m is the index of refraction of the
    sphere or medium at a distance r from the center of the sphere. λ₀
    is the wavelength in a vacuum.

    The units on d_sphere, r and are should be the same (e.g., microns)
    and those on k should be the reciprocal (e.g. 1/microns)

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: (M_r, M_theta, M_phi) where each is an array
    """
    M_r = np.zeros(n, dtype=complex)
    M_theta = np.zeros(n, dtype=complex)
    M_phi = np.zeros(n, dtype=complex)

    rho = 2 * np.pi * m_index * r / lambda0
    inside = r < d_sphere / 2
    for i in range(n):
        M_r[i], M_theta[i], M_phi[i] = M_base(i + 1, rho, theta, inside)

    return np.array([M_r, -np.sin(phi) * M_theta, -np.cos(phi) * M_phi])


def N_base(n, m_index, kr, theta, inside):
    """
    Compute non-angular component of the electric vector spherical harmonic (m=1).

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Multipole order (1 for dipole, 2 for quadrupole, etc.).
        m_index (complex): complex index of refraction
        kr (float): r * 2𝜋/λ₀, where r is radius to calculate the VSH
        theta (float): polar angle in radians
        inside (bool): True if rho is inside sphere

    Returns:
        factor1, factor2
    """
    rho = m_index * kr
    if inside:
        if abs(rho) < 0.01:
            factor2 = (n + 1) / factorial2(2 * n + 1) * rho ** (n - 1)
            factor1 = rho ** (n - 1) / factorial2(2 * n + 1)
        else:
            factor1 = spherical_jn(n, rho)
            # _D_calc returns D_1..D_N; request one extra term and take D_n.
            Dn = mie._D_calc(m_index, kr, n + 1)[n - 1]
            factor2 = factor1 * Dn
            factor1 /= rho
    else:
        factor1 = spherical_h1(n, rho) / rho
        factor2 = d_riccati_bessel_h1(n, rho) / rho

    N_r = n * (n + 1) * np.sin(theta) * mie_pi(n, theta) * factor1
    N_theta = mie_tau(n, theta) * factor2
    N_phi = mie_pi(n, theta) * factor2
    return np.array([N_r, N_theta, N_phi])


def N_odd(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the nth odd electric vector spherical harmonic (m=1).

    This function calculates the odd-parity electric vector spherical harmonic,
    denoted as N_{omn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The wavenumber k=2𝜋m/λ₀ where m is the index of refraction of the
    sphere or medium at a distance r from the center of the sphere. λ₀
    is the wavelength in a vacuum.

    The units on d_sphere, r and are should be the same (e.g., microns)
    and those on k should be the reciprocal (e.g. 1/microns)

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: A tuple (N_r, N_theta, N_phi) representing the radial, polar,
        and azimuthal components of the odd electric vector spherical harmonic.
    """
    kr = 2 * np.pi * r / lambda0
    inside = r < d_sphere / 2
    N_r, N_theta, N_phi = N_base(n, m_index, kr, theta, inside)
    return np.array([np.sin(phi) * N_r, np.sin(phi) * N_theta, np.cos(phi) * N_phi])


def N_even(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the nth even electric vector spherical harmonic (m=1).

    This function calculates the even-parity electric vector spherical harmonic,
    denoted as N_{emn}(rho), for the given multipole order `n`. The proper
    Bessel function is chosen based on whether the calculation is performed
    inside or outside the sphere.

    The units of lambda0, d_sphere, and r should be the same (e.g., microns).

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: A tuple (N_r, N_theta, N_phi) representing the radial, polar,
        and azimuthal components of the even electric vector spherical harmonic.
    """
    kr = 2 * np.pi * r / lambda0
    inside = r < d_sphere / 2
    N_r, N_theta, N_phi = N_base(n, m_index, kr, theta, inside)
    return np.array([np.cos(phi) * N_r, np.cos(phi) * N_theta, -np.sin(phi) * N_phi])


def N_odd_array(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Generate first n odd electric vector spherical harmonics (m=1).

    The units of lambda0, d_sphere, and r should be the same (e.g., microns).

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: [N_r, N_theta, N_phi] where each is an array
    """
    N_r = np.zeros(n, dtype=complex)
    N_theta = np.zeros(n, dtype=complex)
    N_phi = np.zeros(n, dtype=complex)

    kr = 2 * np.pi * r / lambda0
    inside = r < d_sphere / 2
    for i in range(n):
        N_r[i], N_theta[i], N_phi[i] = N_base(i + 1, m_index, kr, theta, inside)

    return np.array([np.sin(phi) * N_r, np.sin(phi) * N_theta, np.cos(phi) * N_phi])


def N_even_array(n, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Compute the first n even electric vector spherical harmonics.

    The units of lambda0, d_sphere, and r should be the same (e.g., microns).

    The conventions used follow the "Vector Spherical Harmonics" Wikipedia
    page and Ladutenko's paper (DOI: https://doi.org/10.1016/j.cpc.2017.01.017).

    Args:
        n (int): Harmonic (1 for dipole, 2 for quadrupole, etc.).
        lambda0 (float): Wavelength in a vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): Refractive index at position r.
        r (float): Radial distance from center of sphere.
        theta (float): Polar angle in radians (angle from z-axis).
        phi (float): Azimuthal angle in radians. (angle from x-axis).

    Returns:
        tuple: [N_r, N_theta, N_phi] where each is an array
    """
    N_r = np.zeros(n, dtype=complex)
    N_theta = np.zeros(n, dtype=complex)
    N_phi = np.zeros(n, dtype=complex)

    kr = 2 * np.pi * r / lambda0
    inside = r < d_sphere / 2
    for i in range(n):
        N_r[i], N_theta[i], N_phi[i] = N_base(i + 1, m_index, kr, theta, inside)

    return np.array([np.cos(phi) * N_r, np.cos(phi) * N_theta, -np.sin(phi) * N_phi])
