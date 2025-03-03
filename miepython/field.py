"""
Electric and magnetic field calculations.
"""

import numpy as np
from miepython.vsh import M_even, M_odd, N_even, N_odd
from miepython.vsh import M_even_array, M_odd_array, N_even_array, N_odd_array
from miepython.util import cs
import miepython as mie

__all__ = ("e_near", "e_plane", "e_far")

import numpy as np


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian coordinates (x, y, z) to spherical (r, theta, phi)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0  # Polar angle (0 to pi)
    phi = np.arctan2(y, x)  # Azimuthal angle (-pi to pi)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def spherical_vector_to_cartesian(E_r, E_theta, E_phi, r, theta, phi):
    """Convert spherical components (E_r, E_theta, E_phi) to Cartesian (Ex, Ey, Ez)."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Ex = E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi - E_phi * sin_phi
    Ey = E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi + E_phi * cos_phi
    Ez = E_r * cos_theta - E_theta * sin_theta

    return Ex, Ey, Ez


def e_far(lambda0, d_sphere, m_sphere, r, theta, phi):
    """
    Calculate the electric field in the far field.

    Args:
        lambda0 (float): Wavelength the incident wave in vacuum.
        d_sphere (float): Diameter of the sphere.
        m_sphere (complex): Rrefractive index of the sphere
        r (float): Radial distance at which the field is evaluated.
        theta (float): Scattering angle (from z-axis) in radians.
        phi (float): Azimuthal angle (from x-axis) in radians.
        norm: type of normalization to use for scattering function

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    x = np.pi * d_sphere / lambda0
    jkr = 1j * 2 * np.pi * r / lambda0
    amp = np.exp(jkr) / (-jkr)

    S1, S2 = mie.S1_S2(m_sphere, x, np.cos(theta), norm="wiscombe")

    E_r = np.zeros_like(S1, dtype=complex)
    E_theta = S2 * amp * np.cos(phi)
    E_phi = S1 * amp * np.sin(phi)
    E_theta = np.conjugate(-E_theta)
    E_phi = np.conjugate(-E_phi)
    return np.array([E_r, E_theta, E_phi])


def e_far_old(lambda0, d_sphere, m_sphere, r, theta, phi):
    """
    Calculate the electric field in the far field.

    Args:
        lambda0 (float): Wavelength the incident wave in vacuum.
        d_sphere (float): Diameter of the sphere.
        m_sphere (complex): Rrefractive index of the sphere
        r (float): Radial distance at which the field is evaluated.
        theta (float): Scattering angle (from z-axis) in radians.
        phi (float): Azimuthal angle (from x-axis) in radians.
        norm: type of normalization to use for scattering function

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    x = np.pi * d_sphere / lambda0
    jkr = 1j * 2 * np.pi * r / lambda0
    amp = np.exp(jkr) / (-jkr)
    mu = np.cos(theta)

    a, b = mie._an_bn(m_sphere, x, 0)
    N = len(a)
    pi = np.zeros(N)
    tau = np.zeros(N)

    n = np.arange(1, N + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    mie._pi_tau(mu, pi, tau)

    E_r = complex(0)
    E_theta = np.sum(scale * (tau * a + pi * b)) * amp * np.cos(phi)
    E_phi = np.sum(scale * (pi * a + tau * b)) * amp * np.sin(phi)
    return np.array([E_r, E_theta, E_phi])


def e_near(abcd, lambda0, d_sphere, m_index, r, theta, phi):
    """
    Calculate the electric field in and around a sphere.

    Args:
        abcd (array): Mie coefficients [a, b, c, d]
        lambda0 (float): Wavelength of the incident wave in vacuum.
        d_sphere (float): Diameter of the sphere.
        m_index (complex): refractive index at r
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    a, b, c, d = abcd
    E_r = np.complex128(0)
    E_theta = np.complex128(0)
    E_phi = np.complex128(0)

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2

    for n in nn:
        Mn = M_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
        Nn = N_even(n, lambda0, d_sphere, m_index, r, theta, phi)

        if inside:
            E_r += scale[n - 1] * (c[n - 1] * Mn[0] - 1j * d[n - 1] * Nn[0])
            E_theta += scale[n - 1] * (c[n - 1] * Mn[1] - 1j * d[n - 1] * Nn[1])
            E_phi += scale[n - 1] * (c[n - 1] * Mn[2] - 1j * d[n - 1] * Nn[2])
        else:
            E_r += scale[n - 1] * (1j * a[n - 1] * Nn[0] - b[n - 1] * Mn[0])
            th_part = scale[n - 1] * (1j * a[n - 1] * Nn[1] - b[n - 1] * Mn[1])
            E_theta += th_part
            #            print(n, cs(Nn[1],-8), cs(Mn[1],-8), cs(th_part,-8), cs(E_theta,-8))
            E_phi += -scale[n - 1] * (1j * a[n - 1] * Nn[2] - b[n - 1] * Mn[2])

    return np.array([E_r, E_theta, E_phi])


def e_plane(x, y, z, N=100):
    """
    Calculate a plane wave using vector spherical harmonics

    Args:
        x, y, z: position

    Returns:
        tuple: Electric field components (E_z, E_y, E_z).
    """
    lambda0 = 1
    m_index = 1

    n = np.arange(1, N + 1)
    scale = (1j) ** n * (2 * n + 1) / n / (n + 1)

    r, theta, phi = cartesian_to_spherical(x, y, z)
    d_sphere = 3 * r

    M_r, M_theta, M_phi = M_odd_array(N, lambda0, d_sphere, m_index, r, theta, phi)
    N_r, N_theta, N_phi = N_even_array(N, lambda0, d_sphere, m_index, r, theta, phi)

    E_r = np.sum(scale * (M_r - 1j * N_r))
    E_theta = np.sum(scale * (M_theta - 1j * N_theta))
    E_phi = np.sum(scale * (M_phi - 1j * N_phi))

    Ex, Ey, Ez = spherical_vector_to_cartesian(E_r, E_theta, E_phi, r, theta, phi)
    return Ex, Ey, Ez
