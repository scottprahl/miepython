"""
Electric and magnetic field calculations.

Conventions
-----------
- Phasor time dependence: exp(-i * omega * t)
- Incident plane wave: propagation along +z, E along +x, H along +y, amplitude E0 = 1
- Spherical coordinates: theta from +z, phi from +x toward +y
- lambda0 is the vacuum wavelength
- n_env is the surrounding medium index; m_sphere is the sphere index
  (relative index m_rel = m_sphere / n_env)
- Size parameter: x = pi * d_sphere * n_env / lambda0
- Near-field definition: outside the sphere, total field = incident + scattered;
  inside the sphere, the internal field only
- Magnetic response: relative permeability mu_r = 1 (non-magnetic materials)

Only e_far has been verified to work.
"""

import numpy as np
from miepython.vsh import M_odd_array, N_even_array
from miepython.util import cartesian_to_spherical, spherical_vector_to_cartesian
import miepython as mie

__all__ = ("e_near", "e_plane", "e_far")


def e_far(lambda0, d_sphere, m_sphere, n_env, r, theta, phi):
    """
    Calculate the electric field in the far field.

    Args:
        lambda0 (float): Wavelength of the incident wave in vacuum.
        d_sphere (float): Diameter of the sphere.
        m_sphere (complex): Refractive index of the sphere.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Scattering angle (from z-axis) in radians.
        phi (float): Azimuthal angle (from x-axis) in radians.

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    jkr = 1j * 2 * np.pi * n_env * r / lambda0
    amp = np.exp(jkr) / (-jkr)

    S1, S2 = mie.S1_S2(m_rel, x, np.cos(theta), norm="wiscombe")

    E_r = np.zeros_like(S1, dtype=complex)
    E_theta = S2 * amp * np.cos(phi)
    E_phi = S1 * amp * np.sin(phi)
    E_theta = np.conjugate(-E_theta)
    E_phi = np.conjugate(-E_phi)
    return np.array([E_r, E_theta, E_phi])


# def e_far_old(lambda0, d_sphere, m_sphere, r, theta, phi):
#     """
#     Calculate the electric field in the far field.
#
#     Args:
#         lambda0 (float): Wavelength the incident wave in vacuum.
#         d_sphere (float): Diameter of the sphere.
#         m_sphere (complex): Rrefractive index of the sphere
#         r (float): Radial distance at which the field is evaluated.
#         theta (float): Scattering angle (from z-axis) in radians.
#         phi (float): Azimuthal angle (from x-axis) in radians.
#         norm: type of normalization to use for scattering function
#
#     Returns:
#         tuple: Electric field components (E_r, E_theta, E_phi).
#     """
#     x = np.pi * d_sphere / lambda0
#     jkr = 1j * 2 * np.pi * r / lambda0
#     amp = np.exp(jkr) / (-jkr)
#     mu = np.cos(theta)
#
#     a, b = mie.an_bn(m_sphere, x, 0)
#     N = len(a)
#     pi = np.zeros(N)
#     tau = np.zeros(N)
#
#     n = np.arange(1, N + 1)
#     scale = (2 * n + 1) / ((n + 1) * n)
#
#     mie._pi_tau(mu, pi, tau)
#
#     E_r = complex(0)
#     E_theta = np.sum(scale * (tau * a + pi * b)) * amp * np.cos(phi)
#     E_phi = np.sum(scale * (pi * a + tau * b)) * amp * np.sin(phi)
#     return np.array([E_r, E_theta, E_phi])
#
#
# def e_near_old(abcd, lambda0, d_sphere, m_index, r, theta, phi):
#     """
#     Calculate the electric field in and around a sphere.
#
#     Args:
#         abcd (array): Mie coefficients [a, b, c, d]
#         lambda0 (float): Wavelength of the incident wave in vacuum.
#         d_sphere (float): Diameter of the sphere.
#         m_index (complex): refractive index at r
#         r (float): Radial distance at which the field is evaluated.
#         theta (float): Polar angle in radians.
#         phi (float): Azimuthal angle in radians.
#
#     Returns:
#         tuple: Electric field components (E_r, E_theta, E_phi).
#     """
#     a, b, c, d = abcd
#     E_r = np.complex128(0)
#     E_theta = np.complex128(0)
#     E_phi = np.complex128(0)
#
#     N = len(a)
#     nn = np.arange(1, N + 1)
#     scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)
#
#     inside = r < d_sphere / 2
#
#     for n in nn:
#         Mn = M_odd(n, lambda0, d_sphere, m_index, r, theta, phi)
#         Nn = N_even(n, lambda0, d_sphere, m_index, r, theta, phi)
#
#         if inside:
#             E_r += scale[n - 1] * (c[n - 1] * Mn[0] - 1j * d[n - 1] * Nn[0])
#             E_theta += scale[n - 1] * (c[n - 1] * Mn[1] - 1j * d[n - 1] * Nn[1])
#             E_phi += scale[n - 1] * (c[n - 1] * Mn[2] - 1j * d[n - 1] * Nn[2])
#         else:
#             E_r += scale[n - 1] * (1j * a[n - 1] * Nn[0] - b[n - 1] * Mn[0])
#             th_part = scale[n - 1] * (1j * a[n - 1] * Nn[1] - b[n - 1] * Mn[1])
#             E_theta += th_part
#             E_phi += -scale[n - 1] * (1j * a[n - 1] * Nn[2] - b[n - 1] * Mn[2])
#
#     return np.array([E_r, E_theta, E_phi])


def _incident_e_spherical(lambda0, n_env, r, theta, phi, amplitude=1.0):
    """Incident plane wave (propagating +z, E along +x) in spherical components."""
    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * r * np.cos(theta))
    amp = amplitude * phase

    e_r = amp * np.sin(theta) * np.cos(phi)
    e_theta = amp * np.cos(theta) * np.cos(phi)
    e_phi = -amp * np.sin(phi)
    return np.array([e_r, e_theta, e_phi])


def _e_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident):
    """Core near-field evaluation using precomputed Mie coefficients."""
    a, b, c, d = abcd

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2
    # miepython coefficients follow the n-ik convention and are conjugated
    # internally; use conjugated sphere index so internal fields are consistent.
    m_index = np.conjugate(m_sphere) if inside else n_env

    M_rad, M_the, M_phi = M_odd_array(N, lambda0, d_sphere, m_index, r, theta, phi)
    N_rad, N_the, N_phi = N_even_array(N, lambda0, d_sphere, m_index, r, theta, phi)

    if inside:
        E_rad = np.sum(scale * (c * M_rad - 1j * d * N_rad))
        E_the = np.sum(scale * (c * M_the - 1j * d * N_the))
        E_phi = np.sum(scale * (c * M_phi - 1j * d * N_phi))
    else:
        E_rad = np.sum(scale * (1j * a * N_rad - b * M_rad))
        E_the = np.sum(scale * (1j * a * N_the - b * M_the))
        E_phi = np.sum(scale * (1j * a * N_phi - b * M_phi))

        if include_incident:
            Ei_rad, Ei_the, Ei_phi = _incident_e_spherical(lambda0, n_env, r, theta, phi)
            E_rad += Ei_rad
            E_the += Ei_the
            E_phi += Ei_phi

    return np.array([E_rad, E_the, -E_phi])


def e_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True, n_pole=0, abcd=None):
    """
    Calculate the electric field in and around a sphere.

    Args:
        lambda0 (float): Wavelength of the incident wave in vacuum.
        d_sphere (float): Diameter of the sphere.
        m_sphere (complex): Refractive index of the sphere.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial distance at which the field is evaluated.
        theta (float): Polar angle in radians.
        phi (float): Azimuthal angle in radians.
        include_incident (bool): If True, include incident field outside the sphere.
        n_pole (int): Multipole order (0 for automatic truncation).
        abcd (array, optional): Precomputed Mie coefficients [a, b, c, d].
            If provided, `n_pole` is ignored.

    Returns:
        tuple: Electric field components (E_r, E_theta, E_phi).
    """
    if abcd is None:
        x = np.pi * d_sphere * n_env / lambda0
        m_rel = m_sphere / n_env
        a, b, c, d = mie.coefficients(m_rel, x, n_pole=n_pole, internal=True)
        abcd = np.array([a, b, c, d])

    return _e_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident)


def e_plane(x, y, z, N=100):
    """
    Calculate a plane wave using vector spherical harmonics.

    Args:
        x: position
        y: position
        z: position
        N: number of points

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
