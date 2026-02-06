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
"""

import numpy as np
from miepython.vsh import M_even_array, M_odd_array, N_even_array, N_odd_array
from miepython.util import cartesian_to_spherical, spherical_vector_to_cartesian
import miepython as mie

__all__ = (
    "e_near",
    "h_near",
    "eh_near",
    "e_near_cartesian",
    "h_near_cartesian",
    "eh_near_cartesian",
    "e_plane",
    "e_far",
)


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


def _incident_h_spherical(lambda0, n_env, r, theta, phi, amplitude=1.0):
    """Incident plane wave (propagating +z, H along +y) in spherical components."""
    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * r * np.cos(theta))
    amp = amplitude * phase

    h_r = amp * np.sin(theta) * np.sin(phi)
    h_theta = amp * np.cos(theta) * np.sin(phi)
    h_phi = amp * np.cos(phi)
    return np.array([h_r, h_theta, h_phi])


def _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole):
    """Return Mie coefficients [a, b, c, d] with consistent medium scaling."""
    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    a, b, c, d = mie.coefficients(m_rel, x, n_pole=n_pole, internal=True)
    return np.array([a, b, c, d])


def _vectorized_field_eval(evaluator, r, theta, phi):
    """
    Evaluate a field callback on scalar or broadcastable spherical coordinates.

    The callback must accept scalar (r, theta, phi) and return a 3-vector.
    """
    rr, tt, pp = np.broadcast_arrays(np.asarray(r), np.asarray(theta), np.asarray(phi))

    if rr.ndim == 0:
        return evaluator(float(rr), float(tt), float(pp))

    out = np.empty((3,) + rr.shape, dtype=complex)
    for idx in np.ndindex(rr.shape):
        out[(slice(None),) + idx] = evaluator(float(rr[idx]), float(tt[idx]), float(pp[idx]))
    return out


def _vectorized_field_pair_eval(evaluator, r, theta, phi):
    """
    Evaluate a field-pair callback on scalar or broadcastable spherical coordinates.

    The callback must accept scalar (r, theta, phi) and return (E, H) 3-vectors.
    """
    rr, tt, pp = np.broadcast_arrays(np.asarray(r), np.asarray(theta), np.asarray(phi))

    if rr.ndim == 0:
        return evaluator(float(rr), float(tt), float(pp))

    e_out = np.empty((3,) + rr.shape, dtype=complex)
    h_out = np.empty((3,) + rr.shape, dtype=complex)
    for idx in np.ndindex(rr.shape):
        e_val, h_val = evaluator(float(rr[idx]), float(tt[idx]), float(pp[idx]))
        e_out[(slice(None),) + idx] = e_val
        h_out[(slice(None),) + idx] = h_val
    return e_out, h_out


def _cartesian_to_spherical_safe(x, y, z):
    """Array-safe Cartesian->spherical conversion for field wrappers."""
    xx, yy, zz = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(z))
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(rr == 0, 1.0, np.clip(zz / rr, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    phi = np.arctan2(yy, xx)
    return rr, theta, phi


def _spherical_components_to_cartesian(field_sph, r, theta, phi):
    """Convert [F_r, F_theta, F_phi] to [F_x, F_y, F_z]."""
    fx, fy, fz = spherical_vector_to_cartesian(field_sph[0], field_sph[1], field_sph[2], r, theta, phi)
    return np.array([fx, fy, fz])


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


def _h_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident):
    """Core near-field magnetic-field evaluation using precomputed Mie coefficients."""
    a, b, c, d = abcd

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2
    m_index = np.conjugate(m_sphere) if inside else n_env

    M_rad, M_the, M_phi = M_even_array(N, lambda0, d_sphere, m_index, r, theta, phi)
    N_rad, N_the, N_phi = N_odd_array(N, lambda0, d_sphere, m_index, r, theta, phi)

    if inside:
        m_rel = np.conjugate(m_sphere / n_env)
        H_rad = m_rel * np.sum(scale * (-d * M_rad - 1j * c * N_rad))
        H_the = m_rel * np.sum(scale * (-d * M_the - 1j * c * N_the))
        H_phi = m_rel * np.sum(scale * (-d * M_phi - 1j * c * N_phi))
    else:
        H_rad = np.sum(scale * (1j * b * N_rad + a * M_rad))
        H_the = np.sum(scale * (1j * b * N_the + a * M_the))
        H_phi = np.sum(scale * (1j * b * N_phi + a * M_phi))

        if include_incident:
            Hi_rad, Hi_the, Hi_phi = _incident_h_spherical(lambda0, n_env, r, theta, phi)
            H_rad += Hi_rad
            H_the += Hi_the
            H_phi += Hi_phi

    return np.array([H_rad, H_the, -H_phi])


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
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    evaluator = lambda rr, tt, pp: _e_near_abcd(  # noqa: E731
        abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident
    )
    return _vectorized_field_eval(evaluator, r, theta, phi)


def h_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True, n_pole=0, abcd=None):
    """
    Calculate the magnetic field in and around a sphere.

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
        tuple: Magnetic field components (H_r, H_theta, H_phi).
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    evaluator = lambda rr, tt, pp: _h_near_abcd(  # noqa: E731
        abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident
    )
    return _vectorized_field_eval(evaluator, r, theta, phi)


def eh_near(
    lambda0,
    d_sphere,
    m_sphere,
    n_env,
    r,
    theta,
    phi,
    include_incident=True,
    n_pole=0,
    abcd=None,
):
    """
    Calculate electric and magnetic fields in and around a sphere.

    Returns:
        tuple: (E, H), each in spherical components.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    def evaluator(rr, tt, pp):
        e_val = _e_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident)
        h_val = _h_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident)
        return e_val, h_val

    return _vectorized_field_pair_eval(evaluator, r, theta, phi)


def e_near_cartesian(
    lambda0,
    d_sphere,
    m_sphere,
    n_env,
    x,
    y,
    z,
    include_incident=True,
    n_pole=0,
    abcd=None,
):
    """Electric near field in Cartesian components at (x, y, z)."""
    r, theta, phi = _cartesian_to_spherical_safe(x, y, z)
    e_sph = e_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, n_pole, abcd)
    return _spherical_components_to_cartesian(e_sph, r, theta, phi)


def h_near_cartesian(
    lambda0,
    d_sphere,
    m_sphere,
    n_env,
    x,
    y,
    z,
    include_incident=True,
    n_pole=0,
    abcd=None,
):
    """Magnetic near field in Cartesian components at (x, y, z)."""
    r, theta, phi = _cartesian_to_spherical_safe(x, y, z)
    h_sph = h_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, n_pole, abcd)
    return _spherical_components_to_cartesian(h_sph, r, theta, phi)


def eh_near_cartesian(
    lambda0,
    d_sphere,
    m_sphere,
    n_env,
    x,
    y,
    z,
    include_incident=True,
    n_pole=0,
    abcd=None,
):
    """Electric and magnetic near fields in Cartesian components at (x, y, z)."""
    r, theta, phi = _cartesian_to_spherical_safe(x, y, z)
    e_sph, h_sph = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, n_pole, abcd)
    e_xyz = _spherical_components_to_cartesian(e_sph, r, theta, phi)
    h_xyz = _spherical_components_to_cartesian(h_sph, r, theta, phi)
    return e_xyz, h_xyz


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
