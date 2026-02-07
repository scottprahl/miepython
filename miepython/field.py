"""Near- and far-field electromagnetic fields for a homogeneous sphere.

This module evaluates complex phasor electric and magnetic fields for a sphere
embedded in a uniform medium.

Main entry points
-----------------
Use these for most workflows:

- ``e_near(...)``: electric field in spherical components ``[E_r, E_theta, E_phi]``.
- ``h_near(...)``: magnetic field in spherical components ``[H_r, H_theta, H_phi]``.
- ``eh_near(...)``: both electric and magnetic fields in one call.
- ``e_near_cartesian(...)`` / ``h_near_cartesian(...)`` / ``eh_near_cartesian(...)``:
  same near-field calculations in Cartesian components ``[Fx, Fy, Fz]``.
- ``e_far(...)``: scattered far-field electric components.

Quick examples
--------------
Evaluate the near electric field on a ring of points:

>>> import numpy as np
>>> import miepython.field as fields
>>> theta = np.linspace(0.0, np.pi, 181)
>>> phi = np.zeros_like(theta)
>>> r = np.full_like(theta, 1.5)  # same length units as lambda0 and d_sphere
>>> E = fields.e_near(
...     lambda0=1.0,
...     d_sphere=1.0,
...     m_sphere=1.5 + 0.0j,
...     n_env=1.0,
...     r=r,
...     theta=theta,
...     phi=phi,
... )
>>> E.shape
(3, 181)

Evaluate both fields in Cartesian coordinates on a 2D x-z slice:

>>> u = np.linspace(-1.5, 1.5, 101)
>>> X, Z = np.meshgrid(u, u, indexing="xy")
>>> E_xyz, H_xyz = fields.eh_near_cartesian(
...     lambda0=1.0,
...     d_sphere=1.0,
...     m_sphere=1.5 + 0.0j,
...     n_env=1.0,
...     x=X,
...     y=np.zeros_like(X),
...     z=Z,
... )
>>> E_xyz.shape, H_xyz.shape
((3, 101, 101), (3, 101, 101))

Reuse precomputed Mie coefficients for repeated field evaluations:

>>> import miepython as mie
>>> m_rel = (1.5 + 0.0j) / 1.0
>>> x_size = np.pi * 1.0 * 1.0 / 1.0
>>> a, b, c, d = mie.coefficients(m_rel, x_size, internal=True)
>>> abcd = np.array([a, b, c, d])
>>> E2, H2 = fields.eh_near(
...     1.0, 1.0, 1.5 + 0.0j, 1.0, r, theta, phi, abcd=abcd
... )

Conventions
-----------
- Phasor time dependence: ``exp(-i * omega * t)``.
- Incident plane wave: propagation along ``+z``, ``E`` along ``+x``, ``H`` along ``+y``,
  incident amplitude ``E0 = 1``.
- Spherical coordinates: ``theta`` from ``+z`` and ``phi`` from ``+x`` toward ``+y``.
- ``lambda0`` is vacuum wavelength.
- ``n_env`` is the surrounding-medium index and ``m_sphere`` is the sphere index;
  relative index is ``m_rel = m_sphere / n_env``.
- Size parameter is ``x = pi * d_sphere * n_env / lambda0``.
- Near field definition:
  outside the sphere, returned field is incident plus scattered;
  inside the sphere, returned field is the internal field.
- Non-magnetic materials are assumed (relative permeability ``mu_r = 1``).
"""

import numpy as np
import miepython as mie
from scipy.special import spherical_jn, factorial2
from miepython.bessel import spherical_h1, d_riccati_bessel_h1
from miepython.util import spherical_vector_to_cartesian

__all__ = (
    "e_near",
    "h_near",
    "eh_near",
    "e_near_cartesian",
    "h_near_cartesian",
    "eh_near_cartesian",
    "e_far",
)


def _sum_two_scaled_terms(scale, coeff1, values1, scale1, coeff2, values2, scale2):
    """Return ``sum(scale * (scale1*coeff1*values1 + scale2*coeff2*values2))``."""
    return np.sum(scale * (scale1 * coeff1 * values1 + scale2 * coeff2 * values2))


def _vsh_components_base(n_terms, lambda0, d_sphere, m_index, r, theta):
    """Compute shared VSH base components for one spatial point.

    Args:
        n_terms (int): Number of multipole terms.
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_index (complex): Refractive index at the evaluation point.
        r (float): Radial coordinate.
        theta (float): Polar angle in radians.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
            ``(M_theta_base, M_phi_base, N_r_base, N_theta_base, N_phi_base)``
            for multipole orders ``1..n_terms``.
    """
    mu = float(np.cos(theta))
    if mu >= 1.0:
        mu = 0.999999
    elif mu <= -1.0:
        mu = -0.999999

    pi = np.empty(n_terms)
    tau = np.empty(n_terms)
    mie._pi_tau(mu, pi, tau)

    n_int = np.arange(1, n_terms + 1, dtype=np.int64)
    n_arr = n_int.astype(np.float64)
    rho = 2 * np.pi * m_index * r / lambda0
    kr = 2 * np.pi * r / lambda0
    inside = r < d_sphere / 2

    if inside:
        jn = spherical_jn(n_int, rho)
        m_factor = jn

        if np.abs(rho) < 0.01:
            rho_pow = rho ** np.arange(0, n_terms)
            denom = factorial2(2 * n_int + 1)
            n_factor1 = rho_pow / denom
            n_factor2 = (n_arr + 1.0) * rho_pow / denom
        else:
            d_vals = mie._D_calc(np.complex128(m_index), float(kr), n_terms + 1)[:n_terms]
            n_factor1 = jn / rho
            n_factor2 = jn * d_vals
    else:
        h1 = spherical_h1(n_int, rho)
        m_factor = h1
        n_factor1 = h1 / rho
        n_factor2 = d_riccati_bessel_h1(n_int, rho) / rho

    sin_theta = np.sin(theta)
    M_theta_base = pi * m_factor
    M_phi_base = tau * m_factor
    N_r_base = n_arr * (n_arr + 1.0) * sin_theta * pi * n_factor1
    N_theta_base = tau * n_factor2
    N_phi_base = pi * n_factor2
    return M_theta_base, M_phi_base, N_r_base, N_theta_base, N_phi_base


def e_far(lambda0, d_sphere, m_sphere, n_env, r, theta, phi):
    """Evaluate the scattered electric far field.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float or ndarray): Radial observation distance.
        theta (float or ndarray): Polar angle in radians.
        phi (float or ndarray): Azimuth angle in radians.

    Returns:
        ndarray: Complex spherical components ``[E_r, E_theta, E_phi]`` with
            shape ``(3, ...)`` following broadcasted input shape.
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


def _incident_e_spherical(lambda0, n_env, r, theta, phi, amplitude=1.0):
    """Return incident electric plane-wave components in spherical coordinates.

    Args:
        lambda0 (float): Vacuum wavelength.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial coordinate.
        theta (float): Polar angle in radians.
        phi (float): Azimuth angle in radians.
        amplitude (complex): Incident electric field amplitude.

    Returns:
        ndarray: Complex spherical vector ``[E_r, E_theta, E_phi]``.
    """
    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * r * np.cos(theta))
    amp = amplitude * phase

    e_r = amp * np.sin(theta) * np.cos(phi)
    e_theta = amp * np.cos(theta) * np.cos(phi)
    e_phi = -amp * np.sin(phi)
    return np.array([e_r, e_theta, e_phi])


def _incident_h_spherical(lambda0, n_env, r, theta, phi, amplitude=1.0):
    """Return incident magnetic plane-wave components in spherical coordinates.

    Args:
        lambda0 (float): Vacuum wavelength.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial coordinate.
        theta (float): Polar angle in radians.
        phi (float): Azimuth angle in radians.
        amplitude (complex): Incident magnetic field amplitude in normalized units.

    Returns:
        ndarray: Complex spherical vector ``[H_r, H_theta, H_phi]``.
    """
    k = 2 * np.pi * n_env / lambda0
    phase = np.exp(1j * k * r * np.cos(theta))
    amp = amplitude * phase

    h_r = amp * np.sin(theta) * np.sin(phi)
    h_theta = amp * np.cos(theta) * np.sin(phi)
    h_phi = amp * np.cos(phi)
    return np.array([h_r, h_theta, h_phi])


def _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole):
    """Compute Mie coefficients with consistent medium scaling.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.

    Returns:
        ndarray: Coefficients packed as ``[a, b, c, d]``.
    """
    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    a, b, c, d = mie.coefficients(m_rel, x, n_pole=n_pole, internal=True)
    return np.array([a, b, c, d])


def _vectorized_field_eval(evaluator, r, theta, phi):
    """Evaluate a spherical-field callback on scalar or broadcasted inputs.

    Args:
        evaluator (callable): Callable that accepts scalar ``(r, theta, phi)``
            and returns a complex 3-vector.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.

    Returns:
        ndarray: Complex array with shape ``(3, ...)``.
    """
    rr, tt, pp = np.broadcast_arrays(np.asarray(r), np.asarray(theta), np.asarray(phi))

    if rr.ndim == 0:
        return evaluator(float(rr), float(tt), float(pp))

    out = np.empty((3,) + rr.shape, dtype=complex)
    for idx in np.ndindex(rr.shape):
        out[(slice(None),) + idx] = evaluator(float(rr[idx]), float(tt[idx]), float(pp[idx]))
    return out


def _vectorized_field_pair_eval(evaluator, r, theta, phi):
    """Evaluate an ``(E, H)`` callback on scalar or broadcasted inputs.

    Args:
        evaluator (callable): Callable that accepts scalar ``(r, theta, phi)``
            and returns a tuple ``(E, H)``, each a complex 3-vector.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(E, H)`` with each array shaped
            ``(3, ...)``.
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
    """Convert Cartesian coordinates to spherical coordinates safely.

    Args:
        x (float or ndarray): Cartesian x coordinate(s).
        y (float or ndarray): Cartesian y coordinate(s).
        z (float or ndarray): Cartesian z coordinate(s).

    Returns:
        tuple[ndarray, ndarray, ndarray]: ``(r, theta, phi)`` arrays after
            broadcasting the inputs.
    """
    xx, yy, zz = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(z))
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(rr == 0, 1.0, np.clip(zz / rr, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    phi = np.arctan2(yy, xx)
    return rr, theta, phi


def _spherical_components_to_cartesian(field_sph, r, theta, phi):
    """Convert spherical vector components to Cartesian components.

    Args:
        field_sph (ndarray): Spherical components ``[F_r, F_theta, F_phi]``.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.

    Returns:
        ndarray: Cartesian components ``[F_x, F_y, F_z]``.
    """
    fx, fy, fz = spherical_vector_to_cartesian(field_sph[0], field_sph[1], field_sph[2], r, theta, phi)
    return np.array([fx, fy, fz])


def _e_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident):
    """Evaluate near-field electric components with precomputed coefficients.

    Args:
        abcd (ndarray): Mie coefficients ``[a, b, c, d]``.
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial coordinate.
        theta (float): Polar angle in radians.
        phi (float): Azimuth angle in radians.
        include_incident (bool): Include incident field outside the sphere.

    Returns:
        ndarray: Spherical electric components ``[E_r, E_theta, E_phi]``.
    """
    a, b, c, d = abcd

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2
    # miepython coefficients follow the n-ik convention and are conjugated
    # internally; use conjugated sphere index so internal fields are consistent.
    m_index = np.conjugate(m_sphere) if inside else n_env

    M_the_base, M_phi_base, N_r_base, N_the_base, N_phi_base = _vsh_components_base(
        N, lambda0, d_sphere, m_index, r, theta
    )
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    M_rad = np.zeros(N, dtype=np.complex128)
    M_the = cos_phi * M_the_base
    M_phi = -sin_phi * M_phi_base
    N_rad = cos_phi * N_r_base
    N_the = cos_phi * N_the_base
    N_phi = -sin_phi * N_phi_base

    if inside:
        E_rad = _sum_two_scaled_terms(scale, c, M_rad, 1.0 + 0.0j, d, N_rad, -1.0j)
        E_the = _sum_two_scaled_terms(scale, c, M_the, 1.0 + 0.0j, d, N_the, -1.0j)
        E_phi = _sum_two_scaled_terms(scale, c, M_phi, 1.0 + 0.0j, d, N_phi, -1.0j)
    else:
        E_rad = _sum_two_scaled_terms(scale, a, N_rad, 1.0j, b, M_rad, -1.0 + 0.0j)
        E_the = _sum_two_scaled_terms(scale, a, N_the, 1.0j, b, M_the, -1.0 + 0.0j)
        E_phi = _sum_two_scaled_terms(scale, a, N_phi, 1.0j, b, M_phi, -1.0 + 0.0j)

        if include_incident:
            Ei_rad, Ei_the, Ei_phi = _incident_e_spherical(lambda0, n_env, r, theta, phi)
            E_rad += Ei_rad
            E_the += Ei_the
            E_phi += Ei_phi

    return np.array([E_rad, E_the, -E_phi])


def _h_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident):
    """Evaluate near-field magnetic components with precomputed coefficients.

    Args:
        abcd (ndarray): Mie coefficients ``[a, b, c, d]``.
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float): Radial coordinate.
        theta (float): Polar angle in radians.
        phi (float): Azimuth angle in radians.
        include_incident (bool): Include incident field outside the sphere.

    Returns:
        ndarray: Spherical magnetic components ``[H_r, H_theta, H_phi]``.
    """
    a, b, c, d = abcd

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2
    m_index = np.conjugate(m_sphere) if inside else n_env

    M_the_base, M_phi_base, N_r_base, N_the_base, N_phi_base = _vsh_components_base(
        N, lambda0, d_sphere, m_index, r, theta
    )
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    M_rad = np.zeros(N, dtype=np.complex128)
    M_the = -sin_phi * M_the_base
    M_phi = -cos_phi * M_phi_base
    N_rad = sin_phi * N_r_base
    N_the = sin_phi * N_the_base
    N_phi = cos_phi * N_phi_base

    if inside:
        m_rel = np.conjugate(m_sphere / n_env)
        H_rad = m_rel * _sum_two_scaled_terms(scale, d, M_rad, -1.0 + 0.0j, c, N_rad, -1.0j)
        H_the = m_rel * _sum_two_scaled_terms(scale, d, M_the, -1.0 + 0.0j, c, N_the, -1.0j)
        H_phi = m_rel * _sum_two_scaled_terms(scale, d, M_phi, -1.0 + 0.0j, c, N_phi, -1.0j)
    else:
        H_rad = _sum_two_scaled_terms(scale, b, N_rad, 1.0j, a, M_rad, 1.0 + 0.0j)
        H_the = _sum_two_scaled_terms(scale, b, N_the, 1.0j, a, M_the, 1.0 + 0.0j)
        H_phi = _sum_two_scaled_terms(scale, b, N_phi, 1.0j, a, M_phi, 1.0 + 0.0j)

        if include_incident:
            Hi_rad, Hi_the, Hi_phi = _incident_h_spherical(lambda0, n_env, r, theta, phi)
            H_rad += Hi_rad
            H_the += Hi_the
            H_phi += Hi_phi

    return np.array([H_rad, H_the, -H_phi])


def _eh_near_abcd(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident):
    """Evaluate electric and magnetic near fields with shared VSH work."""
    a, b, c, d = abcd

    N = len(a)
    nn = np.arange(1, N + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    inside = r < d_sphere / 2
    m_index = np.conjugate(m_sphere) if inside else n_env
    M_the_base, M_phi_base, N_r_base, N_the_base, N_phi_base = _vsh_components_base(
        N, lambda0, d_sphere, m_index, r, theta
    )
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    M_odd_rad = np.zeros(N, dtype=np.complex128)
    M_odd_the = cos_phi * M_the_base
    M_odd_phi = -sin_phi * M_phi_base
    N_even_rad = cos_phi * N_r_base
    N_even_the = cos_phi * N_the_base
    N_even_phi = -sin_phi * N_phi_base

    M_even_rad = np.zeros(N, dtype=np.complex128)
    M_even_the = -sin_phi * M_the_base
    M_even_phi = -cos_phi * M_phi_base
    N_odd_rad = sin_phi * N_r_base
    N_odd_the = sin_phi * N_the_base
    N_odd_phi = cos_phi * N_phi_base

    if inside:
        e_rad = _sum_two_scaled_terms(scale, c, M_odd_rad, 1.0 + 0.0j, d, N_even_rad, -1.0j)
        e_the = _sum_two_scaled_terms(scale, c, M_odd_the, 1.0 + 0.0j, d, N_even_the, -1.0j)
        e_phi = _sum_two_scaled_terms(scale, c, M_odd_phi, 1.0 + 0.0j, d, N_even_phi, -1.0j)

        m_rel = np.conjugate(m_sphere / n_env)
        h_rad = m_rel * _sum_two_scaled_terms(scale, d, M_even_rad, -1.0 + 0.0j, c, N_odd_rad, -1.0j)
        h_the = m_rel * _sum_two_scaled_terms(scale, d, M_even_the, -1.0 + 0.0j, c, N_odd_the, -1.0j)
        h_phi = m_rel * _sum_two_scaled_terms(scale, d, M_even_phi, -1.0 + 0.0j, c, N_odd_phi, -1.0j)
    else:
        e_rad = _sum_two_scaled_terms(scale, a, N_even_rad, 1.0j, b, M_odd_rad, -1.0 + 0.0j)
        e_the = _sum_two_scaled_terms(scale, a, N_even_the, 1.0j, b, M_odd_the, -1.0 + 0.0j)
        e_phi = _sum_two_scaled_terms(scale, a, N_even_phi, 1.0j, b, M_odd_phi, -1.0 + 0.0j)
        h_rad = _sum_two_scaled_terms(scale, b, N_odd_rad, 1.0j, a, M_even_rad, 1.0 + 0.0j)
        h_the = _sum_two_scaled_terms(scale, b, N_odd_the, 1.0j, a, M_even_the, 1.0 + 0.0j)
        h_phi = _sum_two_scaled_terms(scale, b, N_odd_phi, 1.0j, a, M_even_phi, 1.0 + 0.0j)

        if include_incident:
            e_i = _incident_e_spherical(lambda0, n_env, r, theta, phi)
            h_i = _incident_h_spherical(lambda0, n_env, r, theta, phi)
            e_rad += e_i[0]
            e_the += e_i[1]
            e_phi += e_i[2]
            h_rad += h_i[0]
            h_the += h_i[1]
            h_phi += h_i[2]

    return np.array([e_rad, e_the, -e_phi]), np.array([h_rad, h_the, -h_phi])


def e_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True, n_pole=0, abcd=None):
    """Calculate the electric field in and around a sphere.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        ndarray: Spherical electric components ``[E_r, E_theta, E_phi]`` with
            shape ``(3, ...)``.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    evaluator = lambda rr, tt, pp: _e_near_abcd(  # noqa: E731
        abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident
    )
    return _vectorized_field_eval(evaluator, r, theta, phi)


def h_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident=True, n_pole=0, abcd=None):
    """Calculate the magnetic field in and around a sphere.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        ndarray: Spherical magnetic components ``[H_r, H_theta, H_phi]`` with
            shape ``(3, ...)``.
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
    """Calculate electric and magnetic fields in and around a sphere.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(E, H)`` in spherical components,
            each with shape ``(3, ...)``.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    evaluator = lambda rr, tt, pp: _eh_near_abcd(  # noqa: E731
        abcd, lambda0, d_sphere, m_sphere, n_env, rr, tt, pp, include_incident
    )
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
    """Calculate electric near field in Cartesian coordinates.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        x (float or ndarray): Cartesian x coordinate(s).
        y (float or ndarray): Cartesian y coordinate(s).
        z (float or ndarray): Cartesian z coordinate(s).
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        ndarray: Cartesian electric components ``[E_x, E_y, E_z]``.
    """
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
    """Calculate magnetic near field in Cartesian coordinates.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        x (float or ndarray): Cartesian x coordinate(s).
        y (float or ndarray): Cartesian y coordinate(s).
        z (float or ndarray): Cartesian z coordinate(s).
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        ndarray: Cartesian magnetic components ``[H_x, H_y, H_z]``.
    """
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
    """Calculate electric and magnetic near fields in Cartesian coordinates.

    Args:
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        x (float or ndarray): Cartesian x coordinate(s).
        y (float or ndarray): Cartesian y coordinate(s).
        z (float or ndarray): Cartesian z coordinate(s).
        include_incident (bool): Include incident field for points outside sphere.
        n_pole (int): Requested multipole order. ``0`` means automatic truncation.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(E_xyz, H_xyz)`` where each array is
            ``[x, y, z]`` components.
    """
    r, theta, phi = _cartesian_to_spherical_safe(x, y, z)
    e_sph, h_sph = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, n_pole, abcd)
    e_xyz = _spherical_components_to_cartesian(e_sph, r, theta, phi)
    h_xyz = _spherical_components_to_cartesian(h_sph, r, theta, phi)
    return e_xyz, h_xyz
