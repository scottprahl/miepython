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
from scipy.special import factorial2, spherical_jn
from ._backend import D_calc, pi_tau
from .bessel import spherical_h1
from .core import S1_S2, coefficients, wiscombe_terms
from .util import cartesian_to_spherical, spherical_vector_to_cartesian

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
    """Sum ``scale * (scale1*coeff1*values1 + scale2*coeff2*values2)`` over multipoles.

    ``values1`` and ``values2`` are shaped ``(n_points, n_terms)`` and the other
    arguments broadcast along the order axis, so the multipole series is summed away
    and one value per point comes back.
    """
    return np.sum(scale * (scale1 * coeff1 * values1 + scale2 * coeff2 * values2), axis=-1)


def _vsh_components_base(n_terms, lambda0, m_index, r, theta, inside):
    """Compute shared VSH base components for a batch of points.

    Every point in one call must lie on the same side of the sphere surface: a single
    ``inside`` flag chooses the radial function and a single ``m_index`` the medium
    for the whole batch.  ``_near_fields`` groups the points that way before calling.

    Args:
        n_terms (int): Number of multipole terms.
        lambda0 (float): Vacuum wavelength.
        m_index (complex): Refractive index at the evaluation points.
        r (ndarray): Radial coordinates, shape ``(n_points,)``.
        theta (ndarray): Polar angles in radians, shape ``(n_points,)``.
        inside (bool): True when the points lie inside the sphere.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
            ``(M_theta_base, M_phi_base, N_r_base, N_theta_base, N_phi_base)``, each
            shaped ``(n_points, n_terms)`` for multipole orders ``1..n_terms``.
    """
    mu = np.cos(theta)
    # the Legendre recurrence is 0/0 exactly at the poles, so nudge mu off them --
    # and only there, leaving every other value untouched
    mu = np.where(mu >= 1.0, 0.999999, mu)
    mu = np.where(mu <= -1.0, -0.999999, mu)

    n_points = r.size
    pi = np.empty((n_points, n_terms))
    tau = np.empty((n_points, n_terms))
    for k in range(n_points):
        # the kernel writes one point at a time; rows of a C-ordered array are the
        # contiguous 1-D buffers it expects
        pi_tau(float(mu[k]), pi[k], tau[k])

    n_int = np.arange(1, n_terms + 1, dtype=np.int64)
    n_arr = n_int.astype(np.float64)
    r_col = r[:, np.newaxis]
    rho = 2 * np.pi * m_index * r_col / lambda0
    kr = 2 * np.pi * r_col / lambda0

    if inside:
        jn = spherical_jn(n_int, rho)
        m_factor = jn

        n_factor1 = np.empty((n_points, n_terms), dtype=np.complex128)
        n_factor2 = np.empty((n_points, n_terms), dtype=np.complex128)

        # near the origin jn/rho and D_n both lose their meaning, so those points
        # take the leading-order series instead; the split is per point
        small = np.abs(rho[:, 0]) < 0.01
        if small.any():
            rho_pow = rho[small] ** np.arange(0, n_terms)
            denom = factorial2(2 * n_int + 1)
            n_factor1[small] = rho_pow / denom
            n_factor2[small] = (n_arr + 1.0) * rho_pow / denom

        large = ~small
        if large.any():
            # D_calc is a scalar kernel, so the logarithmic derivative is still one
            # call per point; only points inside the sphere need it at all
            d_vals = np.empty((int(np.count_nonzero(large)), n_terms), dtype=np.complex128)
            for k, kr_k in enumerate(kr[large, 0]):
                d_vals[k] = D_calc(np.complex128(m_index), float(kr_k), n_terms + 1)[:n_terms]
            n_factor1[large] = jn[large] / rho[large]
            n_factor2[large] = jn[large] * d_vals
    else:
        # xi'_n needs h1 at orders n-1, n and n+1, so evaluating orders 0..n_terms+1
        # in one call and slicing costs a quarter of what four separate calls do --
        # and h1 at order n is then shared instead of computed twice.
        h_all = spherical_h1(np.arange(0, n_terms + 2, dtype=np.int64), rho)
        h1 = h_all[:, 1 : n_terms + 1]
        m_factor = h1
        n_factor1 = h1 / rho
        # d_riccati_bessel_h1 written out on the cached values; identical arithmetic,
        # and tests/test_field.py pins it against that function so the two cannot drift
        d_xi = 0.5 * (rho * h_all[:, 0:n_terms] + h1 - rho * h_all[:, 2 : n_terms + 2])
        n_factor2 = d_xi / rho

    sin_theta = np.sin(theta)[:, np.newaxis]
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

    S1, S2 = S1_S2(m_rel, x, np.cos(theta), norm="wiscombe")

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
        n_pole (int): Number of terms to keep. ``0`` means automatic truncation.

    Returns:
        ndarray: Coefficients packed as ``[a, b, c, d]``.
    """
    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    if n_pole == 0:
        # Wiscombe's criterion truncates the scattered series, which converges
        # faster than the field evaluated right at the surface.  Now that an_bn
        # builds psi_n stably, extra orders converge instead of diverging: the
        # tangential boundary mismatch falls from 1.5e-5 at the criterion to
        # 3.6e-6 with one extra order and 3.2e-6 with two, then flattens out, so
        # two is where it stops being worth more terms.
        n_pole = wiscombe_terms(x) + 2
    a, b, c, d = coefficients(m_rel, x, n_pole=n_pole, internal=True)
    return np.array([a, b, c, d])


def _near_fields(abcd, lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, want_e, want_h):
    """Evaluate the near fields at every requested point.

    The points are split once into those inside the sphere and those outside, and each
    group is evaluated as a batch.  That grouping is what makes the work vectorizable:
    the two sides use different radial functions and different media, but within a
    side every point does the same arithmetic, so the multipole series becomes a sum
    over the last axis of a ``(n_points, n_terms)`` array.

    Args:
        abcd (ndarray): Mie coefficients ``[a, b, c, d]``.
        lambda0 (float): Vacuum wavelength.
        d_sphere (float): Sphere diameter.
        m_sphere (complex): Sphere refractive index.
        n_env (float): Refractive index of the surrounding medium.
        r (float or ndarray): Radial coordinate(s).
        theta (float or ndarray): Polar angle(s) in radians.
        phi (float or ndarray): Azimuth angle(s) in radians.
        include_incident (bool): Include incident field outside the sphere.
        want_e (bool): Compute the electric field.
        want_h (bool): Compute the magnetic field.

    Returns:
        tuple[ndarray or None, ndarray or None]: ``(E, H)`` in spherical components,
            each shaped ``(3,) + broadcast_shape`` or None when not requested.
    """
    a, b, c, d = abcd

    n_terms = len(a)
    nn = np.arange(1, n_terms + 1)
    scale = 1j**nn * (2 * nn + 1) / ((nn + 1) * nn)

    rr, tt, pp = np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
    )
    shape = rr.shape
    r_flat = rr.reshape(-1)
    theta_flat = tt.reshape(-1)
    phi_flat = pp.reshape(-1)

    e_out = np.empty((3, r_flat.size), dtype=complex) if want_e else None
    h_out = np.empty((3, r_flat.size), dtype=complex) if want_h else None

    is_inside = r_flat < d_sphere / 2
    for inside in (True, False):
        group = is_inside if inside else ~is_inside
        if not group.any():
            continue

        # miepython coefficients follow the n-ik convention and are conjugated
        # internally; use conjugated sphere index so internal fields are consistent.
        m_index = np.conjugate(m_sphere) if inside else n_env
        M_the_base, M_phi_base, N_r_base, N_the_base, N_phi_base = _vsh_components_base(
            n_terms, lambda0, m_index, r_flat[group], theta_flat[group], inside
        )

        cos_phi = np.cos(phi_flat[group])[:, np.newaxis]
        sin_phi = np.sin(phi_flat[group])[:, np.newaxis]
        zero = np.zeros_like(M_the_base, dtype=np.complex128)

        if want_e:
            # the electric field pairs the odd magnetic modes with the even electric ones
            M_odd_rad, M_odd_the, M_odd_phi = zero, cos_phi * M_the_base, -sin_phi * M_phi_base
            N_even_rad, N_even_the, N_even_phi = (
                cos_phi * N_r_base,
                cos_phi * N_the_base,
                -sin_phi * N_phi_base,
            )
            if inside:
                e_rad = _sum_two_scaled_terms(scale, c, M_odd_rad, 1.0 + 0.0j, d, N_even_rad, -1.0j)
                e_the = _sum_two_scaled_terms(scale, c, M_odd_the, 1.0 + 0.0j, d, N_even_the, -1.0j)
                e_phi = _sum_two_scaled_terms(scale, c, M_odd_phi, 1.0 + 0.0j, d, N_even_phi, -1.0j)
            else:
                e_rad = _sum_two_scaled_terms(scale, a, N_even_rad, 1.0j, b, M_odd_rad, -1.0 + 0.0j)
                e_the = _sum_two_scaled_terms(scale, a, N_even_the, 1.0j, b, M_odd_the, -1.0 + 0.0j)
                e_phi = _sum_two_scaled_terms(scale, a, N_even_phi, 1.0j, b, M_odd_phi, -1.0 + 0.0j)
                if include_incident:
                    e_i = _incident_e_spherical(lambda0, n_env, r_flat[group], theta_flat[group], phi_flat[group])
                    e_rad += e_i[0]
                    e_the += e_i[1]
                    e_phi += e_i[2]
            e_out[:, group] = np.array([e_rad, e_the, e_phi])

        if want_h:
            # and the magnetic field the other way round
            M_even_rad, M_even_the, M_even_phi = zero, -sin_phi * M_the_base, -cos_phi * M_phi_base
            N_odd_rad, N_odd_the, N_odd_phi = (
                sin_phi * N_r_base,
                sin_phi * N_the_base,
                cos_phi * N_phi_base,
            )
            if inside:
                m_rel = np.conjugate(m_sphere / n_env)
                h_rad = m_rel * _sum_two_scaled_terms(scale, d, M_even_rad, -1.0 + 0.0j, c, N_odd_rad, -1.0j)
                h_the = m_rel * _sum_two_scaled_terms(scale, d, M_even_the, -1.0 + 0.0j, c, N_odd_the, -1.0j)
                h_phi = m_rel * _sum_two_scaled_terms(scale, d, M_even_phi, -1.0 + 0.0j, c, N_odd_phi, -1.0j)
            else:
                h_rad = _sum_two_scaled_terms(scale, b, N_odd_rad, 1.0j, a, M_even_rad, 1.0 + 0.0j)
                h_the = _sum_two_scaled_terms(scale, b, N_odd_the, 1.0j, a, M_even_the, 1.0 + 0.0j)
                h_phi = _sum_two_scaled_terms(scale, b, N_odd_phi, 1.0j, a, M_even_phi, 1.0 + 0.0j)
                if include_incident:
                    h_i = _incident_h_spherical(lambda0, n_env, r_flat[group], theta_flat[group], phi_flat[group])
                    h_rad += h_i[0]
                    h_the += h_i[1]
                    h_phi += h_i[2]
            h_out[:, group] = np.array([h_rad, h_the, h_phi])

    e_final = e_out.reshape((3,) + shape) if want_e else None
    h_final = h_out.reshape((3,) + shape) if want_h else None
    return e_final, h_final


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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        ndarray: Spherical electric components ``[E_r, E_theta, E_phi]`` with
            shape ``(3, ...)``.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    return _near_fields(
        abcd,
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        r,
        theta,
        phi,
        include_incident,
        want_e=True,
        want_h=False,
    )[0]


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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        ndarray: Spherical magnetic components ``[H_r, H_theta, H_phi]`` with
            shape ``(3, ...)``.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    return _near_fields(
        abcd,
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        r,
        theta,
        phi,
        include_incident,
        want_e=False,
        want_h=True,
    )[1]


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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.
            If provided, ``n_pole`` is ignored.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(E, H)`` in spherical components,
            each with shape ``(3, ...)``.
    """
    if abcd is None:
        abcd = _coefficients_abcd(lambda0, d_sphere, m_sphere, n_env, n_pole)

    return _near_fields(
        abcd,
        lambda0,
        d_sphere,
        m_sphere,
        n_env,
        r,
        theta,
        phi,
        include_incident,
        want_e=True,
        want_h=True,
    )


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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        ndarray: Cartesian electric components ``[E_x, E_y, E_z]``.
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)
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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        ndarray: Cartesian magnetic components ``[H_x, H_y, H_z]``.
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)
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
        n_pole (int): Number of multipole terms to keep. ``0`` (the default) keeps
            two more than Wiscombe's criterion, which converges the near field at
            the sphere surface roughly five times closer than the criterion alone.
        abcd (ndarray or None): Optional precomputed coefficients ``[a, b, c, d]``.

    Returns:
        tuple[ndarray, ndarray]: Tuple ``(E_xyz, H_xyz)`` where each array is
            ``[x, y, z]`` components.
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)
    e_sph, h_sph = eh_near(lambda0, d_sphere, m_sphere, n_env, r, theta, phi, include_incident, n_pole, abcd)
    e_xyz = _spherical_components_to_cartesian(e_sph, r, theta, phi)
    h_xyz = _spherical_components_to_cartesian(h_sph, r, theta, phi)
    return e_xyz, h_xyz
