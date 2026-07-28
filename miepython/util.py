"""Functions to format complex numbers or arrays of complex numbers."""

import numpy as np

__all__ = (
    "cs_scalar",
    "cs_vector",
    "cs",
    "phasor_str_scalar",
    "phasor_str",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "spherical_vector_to_cartesian",
)


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical (r, theta, phi).

    Scalars and arrays are both accepted and the three inputs are broadcast
    against each other.  At the origin theta is reported as 0, and z/r is clipped
    so that rounding cannot push the arccos argument outside [-1, 1].

    Args:
        x: Cartesian x coordinate, scalar or array-like
        y: Cartesian y coordinate, scalar or array-like
        z: Cartesian z coordinate, scalar or array-like

    Returns:
        r, theta, phi: radius, polar angle from +z, azimuth from +x toward +y
    """
    xx, yy, zz = np.broadcast_arrays(np.asarray(x), np.asarray(y), np.asarray(z))
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(r == 0, 1.0, np.clip(zz / r, -1.0, 1.0))

    theta = np.arccos(cos_theta)  # polar angle (0 to pi)
    phi = np.arctan2(yy, xx)  # azimuthal angle (-pi to pi)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def spherical_vector_to_cartesian(E_r, E_theta, E_phi, _r, theta, phi):
    """Convert spherical components (E_r, E_theta, E_phi) to Cartesian (Ex, Ey, Ez)."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Ex = E_r * sin_theta * cos_phi + E_theta * cos_theta * cos_phi - E_phi * sin_phi
    Ey = E_r * sin_theta * sin_phi + E_theta * cos_theta * sin_phi + E_phi * cos_phi
    Ez = E_r * cos_theta - E_theta * sin_theta

    return Ex, Ey, Ez


def cs_scalar(z, N=5, include_parens=True):
    """Convert complex number to string for printing."""
    if N > 0:
        if z.imag < 0:
            form = "%% .%df - %%.%dfj" % (N, N)
        else:
            form = "%% .%df + %%.%dfj" % (N, N)
    else:
        N = abs(N)
        if z.imag < 0:
            form = "%% .%de - %%.%dej" % (N, N)
        else:
            form = "%% .%de + %%.%dej" % (N, N)

    if include_parens:
        form = "(" + form + ")"
    return form % (z.real, abs(z.imag))


def cs_vector(z, N=5):
    """Convert a tuple to string for printing."""
    s = "("
    for zz in z:
        if s != "(":
            s += ", "
        s += cs_scalar(zz, N, include_parens=False)
    s += ")"
    return s


def cs(z, N=5):
    """Convert complex numbers to string for printing."""
    if np.isscalar(z):
        return cs_scalar(z, N)

    s = ""
    for zz in z:
        s += cs_scalar(zz, N)
        s += ", "

    return s[:-2]


def phasor_str_scalar(z, N=2):
    """Convert complex scalar to phasors for printing."""
    if N > 0:
        form = "%% %d.%df" % (N + 4, N)
    else:
        N = abs(N)
        form = "%%.%de" % (N)

    if np.isreal(z):
        return form % z.real + " ∠ 0°"
    form = form + " ∠ " + form + "°"
    return form % (np.abs(z), np.angle(z, deg=True))


def phasor_str(z, N=2):
    """Convert complex numbers to phasor for printing."""
    if np.isscalar(z):
        return phasor_str_scalar(z, N)

    s = ""
    for zz in z:
        s += phasor_str_scalar(zz, N)
        s += ", "

    return s[:-2]
