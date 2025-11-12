"""Rayleigh approximation for scattering by small particles."""

import numpy as np

__all__ = (
    "efficiencies_mx",
    "efficiencies",
    "S1_S2",
    "i_par",
    "i_per",
    "i_unpolarized",
    "intensities",
    "phase_matrix",
)


def efficiencies_mx(m, x):
    """
    Calculate the efficiencies for a small sphere.

    Based on equations 5.7 - 5.9 in Bohren and Huffman

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere

    Returns:
        qext: the total extinction efficiency
        qsca: the scattering efficiency
        qback: the backscatter efficiency
        g: the average cosine of the scattering phase function
    """
    ratio = (m**2 - 1) / (m**2 + 2)
    qsca = 8 / 3 * x**4 * abs(ratio) ** 2
    qext = 4 * x * ratio * (1 + x**2 / 15 * ratio * (m**4 + 27 * m**2 + 38) / (2 * m**2 + 3))
    qext = abs(qext.imag + qsca)
    qback = 4 * x**4 * abs(ratio) ** 2
    g = 0
    return qext, qsca, qback, g


def efficiencies(m, d, lambda0, n_env=1.0):
    """
    Calculate the efficiencies of a sphere using Rayleigh's approximation.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        n_env: real index of medium around sphere, optional.

    Returns:
        qext: the total extinction efficiency                  [-]
        qsca: the scattering efficiency                        [-]
        qback: the backscatter efficiency                      [-]
        g: the average cosine of the scattering phase function [-]
    """
    m_env = m / n_env
    x_env = np.pi * d / (lambda0 / n_env)
    return efficiencies_mx(m_env, x_env)


def normalization_factor(m, x, norm_str):
    """
    Figure out scattering function normalization.

    Args:
        m: complex index of refraction of sphere
        x: dimensionless sphere size
        norm_str: string describing type of normalization

    Returns:
        scaling factor needed for scattering function
    """
    factor = None
    norm = norm_str.lower()

    if norm in ["bohren"]:
        factor = 1 / 2

    elif norm in ["wiscombe"]:
        factor = 1

    elif norm in ["qsca", "scattering_efficiency"]:
        factor = x * np.sqrt(np.pi)

    else:
        qext, qsca, _, _ = efficiencies_mx(m, x)

        if norm in ["a", "albedo"]:
            factor = x * np.sqrt(np.pi * qext)

        if norm in ["1", "one", "unity"]:
            factor = x * np.sqrt(qsca * np.pi)

        if norm in ["four_pi", "4pi"]:
            factor = x * np.sqrt(qsca / 4)

        if norm in ["qext", "extinction_efficiency"]:
            factor = x * np.sqrt(qsca * np.pi / qext)

    if factor is None:
        raise ValueError(
            "normalization must be one of 'albedo' (default), 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe'"
        )
    return factor


def S1_S2(m, x, mu, norm="albedo"):
    """
    Calculate the scattering amplitude functions for small spheres.

    Based on equation 5.4 in Bohren and Huffman

    The amplitude functions are normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    The units are weird, sr**(-0.5)

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), to calculate scattering amplitudes
        norm: (optional) string describing scattering function normalization

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    if np.imag(m) > 0:  # ensure imaginary part of refractive index is negative
        m = np.conj(m)

    a1 = (2 * x**3) / 3 * (m**2 - 1) / (m**2 + 2) * 1j
    a1 += (2 * x**5) / 5 * (m**2 - 2) * (m**2 - 1) / (m**2 + 2) ** 2 * 1j

    S1 = (3 / 2) * a1 * np.ones_like(mu)
    S2 = (3 / 2) * a1 * mu

    normalization = normalization_factor(m, x, norm)

    S1 /= normalization
    S2 /= normalization

    return S1, S2


def i_per(m, x, mu, norm="albedo"):
    """
    Return the perpendicular scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, _ = S1_S2(m, x, mu, norm)
    intensity = np.abs(s1) ** 2
    return intensity.astype("float")


def i_par(m, x, mu, norm="albedo"):
    """
    Return the parallel scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    _, s2 = S1_S2(m, x, mu, norm)
    intensity = np.abs(s2) ** 2
    return intensity.astype("float")


def i_unpolarized(m, x, mu, norm="albedo"):
    """
    Return the unpolarized scattered intensity for small spheres.

    The default normalization sets the integral of the unpolarized
    intensity over 4pi steradians to equal the single scattering albedo.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter
        mu: the cos(theta) of each direction desired
        norm: (optional) string describing scattering function normalization

    Returns:
        The intensity at each angle in the array mu.  Units [1/sr]
    """
    s1, s2 = S1_S2(m, x, mu, norm)
    intensity = (abs(s1) ** 2 + abs(s2) ** 2) / 2
    return intensity.astype("float")


def intensities(m, d, lambda0, mu, n_env=1.0, norm="albedo"):
    """
    Return the scattered intensities from a sphere.

    These are the scattered intensities in a plane that is parallel (ipar) and
    perpendicular (iper) to the field of the incident plane wave.

    The scattered intensity is normalized such that the integral of the
    unpolarized intensity over 4𝜋 steradians is equal to the single scattering
    albedo.  The scattered intensity has units of inverse steradians [1/sr].

    The unpolarized scattering is the average of the two scattered intensities.

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere [-]
        d: the diameter of the sphere                    [same units as lambda0]
        lambda0: wavelength in a vacuum                  [same units as d]
        mu: the cos(theta) of each direction desired     [-]
        n_env: real index of medium around sphere, optional.
        norm: (optional) string describing scattering function normalization
        n_pole: return n_pole term from series (default=0 means include all terms)

    Returns:
        ipar, iper: scattered intensity in parallel and perpendicular planes [1/sr]
    """
    m_env = m / n_env
    lambda_env = lambda0 / n_env
    x_env = np.pi * d / lambda_env
    s1, s2 = S1_S2(m_env, x_env, mu, norm)
    ipar = np.abs(s2) ** 2
    iper = np.abs(s1) ** 2
    Ipar = ipar.astype("float")
    Iper = iper.astype("float")
    return Ipar, Iper


def phase_matrix(m, x, mu, norm="albedo"):
    """
    Calculate the scattering (Mueller) matrix.

    If mu has length N, then the returned matrix is 4x4xN.  If mu is a scalar
    then the matrix is 4x4

    The normalization is controlled by `norm` and should be one of
    ['albedo', 'one', '4pi', 'qext', 'qsca', 'bohren', or 'wiscombe']
    The normalization describes the integral of the scattering phase
    function over all 4𝜋 steradians.

    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        mu: the angles, cos(theta), of the phase scattering matrix
        n_pole: return n_pole term from series (default=0 means include all terms)
        norm: (optional) string describing scattering function normalization

    Returns:
        p: the phase scattering matrix [sr**(-1.0)]
    """
    mu = np.atleast_1d(mu)
    s1, s2 = S1_S2(m, x, mu, norm)

    s1_star = np.conjugate(s1)
    s2_star = np.conjugate(s2)
    m1 = (s1 * s1_star).real
    m2 = (s2 * s2_star).real
    s21 = (0.5 * (s1 * s2_star + s2 * s1_star)).real
    d21 = (-0.5j * (s1 * s2_star - s2 * s1_star)).real
    phase = np.zeros(shape=(4, 4, mu.size))
    phase[0, 0] = 0.5 * (m2 + m1)
    phase[0, 1] = 0.5 * (m2 - m1)
    phase[1, 0] = phase[0, 1]
    phase[1, 1] = phase[0, 0]
    phase[2, 2] = s21
    phase[2, 3] = -d21
    phase[3, 2] = d21
    phase[3, 3] = s21

    # squeeze returns a (4, 4) matrix rather than (4, 4, 1) when mu.size == 1
    return phase.squeeze()
