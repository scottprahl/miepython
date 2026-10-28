"""
Sample random scattering angles from the Mie phase function.

A Monte Carlo simulation of light transport needs, at every scattering event, a
random direction in which the photon leaves the sphere. For unpolarized light
only the polar angle theta depends on the sphere, and it must be drawn so that
its cosine mu follows the phase function. This module does that by inverse
transform sampling: build the cumulative distribution of mu once, then map each
uniform random number u to the cosine at which the distribution reaches u.

``cdf``
    The cumulative distribution at evenly spaced cosines.
``mu_with_uniform_cdf``
    The cosines at which the cumulative distribution reaches evenly spaced
    probabilities, a lookup table that makes each draw cheap.
``generate_mie_costheta``
    One random cosine drawn from that table.

``docs/06_random_deviates.ipynb`` works through the whole procedure.

This module is *not* imported with the package, because it imports
``miepython`` itself and doing so from ``miepython/__init__.py`` would be
circular. Import it by name, as the examples below do.

Examples:
    Invert the distribution with ``np.interp`` to draw many cosines at once.
    Their average is the asymmetry parameter g:

    >>> import numpy as np
    >>> import miepython as mie
    >>> import miepython.monte_carlo as mc
    >>> m, x = 1.33, 2.0                          # a water droplet
    >>> mu, cdf = mc.cdf(m, x, 1000)
    >>> rng = np.random.default_rng(42)
    >>> cosines = np.interp(rng.random(100_000), cdf, mu)
    >>> print(f"{np.mean(cosines):.2f}")
    0.67
    >>> qext, qsca, qback, g = mie.efficiencies_mx(m, x)
    >>> print(f"{g:.2f}")
    0.67

    Or build a lookup table once and draw one cosine at a time, as a photon
    transport loop does:

    >>> mu_table, _ = mc.mu_with_uniform_cdf(m, x, 1000)
    >>> cosines = [mc.generate_mie_costheta(mu_table, rng=rng) for _ in range(100_000)]
    >>> print(f"{np.mean(cosines):.2f}")
    0.67

Building either table costs a full Mie calculation, while each draw costs only a
random number or two, so build it once and reuse it. Evenly spaced cosines
resolve the narrow forward peak of a large sphere poorly; ``cdf`` explains how
many are needed and warns when there are too few, while ``mu_with_uniform_cdf``
chooses enough by itself.

Passing ``rng`` keeps the draws reproducible without disturbing the process-wide
``numpy.random``; omitting it falls back to that global generator.
"""

import warnings

import numpy as np
import miepython as mie

__all__ = (
    "cdf",
    "mu_with_uniform_cdf",
    "generate_mie_costheta",
)

# Largest acceptable relative error in the integral of the phase function
CDF_TOLERANCE = 0.01


def cdf(m, x, num):
    """
    Cumulative distribution of the unpolarized phase function in cos(theta).

    The cosines mu are evenly spaced from -1 (backscattering) to 1 (forward
    scattering), and cdf[i] is the probability that a scattered photon leaves
    with a cosine no greater than mu[i]. It is found by integrating the phase
    function with the trapezoidal rule, so cdf[0] is exactly 0 and the values
    never decrease.

    A large sphere scatters much of its light into a diffraction peak about
    4/x radians wide around the forward direction. Near theta = 0, mu = cos(theta)
    is close to 1 - theta**2/2, so that peak spans only about 8/x**2 in mu,
    while the cosines are 2/(num - 1) apart. To put several cosines inside the
    peak, num must therefore grow roughly as x**2; x = 30 needs about 5000.

    The phase function integrates to exactly one, so the raw trapezoidal sum
    shows how well num resolves it. If that sum is off by more than
    ``CDF_TOLERANCE``, a ``RuntimeWarning`` is issued; increase num until it
    stops. The result is then divided by the sum so that cdf[-1] is exactly 1.

    Args:
        m: complex index of refraction of the sphere
        x: size parameter of the sphere
        num: number of cosines, at least two

    Returns:
        mu: evenly spaced cosines of the scattering angle, from -1 to 1
        cdf: cumulative probability at each cosine, from 0 to 1

    Raises:
        ValueError: if num is less than two or the phase function is not finite
    """
    if num < 2:
        raise ValueError("num must be at least 2")
    return _trapezoid_cdf(m, x, num, stacklevel=3)


def _trapezoid_cdf(m, x, num, stacklevel):
    """Integrate the phase function at num cosines, warning at the caller's caller."""
    mu = np.linspace(-1, 1, num)
    intensity_per_mu = mie.i_unpolarized(m, x, mu, norm="4pi") / 2
    cdf_values = np.zeros(num)
    cdf_values[1:] = np.cumsum((intensity_per_mu[1:] + intensity_per_mu[:-1]) * np.diff(mu) / 2)
    total = cdf_values[-1]
    if not np.isfinite(total):
        raise ValueError(f"phase function is not finite for m={m}, x={x}")
    if abs(total - 1) > CDF_TOLERANCE:
        warnings.warn(
            f"phase function under-resolved for x={x}: integral over {num} angles is {total:.4g}, "
            "not 1, so sampled angles will be inaccurate; use more angles",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    cdf_values /= total
    return mu, cdf_values


def mu_with_uniform_cdf(m, x, num):
    """
    Inverse-transform table for sampling the unpolarized phase function.

    The table holds num cosines mu running from -1 (backscattering) to 1
    (forward scattering), spaced so that a scattered photon is equally likely
    to fall between any two neighbours: cdf[i] is exactly i/(num - 1). The
    cosines are therefore bunched where scattering is strong and sparse where
    it is weak. ``generate_mie_costheta`` draws random cosines from this table.

    The table is found by calculating ``cdf`` at max(2000, 5*x**2) evenly
    spaced cosines, enough to resolve the forward peak whatever num is (see
    ``cdf``), and interpolating to find where it reaches each step. Because
    that count grows as x**2, building the table takes seconds at x = 100 and
    over a minute at x = 300.

    Args:
        m: complex index of refraction of the sphere
        x: size parameter of the sphere
        num: number of entries in the table, at least two

    Returns:
        mu: unevenly spaced cosines of the scattering angle, from -1 to 1
        cdf: evenly spaced cumulative probabilities, from 0 to 1

    Raises:
        ValueError: if num is less than two or the phase function is not finite
    """
    if num < 2:
        raise ValueError("num must be at least 2")
    big_num = max(2000, int(5 * x**2))  # resolve the forward peak, see cdf
    big_mu, big_cdf = _trapezoid_cdf(m, x, big_num, stacklevel=3)
    cdf_values = np.arange(num) / (num - 1)
    mu = np.interp(cdf_values, big_cdf, big_mu)

    return [mu, cdf_values]


def generate_mie_costheta(mu_cdf, rng=None):
    """
    Generate a new scattering angle using a cdf.

    A uniformly spaced cumulative distribution function (CDF) is needed.
    New random angles are generated by selecting a random interval
    mu[i] to mu[i+1] and choosing an angle uniformly distributed over
    the interval.

    Args:
       mu_cdf: a cumulative distribution function
       rng: optional source of randomness, anything with a ``random()`` method
           such as ``numpy.random.default_rng()``.  Omitting it uses the process
           wide ``numpy.random``, whose stream a caller cannot control without
           reseeding the whole program, so pass a generator when reproducibility
           matters.

    Returns:
       The cosine of the scattering angle

    Examples:
        Draw a reproducible angle without disturbing the global stream:

        >>> import numpy as np
        >>> import miepython.monte_carlo as mc
        >>> mu, cdf = mc.mu_with_uniform_cdf(1.33, 1.7, 50)
        >>> rng = np.random.default_rng(42)
        >>> first = mc.generate_mie_costheta(mu, rng=rng)
        >>> bool(-1.0 <= first <= 1.0)
        True
        >>> bool(first == mc.generate_mie_costheta(mu, rng=np.random.default_rng(42)))
        True
    """
    if rng is None:
        rng = np.random

    # rng.random() draws from [0, 1), so int(r * num) never reaches num and
    # index + 1 stays inside the table.  No clamp is needed.
    num = len(mu_cdf) - 1
    index = int(rng.random() * num)

    x = mu_cdf[index]
    x += (mu_cdf[index + 1] - mu_cdf[index]) * rng.random()

    return x
