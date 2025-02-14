"""
Module for Bessel and Riccati-Bessel functions.

This module provides a collection of functions related to spherical Bessel functions,
Hankel functions, Riccati-Bessel functions, and their derivatives. These functions
are commonly used in physics, especially in problems involving wave propagation,
scattering, and spherical geometries.

Functions provided:

- Spherical Bessel functions:

  - `spherical_h1`: Spherical Hankel function of the first kind h_n^{(1)}(z).
  - `spherical_h2`: Spherical Hankel function of the second kind h_n^{(2)}(z).

- Riccati-Bessel functions:

  - `riccati_bessel_jn`: Riccati-Bessel function of the first kind ψ_n(z).
  - `riccati_bessel_h1`: Riccati-Bessel function of the third kind ξ_n(z).
  - `riccati_bessel_h2`: Riccati-Bessel function of the third kind ζ_n(z).

- Derivatives of Bessel functions:

  - `d_spherical_jn`: Derivative of the spherical Bessel function j_n'(z).
  - `d_spherical_h1`: Derivative of the spherical Hankel function of first kind h_n^{(1)}'(z).
  - `d_spherical_h2`: Derivative of the spherical Hankel function of second kind h_n^{(2)}'(z).

- Derivatives of Riccati-Bessel functions:

  - `d_riccati_bessel_jn`: Derivative of the Riccati-Bessel function of the first kind ψ_n'(z).
  - `d_riccati_bessel_h1`: Derivative of the Riccati-Bessel function of the third kind ξ_n'(z).
  - `d_riccati_bessel_h2`: Derivative of the Riccati-Bessel function of the third kind ζ_n'(z).
"""

__all__ = (
    "spherical_h1",
    "spherical_h2",
    "riccati_bessel_jn",
    "riccati_bessel_h1",
    "riccati_bessel_h2",
    "d_spherical_jn",
    "d_spherical_h1",
    "d_spherical_h2",
    "d_riccati_bessel_jn",
    "d_riccati_bessel_h1",
    "d_riccati_bessel_h2",
)

from scipy.special import spherical_jn, spherical_yn


def spherical_h1(n, z):
    """
    Spherical Hankel function of the first kind.
    """
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def spherical_h2(n, z):
    """
    Spherical Hankel function of the second kind.
    """
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)


def riccati_bessel_jn(n, z):
    """
    Riccati-Bessel function of the first kind ψ_n(z)=z * j_n(z).
    """
    return z * spherical_jn(n, z)


def riccati_bessel_h1(n, z):
    """
    Riccati-Bessel function of the third kind.

    This follows the Bohren and Huffman assumption that xi_n(z) = z * h_n^{(1)}(z)
    """
    return z * spherical_h1(n, z)


def riccati_bessel_h2(n, z):
    """
    Riccati-Bessel function of the third kind.

    This follows the van de Hulst or Kerker's assumption that xi_n(z) = z * h_n^{(2)}(z)
    """
    return z * spherical_h2(n, z)


def d_spherical_jn(n, z):
    """
    Derivative of the spherical Bessel function of the first kind.
    """
    return n * spherical_jn(n, z) / z - spherical_jn(n + 1, z)


def d_spherical_h1(n, z):
    """
    Derivative of the spherical Hankel function of the first kind.
    """
    return 0.5 * (spherical_h1(n - 1, z) - spherical_h1(n, z) / z - spherical_h1(n + 1, z))


def d_spherical_h2(n, z):
    """
    Derivative of the spherical Hankel function of the second kind.
    """
    return 0.5 * (spherical_h2(n - 1, z) - spherical_h2(n, z) / z - spherical_h2(n + 1, z))


def d_riccati_bessel_jn(n, z):
    """
    Derivative of the Riccati-Bessel function of the first kind ψ_n'(z).
    """
    return (n + 1) * spherical_jn(n, z) - z * spherical_jn(n + 1, z)


def d_riccati_bessel_h1(n, z):
    """
    Derivative of the Riccati-Bessel function of the third kind.

    This follows the Bohren and Huffman assumption that xi_n(z) = z * h_n^{(1)}(z)
    """
    return 0.5 * (z * spherical_h1(n - 1, z) + spherical_h1(n, z) - z * spherical_h1(n + 1, z))


def d_riccati_bessel_h2(n, z):
    """
    Derivative of the Riccati-Bessel function of the third kind.

    This follows the van de Hulst or Kerker's assumption that xi_n(z) = z * h_n^{(2)}(z)
    """
    return 0.5 * (z * spherical_h2(n - 1, z) + spherical_h2(n, z) - z * spherical_h2(n + 1, z))
