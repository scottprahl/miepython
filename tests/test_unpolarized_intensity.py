"""Tests for unpolarized scattering intensity consistency."""

import numpy as np
import miepython as mie


def test_i_unpolarized_matches_intensities():
    """Two independent methods should return the same unpolarized intensities."""
    medium_index = 1.34
    index_of_refraction = 1.4
    radius_m = 1e-7
    wavelength_m = 8.4e-7

    angles_deg = np.linspace(-180, 180, 1800)
    mu = np.cos(np.deg2rad(angles_deg))

    i_par, i_per = mie.intensities(index_of_refraction, 2 * radius_m, wavelength_m, mu, n_env=medium_index)
    i_un1 = (i_par + i_per) / 2

    m = index_of_refraction / medium_index
    size = 2 * np.pi * radius_m * medium_index / wavelength_m
    i_un2 = mie.i_unpolarized(m, size, mu)

    assert np.allclose(i_un1, i_un2, rtol=1e-8, atol=1e-8)
