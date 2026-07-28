"""Tests for the coordinate and formatting helpers in miepython.util."""

import numpy as np
import pytest

from miepython import util


class TestExports:
    """Test the module's declared exports."""

    def test_all_is_spelled_correctly(self):
        """A misspelled __all__ silently exports everything."""
        assert hasattr(util, "__all__")

    def test_every_exported_name_exists(self):
        """Each name in __all__ must resolve."""
        for name in util.__all__:
            assert hasattr(util, name), name


class TestCartesianToSpherical:
    """Test the Cartesian to spherical conversion."""

    def test_scalars(self):
        """Scalar input returns the expected radius and angles."""
        r, theta, phi = util.cartesian_to_spherical(1.0, 0.0, 1.0)
        assert r == pytest.approx(np.sqrt(2.0))
        assert theta == pytest.approx(np.pi / 4)
        assert phi == pytest.approx(0.0)

    def test_arrays(self):
        """Array input works instead of raising on an ambiguous truth value."""
        x = np.array([1.0, 2.0, 0.0])
        y = np.zeros(3)
        z = np.ones(3)
        r, theta, phi = util.cartesian_to_spherical(x, y, z)
        np.testing.assert_allclose(r, np.sqrt(x**2 + z**2))
        np.testing.assert_allclose(theta, np.arctan2(np.hypot(x, y), z))
        np.testing.assert_allclose(phi, np.arctan2(y, x))

    def test_broadcasting(self):
        """The three coordinates are broadcast against each other."""
        r, theta, phi = util.cartesian_to_spherical(np.zeros((3, 4)), 0.0, 1.0)
        assert r.shape == theta.shape == phi.shape == (3, 4)

    def test_origin_is_finite(self):
        """The origin has no defined direction but must not produce nan."""
        r, theta, phi = util.cartesian_to_spherical(0.0, 0.0, 0.0)
        assert r == 0.0
        assert np.isfinite(theta) and np.isfinite(phi)
        assert theta == pytest.approx(0.0)

    def test_on_axis_is_finite(self):
        """Straight up and straight down are where clipping matters."""
        _, theta_up, _ = util.cartesian_to_spherical(0.0, 0.0, 2.0)
        _, theta_down, _ = util.cartesian_to_spherical(0.0, 0.0, -2.0)
        assert theta_up == pytest.approx(0.0)
        assert theta_down == pytest.approx(np.pi)

    def test_mixed_grid_has_no_nan(self):
        """A grid straddling the origin and both poles stays finite."""
        u = np.linspace(-1.0, 1.0, 7)
        X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
        r, theta, phi = util.cartesian_to_spherical(X, Y, Z)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(theta))
        assert np.all(np.isfinite(phi))
        assert np.all(theta >= 0) and np.all(theta <= np.pi)

    def test_round_trip(self):
        """Converting out and back reproduces the original points."""
        rng = np.random.default_rng(7)
        x, y, z = rng.normal(size=(3, 50))
        r, theta, phi = util.cartesian_to_spherical(x, y, z)
        xx, yy, zz = util.spherical_to_cartesian(r, theta, phi)
        np.testing.assert_allclose(xx, x, atol=1e-12)
        np.testing.assert_allclose(yy, y, atol=1e-12)
        np.testing.assert_allclose(zz, z, atol=1e-12)
