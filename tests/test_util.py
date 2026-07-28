"""Tests for the coordinate and formatting helpers in miepython.util."""

import os

import numpy as np
import pytest

import miepython
from miepython import util

# tests/ is on sys.path during a pytest run, so the guard below can inspect conftest
import conftest  # pylint: disable=wrong-import-order


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


class TestBackendGuard:
    """Test that the active backend matches what the test files require.

    These live here, in a file that belongs to neither backend, so they run in
    every pytest process.
    """

    def test_backend_matches_environment(self):
        """The package honoured MIEPYTHON_USE_JIT for this process."""
        want = os.environ.get("MIEPYTHON_USE_JIT", "0") == "1"
        assert miepython.USE_JIT is want, (
            "MIEPYTHON_USE_JIT=%r but miepython.USE_JIT=%r; something imported "
            "miepython before the variable was set" % (os.environ.get("MIEPYTHON_USE_JIT"), miepython.USE_JIT)
        )

    def test_kernels_come_from_the_expected_module(self):
        """The bound kernels really are the numba or the python ones."""
        kind = type(miepython.single_sphere).__name__
        # NUMBA_DISABLE_JIT=1, which `make coverage` uses so coverage.py can see
        # inside the njit bodies, leaves them as plain functions
        disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"
        want = "CPUDispatcher" if (miepython.USE_JIT and not disabled) else "function"
        assert kind == want, kind

    def test_conftest_filters_the_other_backend(self):
        """Filtering must stay in place, or half the suite tests the wrong code."""
        assert conftest.USE_JIT is miepython.USE_JIT
        assert conftest.JIT_PREFIX == "test_jit"
        assert conftest.NOJIT_PREFIX == "test_nojit"
        assert callable(conftest.pytest_collection_modifyitems)


class TestComplexFormatting:
    """Test the complex-number formatting helpers.

    These are exported but currently have no callers inside the package, so
    nothing else would notice if they broke.
    """

    def test_cs_scalar_sign_follows_the_imaginary_part(self):
        """The sign is carried by the separator, and the magnitude printed bare."""
        assert util.cs_scalar(1.5 + 0.1j) == "( 1.50000 + 0.10000j)"
        assert util.cs_scalar(1.5 - 0.1j) == "( 1.50000 - 0.10000j)"

    def test_cs_scalar_digit_count(self):
        """N sets the number of decimals."""
        assert util.cs_scalar(1.5 + 0.25j, 2) == "( 1.50 + 0.25j)"
        assert util.cs_scalar(1.5 + 0.25j, 1) == "( 1.5 + 0.2j)"

    def test_cs_scalar_negative_n_switches_to_exponential(self):
        """A negative N asks for exponential notation with abs(N) digits."""
        assert util.cs_scalar(1.5e-8 - 2e-9j, -3) == "( 1.500e-08 - 2.000e-09j)"
        # the sign branch is duplicated inside the exponential form, so check both
        assert util.cs_scalar(1.5e-8 + 2e-9j, -3) == "( 1.500e-08 + 2.000e-09j)"

    def test_cs_scalar_without_parentheses(self):
        """include_parens=False is what cs_vector uses to build its own."""
        assert util.cs_scalar(1.5 - 0.1j, 5, include_parens=False) == " 1.50000 - 0.10000j"

    def test_cs_vector_joins_with_one_pair_of_parentheses(self):
        """A tuple becomes a single parenthesised, comma-separated list."""
        text = util.cs_vector((1 + 1j, 2 - 2j))
        assert text.startswith("(") and text.endswith(")")
        assert text.count("(") == 1 and text.count(")") == 1
        assert text.count(",") == 1
        assert "+ 1.00000j" in text and "- 2.00000j" in text

    def test_cs_dispatches_on_scalar_or_sequence(self):
        """Both a lone number and an iterable of them are handled."""
        assert util.cs(1 + 1j) == "( 1.00000 + 1.00000j)"
        joined = util.cs([1 + 1j, 2 - 2j])
        assert joined == "( 1.00000 + 1.00000j), ( 2.00000 - 2.00000j)"
        assert not joined.endswith(", ")

    def test_phasor_of_a_complex_number(self):
        """A complex value prints as magnitude and angle in degrees."""
        text = util.phasor_str_scalar(1 + 1j)
        assert "∠" in text
        magnitude, angle = text.split("∠")
        assert float(magnitude) == pytest.approx(np.sqrt(2), abs=5e-3)
        assert float(angle.rstrip("°")) == pytest.approx(45.0, abs=5e-3)

    def test_phasor_of_a_positive_real_number(self):
        """A real value is special cased to avoid a meaningless angle."""
        assert util.phasor_str_scalar(2.0).split("∠")[1].strip() == "0°"

    def test_phasor_negative_n_switches_to_exponential(self):
        """A negative N asks for exponential notation."""
        text = util.phasor_str_scalar(1 + 1j, -3)
        assert "e+00" in text
        magnitude, angle = text.split("∠")
        assert float(magnitude) == pytest.approx(np.sqrt(2), rel=1e-3)
        assert float(angle.rstrip("°")) == pytest.approx(45.0, rel=1e-3)

    def test_phasor_str_dispatches_on_scalar_or_sequence(self):
        """phasor_str handles a lone number and an iterable of them."""
        single = util.phasor_str(1 + 1j)
        assert single == util.phasor_str_scalar(1 + 1j)
        joined = util.phasor_str([1 + 1j, 2 - 2j])
        assert joined.count("∠") == 2
        assert not joined.endswith(", ")

    def test_spherical_vector_round_trip(self):
        """A vector converted to Cartesian keeps its length."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            r, theta, phi = rng.uniform(0.2, 2), rng.uniform(0.1, np.pi - 0.1), rng.uniform(0, 2 * np.pi)
            v = rng.normal(size=3)
            xyz = util.spherical_vector_to_cartesian(v[0], v[1], v[2], r, theta, phi)
            assert np.linalg.norm(xyz) == pytest.approx(np.linalg.norm(v))
