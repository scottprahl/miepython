"""Tests for sampling scattering angles from the Mie phase function.

``miepython.monte_carlo`` builds an inverse-transform table -- angles spaced so
that the cumulative distribution is uniform -- and draws from it.  The workflow is
the one in ``docs/06_random_deviates.ipynb``.

``generate_mie_costheta`` uses the legacy global ``numpy.random``, so these tests
seed it and restore the previous state afterwards rather than leaving the process
RNG disturbed for whatever runs next.
"""

import contextlib

import numpy as np
import pytest

import miepython as mie
import miepython.monte_carlo as mc

# a water droplet at 550 nm, as the notebook uses
M_WATER = 1.33
X_DROPLET = 2 * np.pi * 0.15 / 0.550


@contextlib.contextmanager
def seeded(seed):
    """Seed the global RNG that generate_mie_costheta draws from, then restore it."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def draw(mu_table, count, seed):
    """Return `count` sampled cosines from an inverse-transform table."""
    with seeded(seed):
        return np.array([mc.generate_mie_costheta(mu_table) for _ in range(count)])


class TestCdf:
    """Test the cumulative distribution sampled uniformly in cos(theta)."""

    @pytest.mark.parametrize("num", [20, 50, 500])
    def test_shape_and_angles(self, num):
        """The angles span the full exit range at even spacing."""
        mu, cdf = mc.cdf(M_WATER, X_DROPLET, num)
        assert len(mu) == len(cdf) == num
        assert mu[0] == -1.0 and mu[-1] == 1.0
        np.testing.assert_allclose(mu, np.linspace(-1, 1, num))

    @pytest.mark.parametrize("num", [20, 50, 500])
    def test_is_non_decreasing_and_starts_near_zero(self, num):
        """A cumulative distribution may never run backwards."""
        _, cdf = mc.cdf(M_WATER, X_DROPLET, num)
        assert np.all(np.diff(cdf) >= 0)
        assert 0 < cdf[0] < 2.0 / num

    def test_endpoint_approaches_one_from_above(self):
        """The quadrature overshoots, and the error falls off like 1/num.

        ``cdf`` sums ``i_unpolarized / num`` rather than multiplying by the true
        spacing 2/(num-1), so the last value is not exactly 1: it is about
        1 + 1.3/num for this sphere.  The docstring claims a maximum of 1, so the
        overshoot is worth pinning rather than leaving as a surprise -- at the
        num=20 the notebook plots, the CDF reaches 1.067.
        """
        errors = {}
        for num in (50, 500, 5000):
            _, cdf = mc.cdf(M_WATER, X_DROPLET, num)
            errors[num] = cdf[-1] - 1.0
            assert errors[num] > 0, "overshoot expected"

        # ten times as many points, a tenth of the error
        assert errors[50] / errors[500] == pytest.approx(10.0, rel=0.1)
        assert errors[500] / errors[5000] == pytest.approx(10.0, rel=0.1)
        assert errors[5000] < 1e-3

    def test_more_forward_scattering_delays_the_rise(self):
        """A larger sphere throws light forward, so the cdf climbs later."""
        _, small = mc.cdf(M_WATER, 0.1, 200)
        _, large = mc.cdf(M_WATER, 6.0, 200)
        midpoint = 100  # mu = 0, halfway through the angles
        assert large[midpoint] < small[midpoint]


class TestUniformCdfTable:
    """Test the inverse-transform table used for sampling."""

    @pytest.mark.parametrize("num", [20, 200, 3000])
    def test_cdf_values_are_exactly_uniform(self, num):
        """That is the whole point: equal probability per interval."""
        _, cdf = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, num)
        assert cdf[0] == 0.0 and cdf[-1] == 1.0
        np.testing.assert_allclose(cdf, np.arange(num) / (num - 1))

    @pytest.mark.parametrize("num", [20, 200, 3000])
    def test_angles_are_sorted_and_span_the_range(self, num):
        """Inverting a monotone cdf has to give monotone angles."""
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, num)
        assert len(mu) == num
        assert mu[0] == -1.0 and mu[-1] == 1.0
        assert np.all(np.diff(mu) >= 0)

    def test_table_angles_bunch_where_scattering_is_strong(self):
        """Equal probability per interval means narrow intervals near the peak."""
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, 6.0, 200)
        spacing = np.diff(mu)
        # this sphere throws light forward, so the tightest intervals sit near mu=+1
        assert spacing[-1] < spacing[0]
        assert mu[np.argmin(spacing)] > 0.9
        # and the loosest are back in the weakly lit rearward hemisphere
        assert mu[np.argmax(spacing)] < 0.0
        assert np.max(spacing) / np.min(spacing) > 100


class TestSampling:
    """Test the random deviates drawn from the table."""

    def test_draws_stay_within_the_physical_range(self):
        """A cosine cannot leave [-1, 1]."""
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, 50)
        samples = draw(mu, 20000, seed=1)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_sampling_is_reproducible_for_a_given_seed(self):
        """The function reads the global RNG, so a seed must pin the sequence."""
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, 50)
        np.testing.assert_array_equal(draw(mu, 200, seed=11), draw(mu, 200, seed=11))
        assert not np.array_equal(draw(mu, 200, seed=11), draw(mu, 200, seed=12))

    def test_the_global_rng_is_left_undisturbed(self):
        """The helper here must restore the process RNG it borrows."""
        np.random.seed(99)
        expected = np.random.random(3)
        np.random.seed(99)
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, 20)
        draw(mu, 10, seed=5)
        np.testing.assert_array_equal(np.random.random(3), expected)

    @pytest.mark.parametrize("size_parameter", [0.1, 2.0, 6.0])
    def test_mean_cosine_reproduces_the_asymmetry_parameter(self, size_parameter):
        """The average of cos(theta) over the phase function *is* g.

        This is the strongest statement available about the sampler: it ties the
        drawn angles back to a quantity computed by a completely separate path.
        """
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, size_parameter, 500)
        samples = draw(mu, 200000, seed=7)
        g = mie.efficiencies_mx(M_WATER, size_parameter)[3]
        # the standard error on 200000 draws is a few times 1e-3
        assert samples.mean() == pytest.approx(g, abs=5e-3)

    def test_empirical_distribution_follows_the_table(self):
        """The fraction of draws below each tabulated angle must be its cdf value."""
        num = 25
        mu, cdf = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, num)
        samples = draw(mu, 400000, seed=3)
        empirical = np.array([np.mean(samples <= mu[k]) for k in range(num)])
        np.testing.assert_allclose(empirical, cdf, atol=0.01)

    def test_a_rayleigh_sphere_scatters_symmetrically(self):
        """A tiny sphere has g = 0, so the drawn cosines must average to zero."""
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, 0.05, 500)
        samples = draw(mu, 200000, seed=5)
        assert samples.mean() == pytest.approx(0.0, abs=5e-3)
        # and symmetrically means as much backward as forward
        assert np.mean(samples > 0) == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("num", [2, 20, 500])
    def test_the_largest_possible_draw_stays_inside_the_table(self, monkeypatch, num):
        """No index clamp is needed, and this is why.

        ``numpy.random.random`` draws from a half-open interval, so the largest
        value it can return is the float just below 1.  Even then ``int(r * num)``
        is num - 1, and ``index + 1`` is the last entry rather than one past it.
        """
        mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, num)
        largest = np.nextafter(1.0, 0.0)
        monkeypatch.setattr(np.random, "random", lambda: largest)

        value = mc.generate_mie_costheta(mu)  # would raise IndexError if it ran off
        assert value == pytest.approx(mu[-1], rel=1e-12)
        assert -1.0 <= value <= 1.0

    def test_no_draw_ever_leaves_the_table(self):
        """Hammer the sampler to confirm the half-open interval really holds."""
        for num in (2, 3, 50):
            mu, _ = mc.mu_with_uniform_cdf(M_WATER, X_DROPLET, num)
            samples = draw(mu, 20000, seed=num)
            assert samples.min() >= mu[0]
            assert samples.max() <= mu[-1]
