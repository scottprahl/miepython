"""Benchmark near-field runtime for the single field implementation."""

import os
from time import perf_counter

import numpy as np
import pytest
import miepython as mie
from miepython import field

RUN_BENCH = os.environ.get("MIEPYTHON_RUN_FIELD_SPEED", "0") == "1"


def _median_runtime(func, repeats=5):
    """Return median runtime and final function output."""
    timings = []
    last = None
    for _ in range(repeats):
        t0 = perf_counter()
        last = func()
        timings.append(perf_counter() - t0)
    return float(np.median(timings)), last


@pytest.mark.skipif(not RUN_BENCH, reason="Set MIEPYTHON_RUN_FIELD_SPEED=1 to run field speed benchmark")
def test_field_2d_eh_near_cartesian_speed():
    """Benchmark typical 2D near-field evaluation and print useful speed levers."""
    lambda0 = 0.6328
    d_sphere = 1.2
    m_sphere = 1.5 - 0.05j
    n_env = 1.33

    # Typical 2D slice through and around the sphere.
    axis = np.linspace(-1.2, 1.2, 101)
    x_grid, z_grid = np.meshgrid(axis, axis, indexing="ij")
    y_grid = np.zeros_like(x_grid)

    x = np.pi * d_sphere * n_env / lambda0
    m_rel = m_sphere / n_env
    abcd = np.array(mie.coefficients(m_rel, x, internal=True))

    def run_cached():
        return field.eh_near_cartesian(
            lambda0,
            d_sphere,
            m_sphere,
            n_env,
            x_grid,
            y_grid,
            z_grid,
            True,
            0,
            abcd,
        )

    def run_nocache():
        return field.eh_near_cartesian(
            lambda0,
            d_sphere,
            m_sphere,
            n_env,
            x_grid,
            y_grid,
            z_grid,
            True,
            0,
            None,
        )

    def run_split_cached():
        e_xyz = field.e_near_cartesian(
            lambda0,
            d_sphere,
            m_sphere,
            n_env,
            x_grid,
            y_grid,
            z_grid,
            True,
            0,
            abcd,
        )
        h_xyz = field.h_near_cartesian(
            lambda0,
            d_sphere,
            m_sphere,
            n_env,
            x_grid,
            y_grid,
            z_grid,
            True,
            0,
            abcd,
        )
        return e_xyz, h_xyz

    t_cached, out_cached = _median_runtime(run_cached, repeats=5)
    t_nocache, out_nocache = _median_runtime(run_nocache, repeats=3)
    t_split, out_split = _median_runtime(run_split_cached, repeats=5)

    e_cached, h_cached = out_cached
    e_nocache, h_nocache = out_nocache
    e_split, h_split = out_split

    np.testing.assert_allclose(e_cached, e_nocache, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_cached, h_nocache, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_cached, e_split, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_cached, h_split, rtol=1e-12, atol=1e-12)

    print(f"eh_near_cartesian cached abcd median: {t_cached:.4f} s")
    print(f"eh_near_cartesian internal abcd median: {t_nocache:.4f} s")
    print(f"e_near_cartesian + h_near_cartesian median: {t_split:.4f} s")
    print(f"coeff-cache speedup: {t_nocache / t_cached:.2f}x")
    print(f"combined-vs-split speedup: {t_split / t_cached:.2f}x")

    assert t_cached > 0.0
    assert t_nocache > 0.0
    assert t_split > 0.0
