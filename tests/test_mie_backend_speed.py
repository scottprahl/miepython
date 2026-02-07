"""Benchmark and compare mie_nojit and mie_jit backend performance."""

import os
from time import perf_counter

import numpy as np
import pytest

from miepython import mie_jit, mie_nojit

RUN_BENCH = os.environ.get("MIEPYTHON_RUN_MIE_SPEED", "0") == "1"


def _median_runtime(func, repeats=3):
    """Return median runtime and final function output."""
    timings = []
    last = None
    for _ in range(repeats):
        t0 = perf_counter()
        last = func()
        timings.append(perf_counter() - t0)
    return float(np.median(timings)), last


@pytest.mark.skipif(not RUN_BENCH, reason="Set MIEPYTHON_RUN_MIE_SPEED=1 to run mie backend speed benchmark")
def test_mie_backend_speed_compare():
    """Compare runtime and output agreement for no-JIT and JIT Mie backends."""
    rng = np.random.default_rng(12345)

    n_particles = 4000
    refr = rng.uniform(1.2, 2.0, n_particles)
    refi = np.exp(rng.uniform(np.log(1e-4), np.log(5e-1), n_particles))
    xvals = np.exp(rng.uniform(np.log(5e-2), np.log(50), n_particles))
    mvals = refr - 1j * refi

    mu = np.linspace(-1.0, 1.0, 361)
    m_ref = 1.5 - 0.05j
    x_ref = 12.0

    # Warm up numba kernels so benchmark times reflect steady-state runtime.
    mie_jit._single_sphere_nb(np.complex128(mvals[0]), float(xvals[0]), 0, True)
    mie_jit._S1_S2_nb(np.complex128(m_ref), float(x_ref), mu, 0)

    def run_single_nojit():
        out = np.empty((4, n_particles), dtype=np.float64)
        for i in range(n_particles):
            out[:, i] = mie_nojit._single_sphere_py(mvals[i], float(xvals[i]), 0, True)
        return out

    def run_single_jit():
        out = np.empty((4, n_particles), dtype=np.float64)
        for i in range(n_particles):
            out[:, i] = mie_jit._single_sphere_nb(np.complex128(mvals[i]), float(xvals[i]), 0, True)
        return out

    def run_s12_nojit():
        return mie_nojit._S1_S2_py(m_ref, x_ref, mu, 0)

    def run_s12_jit():
        return mie_jit._S1_S2_nb(np.complex128(m_ref), float(x_ref), mu, 0)

    t_single_nojit, single_nojit = _median_runtime(run_single_nojit, repeats=3)
    t_single_jit, single_jit = _median_runtime(run_single_jit, repeats=3)

    t_s12_nojit, s12_nojit = _median_runtime(run_s12_nojit, repeats=5)
    t_s12_jit, s12_jit = _median_runtime(run_s12_jit, repeats=5)

    np.testing.assert_allclose(single_nojit, single_jit, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s12_nojit[0], s12_jit[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s12_nojit[1], s12_jit[1], rtol=1e-10, atol=1e-12)

    print(f"single_sphere nojit median: {t_single_nojit:.4f} s")
    print(f"single_sphere jit median:   {t_single_jit:.4f} s")
    print(f"single_sphere speedup:      {t_single_nojit / t_single_jit:.2f}x")
    print(f"S1_S2 nojit median:         {t_s12_nojit:.4f} s")
    print(f"S1_S2 jit median:           {t_s12_jit:.4f} s")
    print(f"S1_S2 speedup:              {t_s12_nojit / t_s12_jit:.2f}x")

    assert t_single_nojit > 0.0
    assert t_single_jit > 0.0
    assert t_s12_nojit > 0.0
    assert t_s12_jit > 0.0
