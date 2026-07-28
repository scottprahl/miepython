"""
Benchmark of miepython.efficiencies_mx on a large ensemble of particles.

This benchmark (contributed by Andrew Geiss) measures the runtime of
`miepython.efficiencies_mx` for N = 100,000 randomly generated particles, and
demonstrates the performance regression observed between versions 2.5.5 and
3.0.0:

  • v2.5.5: 0.27 seconds for N=100000 jitted
  • v3.0.0: 4.41 seconds for N=100000 jitted
  • v3.0.1: 0.15 seconds for N=100000 jitted
  • v3.0.1: 4.00 seconds for N=100000 no jit

Run it once per backend::

    MIEPYTHON_USE_JIT=0 python tests/benchmark_efficiencies.py
    MIEPYTHON_USE_JIT=1 python tests/benchmark_efficiencies.py

``--compare`` instead times the two kernel sets against each other in one process
and prints the speedup quoted in the README.  ``make speed`` does all three.

The file deliberately does not start with ``test_``: it used to exist as a
``test_jit_speed.py``/``test_nojit_speed.py`` pair whose timing ran at import,
so pytest spent about seven seconds running benchmarks during collection while
collecting no tests at all.  The backend now comes from the environment, which
also stops the JIT-named copy from silently timing the pure-python kernels.
The correctness half of the old ``test_mie_backend_speed.py`` -- a random-ensemble
agreement check between the backends -- now lives in ``test_backend_parity.py``,
where it actually runs; the timing half is the ``--compare`` mode below.

Original discussion and issue filed at:
https://github.com/scottprahl/miepython/issues/28
"""

import sys
from time import perf_counter, time

import numpy as np

import miepython as mie
from miepython import mie_jit, mie_nojit

# Number of particles
N = 100_000


def main():
    """Time efficiencies_mx over N particles with whichever backend is bound."""
    rng = np.random.default_rng(0)

    # Random refractive indices (n - ik) and size parameters x
    refr = rng.uniform(1.0, 2.0, N)
    refi = np.exp(rng.uniform(np.log(1e-4), np.log(1.0), N))
    x = np.exp(rng.uniform(np.log(0.01), np.log(100), N))
    m = refr - 1j * refi

    # warm up so a JIT run reports steady-state speed rather than compile time
    mie.efficiencies_mx(m[:2], x[:2])

    t0 = time()
    mie.efficiencies_mx(m, x)
    elapsed = time() - t0

    state = "enabled" if mie.USE_JIT else "not enabled"
    print(f"JIT is {state}, miepython version is {mie.__version__}")
    print(f"{elapsed:.3f} seconds when N={N}")


def _median(func, repeats=3):
    """Return the median runtime of func over a few repeats."""
    timings = []
    for _ in range(repeats):
        t0 = perf_counter()
        func()
        timings.append(perf_counter() - t0)
    return float(np.median(timings))


def compare():
    """Time the two kernel sets side by side, independent of MIEPYTHON_USE_JIT.

    Calling the kernels directly means one process can time both, so this does not
    care which backend the package bound.
    """
    rng = np.random.default_rng(12345)
    n = 4000
    refr = rng.uniform(1.2, 2.0, n)
    refi = np.exp(rng.uniform(np.log(1e-4), np.log(5e-1), n))
    xvals = np.exp(rng.uniform(np.log(5e-2), np.log(50), n))
    mvals = refr - 1j * refi

    mu = np.linspace(-1.0, 1.0, 361)
    m_ref, x_ref = 1.5 - 0.05j, 12.0

    # warm up the numba kernels so the timings show steady-state speed
    mie_jit._single_sphere_nb(np.complex128(mvals[0]), float(xvals[0]), 0, True)
    mie_jit._S1_S2_nb(np.complex128(m_ref), float(x_ref), mu, 0)

    def single_py():
        for i in range(n):
            mie_nojit._single_sphere_py(mvals[i], float(xvals[i]), 0, True)

    def single_nb():
        for i in range(n):
            mie_jit._single_sphere_nb(np.complex128(mvals[i]), float(xvals[i]), 0, True)

    pairs = (
        ("single_sphere", single_py, single_nb, 3),
        (
            "S1_S2",
            lambda: mie_nojit._S1_S2_py(m_ref, x_ref, mu, 0),
            lambda: mie_jit._S1_S2_nb(np.complex128(m_ref), float(x_ref), mu, 0),
            5,
        ),
    )
    for label, py_func, nb_func, repeats in pairs:
        t_py = _median(py_func, repeats)
        t_nb = _median(nb_func, repeats)
        print(f"{label:<14} pure python {t_py:.4f} s   numba {t_nb:.4f} s   speedup {t_py / t_nb:.1f}x")


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    else:
        main()
