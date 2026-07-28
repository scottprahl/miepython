"""Test that the numba and pure-python kernels are interchangeable.

Both kernel modules can be imported regardless of ``MIEPYTHON_USE_JIT``, so this
file compares them side by side in one process and needs no JIT/no-JIT pair.  It
guards the invariant the two implementations exist to satisfy: same signatures,
same answers.

Reaching into the underscore-prefixed kernels is the point of this file, so the
protected-access warning is off for the whole module.
"""

# pylint: disable=protected-access

import inspect

import numpy as np
import pytest

from miepython import mie_jit, mie_nojit

# (label, pure-python kernel, numba kernel).  The tolerances further down sit just
# above the worst disagreement measured over the sweep, so real drift trips them.
KERNEL_PAIRS = [
    ("D_calc", "_D_calc_py", "_D_calc_nb"),
    ("an_bn", "_an_bn_py", "_an_bn_nb"),
    ("cn_dn", "_cn_dn_py", "_cn_dn_nb"),
    ("pi_tau", "_pi_tau_py", "_pi_tau_nb"),
    ("S1_S2", "_S1_S2_py", "_S1_S2_nb"),
    ("single_sphere", "_single_sphere_py", "_single_sphere_nb"),
    ("small_sphere", "_small_sphere_py", "_small_sphere_nb"),
    ("small_conducting_sphere", "_small_conducting_sphere_py", "_small_conducting_sphere_nb"),
]

INDICES = [complex(0.75, 0.0), complex(1.05, 0.0), complex(1.33, -0.01), complex(1.5, -0.5), complex(2.5, 0.0)]
SIZES = [0.1, 1.0, 5.0, 20.0, 62.0]


def worst_relative(a, b):
    """Largest relative gap between two arrays, ignoring underflowed entries."""
    a = np.asarray(a)
    b = np.asarray(b)
    keep = np.abs(a) > 1e-250
    if not np.any(keep):
        return 0.0
    return float(np.max(np.abs(b[keep] - a[keep]) / np.abs(a[keep])))


@pytest.mark.parametrize("name,py_name,nb_name", KERNEL_PAIRS, ids=[p[0] for p in KERNEL_PAIRS])
def test_signatures_match(name, py_name, nb_name):
    """A call that works on one backend must work on the other."""
    py_sig = inspect.signature(getattr(mie_nojit, py_name))
    nb_sig = inspect.signature(getattr(mie_jit, nb_name).py_func)
    assert str(py_sig) == str(nb_sig), name


def test_an_bn_accepts_the_default_n_pole():
    """an_bn(m, x) worked only on the pure-python backend before."""
    for m in (complex(1.5, -0.1), 1.5):
        for x in (0.5, 5.0):
            py_a, py_b = mie_nojit._an_bn_py(m, x)
            nb_a, nb_b = mie_jit._an_bn_nb(m, x)
            assert len(py_a) == len(nb_a)
            assert worst_relative(py_a, nb_a) < 1e-12
            assert worst_relative(py_b, nb_b) < 1e-11
            # and the default must mean the same thing as passing it
            np.testing.assert_allclose(py_a, mie_nojit._an_bn_py(m, x, 0)[0])
            np.testing.assert_allclose(nb_a, mie_jit._an_bn_nb(m, x, 0)[0])


def test_an_bn_agrees():
    """a_n and b_n match across the two backends."""
    worst_a = worst_b = 0.0
    for m in INDICES:
        for x in SIZES:
            py_a, py_b = mie_nojit._an_bn_py(m, x, 0)
            nb_a, nb_b = mie_jit._an_bn_nb(m, x, 0)
            assert len(py_a) == len(nb_a)
            worst_a = max(worst_a, worst_relative(py_a, nb_a))
            worst_b = max(worst_b, worst_relative(py_b, nb_b))
    assert worst_a < 1e-11, worst_a
    assert worst_b < 1e-10, worst_b


def test_cn_dn_agrees():
    """The internal-field coefficients match across the two backends."""
    worst = 0.0
    for m in INDICES:
        for x in SIZES:
            py_c, py_d = mie_nojit._cn_dn_py(m, x, 0)
            nb_c, nb_d = mie_jit._cn_dn_nb(m, x, 0)
            assert len(py_c) == len(nb_c)
            worst = max(worst, worst_relative(py_c, nb_c), worst_relative(py_d, nb_d))
    assert worst < 1e-11, worst


def test_d_calc_agrees():
    """The logarithmic derivatives match across the two backends."""
    worst = 0.0
    for m in INDICES:
        for x in SIZES:
            worst = max(worst, worst_relative(mie_nojit._D_calc_py(m, x, 15), mie_jit._D_calc_nb(m, x, 15)))
    assert worst < 1e-12, worst


def test_pi_tau_agrees():
    """The angular functions match across the two backends."""
    for n_terms in (1, 4, 12):
        for mu in (-0.9, -0.25, 0.0, 0.4, 0.99):
            py_pi, py_tau = np.zeros(n_terms), np.zeros(n_terms)
            nb_pi, nb_tau = np.zeros(n_terms), np.zeros(n_terms)
            mie_nojit._pi_tau_py(mu, py_pi, py_tau)
            mie_jit._pi_tau_nb(mu, nb_pi, nb_tau)
            np.testing.assert_allclose(nb_pi, py_pi, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(nb_tau, py_tau, rtol=1e-13, atol=1e-15)


def test_s1_s2_agrees():
    """The scattering amplitudes match across the two backends."""
    mu = np.linspace(-1.0, 1.0, 21)
    worst = 0.0
    for m in INDICES:
        for x in SIZES:
            py_s1, py_s2 = mie_nojit._S1_S2_py(m, x, mu, 0)
            nb_s1, nb_s2 = mie_jit._S1_S2_nb(m, x, mu, 0)
            worst = max(worst, worst_relative(py_s1, nb_s1), worst_relative(py_s2, nb_s2))
    assert worst < 1e-11, worst


def test_single_sphere_agrees():
    """The efficiencies match across the two backends."""
    worst = {}
    for m in INDICES:
        for x in SIZES:
            py_q = mie_nojit._single_sphere_py(m, x, 0, True)
            nb_q = mie_jit._single_sphere_nb(m, x, 0, True)
            for label, want, got in zip(("qext", "qsca", "qback", "g"), py_q, nb_q):
                if abs(want) > 1e-30:
                    worst[label] = max(worst.get(label, 0.0), abs(got - want) / abs(want))
    # qback and g sum the series differently in the two kernels (np.dot against a
    # scalar loop), so they carry a slightly looser bound
    assert worst["qext"] < 1e-13, worst
    assert worst["qsca"] < 1e-13, worst
    assert worst["qback"] < 1e-11, worst
    assert worst["g"] < 1e-11, worst


def test_small_sphere_kernels_agree():
    """The small-sphere shortcuts match across the two backends."""
    for m in INDICES:
        for x in (0.001, 0.01, 0.05, 0.099):
            np.testing.assert_allclose(mie_jit._small_sphere_nb(m, x), mie_nojit._small_sphere_py(m, x), rtol=1e-12)
            np.testing.assert_allclose(
                mie_jit._small_conducting_sphere_nb(m, x),
                mie_nojit._small_conducting_sphere_py(m, x),
                rtol=1e-12,
            )


def test_exported_kernel_sets_match():
    """Neither module may grow or lose a kernel without the other."""
    py_names = {n[: -len("_py")] for n in mie_nojit.__all__ if n.endswith("_py")}
    nb_names = {n[: -len("_nb")] for n in mie_jit.__all__ if n.endswith("_nb")}
    assert py_names == nb_names


def test_agrees_over_a_random_sweep():
    """Both backends match across a broad random ensemble, not just fixed cases.

    This check used to live in a benchmark gated behind MIEPYTHON_RUN_MIE_SPEED,
    which nothing ever set, so it never ran.  It costs about a tenth of a second.
    """
    rng = np.random.default_rng(12345)
    n_particles = 4000
    refr = rng.uniform(1.2, 2.0, n_particles)
    refi = np.exp(rng.uniform(np.log(1e-4), np.log(5e-1), n_particles))
    xvals = np.exp(rng.uniform(np.log(5e-2), np.log(50), n_particles))
    mvals = refr - 1j * refi

    py_out = np.empty((4, n_particles))
    nb_out = np.empty((4, n_particles))
    for i in range(n_particles):
        py_out[:, i] = mie_nojit._single_sphere_py(mvals[i], float(xvals[i]), 0, True)
        nb_out[:, i] = mie_jit._single_sphere_nb(np.complex128(mvals[i]), float(xvals[i]), 0, True)
    np.testing.assert_allclose(nb_out, py_out, rtol=1e-10, atol=1e-12)

    mu = np.linspace(-1.0, 1.0, 361)
    m_ref, x_ref = 1.5 - 0.05j, 12.0
    py_s1, py_s2 = mie_nojit._S1_S2_py(m_ref, x_ref, mu, 0)
    nb_s1, nb_s2 = mie_jit._S1_S2_nb(np.complex128(m_ref), float(x_ref), mu, 0)
    np.testing.assert_allclose(nb_s1, py_s1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(nb_s2, py_s2, rtol=1e-10, atol=1e-12)
