"""Keep each backend's tests in a process that really uses that backend.

``miepython/_backend.py`` binds its kernels the first time the package is
imported, so ``MIEPYTHON_USE_JIT`` only has an effect for whichever module
imports ``miepython`` first.  One pytest process therefore cannot exercise both
backends, and the ``test_jit*`` files used to run against the pure-python
kernels whenever anything else imported the package ahead of them, passing while
testing the wrong code.

This file reads the backend that actually got selected -- a conftest is imported
before any test module, so the value here is the one every test will see -- and
skips the files belonging to the other backend with a reason that says how to run
them.  ``make test`` runs pytest once per backend so both halves are covered.

The ``kernels`` fixture below is the way to avoid that split entirely.  The
env-var switch only governs which kernels ``miepython`` re-exports; both kernel
modules can always be imported directly.  A test that names the kernels through
the fixture therefore covers both backends inside a single process, so it needs
only one copy.  Prefer it over writing a ``test_jit*``/``test_nojit*`` pair.
Tests that go through the high-level API in ``core.py`` cannot use the fixture,
but ``core.py`` holds no backend-specific code, so one copy of those run under
both passes is enough.
"""

from types import SimpleNamespace

import pytest

import miepython

USE_JIT = miepython.USE_JIT

JIT_PREFIX = "test_jit"
NOJIT_PREFIX = "test_nojit"

# the low-level kernels, named without their backend suffix
KERNEL_NAMES = (
    "D_calc",
    "D_calc_down",
    "psi_downwards",
    "an_bn",
    "cn_dn",
    "pi_tau",
    "S1_S2",
    "single_sphere",
    "small_sphere",
    "small_conducting_sphere",
)

# these already share a name across the two modules
SHARED_NAMES = ("_Lentz_Dn", "_D_upwards", "_D_downwards")


@pytest.fixture(params=["nojit", "jit"], ids=["pure-python", "numba"])
def kernels(request):
    """Expose one backend's kernels under names that do not mention the backend.

    Parametrized over both, so a test using this fixture runs twice in the same
    process, once per backend, and does not need a duplicated file.
    """
    from miepython import mie_jit, mie_nojit  # pylint: disable=import-outside-toplevel

    module, suffix = (mie_jit, "_nb") if request.param == "jit" else (mie_nojit, "_py")

    namespace = SimpleNamespace(name=request.param, module=module)
    for name in KERNEL_NAMES:
        setattr(namespace, name, getattr(module, f"_{name}{suffix}"))
    for name in SHARED_NAMES:
        setattr(namespace, name.lstrip("_"), getattr(module, name))
    return namespace


def pytest_report_header(config):
    """Say which backend this run exercises."""
    _ = config
    backend = "numba (MIEPYTHON_USE_JIT=1)" if USE_JIT else "pure python (MIEPYTHON_USE_JIT=0)"
    return f"miepython backend: {backend}"


def pytest_collection_modifyitems(config, items):
    """Skip the test files that do not match the active backend."""
    _ = config
    skip_jit = pytest.mark.skip(reason="needs MIEPYTHON_USE_JIT=1; the numba backend is not active")
    skip_nojit = pytest.mark.skip(reason="needs MIEPYTHON_USE_JIT=0; the numba backend is active")

    for item in items:
        name = item.path.name
        if name.startswith(NOJIT_PREFIX):
            if USE_JIT:
                item.add_marker(skip_nojit)
        elif name.startswith(JIT_PREFIX):
            if not USE_JIT:
                item.add_marker(skip_jit)
