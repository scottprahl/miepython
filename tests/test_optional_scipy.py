"""Test that miepython imports and behaves sensibly without SciPy.

``miepython/__init__.py`` catches a missing SciPy, substitutes placeholders for
the near-field functions, and lets everything else import as usual.  That branch
is what keeps the package usable in JupyterLite, where SciPy is an extra
download, so it is worth testing rather than assuming.

The tests here re-import the package with an import hook that hides a module.
Every one restores ``sys.modules`` exactly, so the copy other test files already
hold keeps working.
"""

import contextlib
import importlib
import sys

import pytest

import miepython
from miepython import field as miepython_field

NEAR_FIELD_NAMES = (
    "e_far",
    "e_near",
    "h_near",
    "eh_near",
    "e_near_cartesian",
    "h_near_cartesian",
    "eh_near_cartesian",
)


class _Blocker:
    """Meta-path finder that refuses to import a chosen module tree."""

    def __init__(self, prefix):
        self.prefix = prefix

    def find_spec(self, fullname, path=None, target=None):
        """Raise for the blocked tree; returning nothing defers to the next finder."""
        _ = path, target
        if fullname == self.prefix or fullname.startswith(self.prefix + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)


@contextlib.contextmanager
def reimported_without(prefix):
    """Re-import miepython while `prefix` is unimportable, then put everything back."""
    saved = dict(sys.modules)
    blocker = _Blocker(prefix)
    for name in list(sys.modules):
        if name == "miepython" or name.startswith("miepython.") or name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        yield importlib.import_module("miepython")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.clear()
        sys.modules.update(saved)


def test_the_blocker_actually_blocks():
    """Guard the mechanism itself, so a silent no-op cannot fake a pass."""
    with reimported_without("scipy"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("scipy.special")


def test_package_imports_without_scipy():
    """The core Mie code must not need SciPy."""
    with reimported_without("scipy") as mie_noscipy:
        assert mie_noscipy.efficiencies_mx(1.5 - 0.01j, 2.0)[0] > 0
        s1, s2 = mie_noscipy.S1_S2(1.5, 2.0, [0.0, 1.0], norm="wiscombe")  # a list, deliberately
        assert len(s1) == len(s2) == 2


@pytest.mark.parametrize("name", NEAR_FIELD_NAMES)
def test_near_field_placeholder_explains_itself(name):
    """Each near-field entry point is replaced by something that says why."""
    with reimported_without("scipy") as mie_noscipy:
        placeholder = getattr(mie_noscipy, name)
        assert placeholder.__name__ == name
        assert placeholder.__qualname__ == name
        assert "scipy" in placeholder.__doc__

        with pytest.raises(ModuleNotFoundError) as excinfo:
            placeholder()
        message = str(excinfo.value)
        assert "scipy" in message
        # the original ImportError is chained on, so the real cause stays visible
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_placeholders_accept_any_arguments():
    """They must fail on the missing dependency, not on a signature mismatch."""
    with reimported_without("scipy") as mie_noscipy:
        with pytest.raises(ModuleNotFoundError, match="scipy"):
            mie_noscipy.e_near(1.0, 1.0, 1.5 + 0j, 1.0, 1.5, 0.5, 0.0, include_incident=True)


def test_with_scipy_present_the_real_functions_are_exported():
    """The healthy path must bind the actual implementations."""
    for name in NEAR_FIELD_NAMES:
        assert getattr(miepython, name) is getattr(miepython_field, name), name


def test_rayleigh_is_still_reachable_without_scipy():
    """It needs only NumPy, so the guard must not take it down with the rest."""
    with reimported_without("scipy") as mie_noscipy:
        qext, qsca, _, g = mie_noscipy.rayleigh.efficiencies_mx(1.5, 0.01)
        assert qsca > 0
        assert qext >= qsca
        assert g == pytest.approx(0.0, abs=1e-9)
        assert "rayleigh" in mie_noscipy.__all__


def test_vsh_is_absent_without_scipy():
    """Every function in it needs scipy, so there is nothing useful to bind."""
    with reimported_without("scipy") as mie_noscipy:
        assert not hasattr(mie_noscipy, "vsh")
        # and it must stay out of __all__, or ``from miepython import *`` would raise
        assert "vsh" not in mie_noscipy.__all__
        missing = [name for name in mie_noscipy.__all__ if not hasattr(mie_noscipy, name)]
        assert missing == [], f"__all__ promises unbound names without scipy: {missing}"


def test_monte_carlo_works_without_scipy():
    """Sampling the phase function is core Mie work, not near-field work."""
    with reimported_without("scipy"):
        monte_carlo = importlib.import_module("miepython.monte_carlo")
        mu, cdf = monte_carlo.mu_with_uniform_cdf(1.33, 2.0, 50)
        assert len(mu) == 50
        assert cdf[0] == 0.0 and cdf[-1] == 1.0


def test_a_non_scipy_import_failure_is_not_swallowed():
    """Only a missing SciPy is tolerated; anything else must propagate."""
    with pytest.raises(ModuleNotFoundError) as excinfo:
        with reimported_without("miepython.bessel"):
            pass
    assert "bessel" in str(excinfo.value)


def test_import_machinery_is_left_clean():
    """No blocker may outlive the context manager."""
    before = list(sys.meta_path)
    with contextlib.suppress(ModuleNotFoundError):
        with reimported_without("scipy"):
            pass
    assert list(sys.meta_path) == before
    # a fresh import must still find the real package, not a hollowed-out one
    reloaded = importlib.import_module("miepython")
    assert reloaded.efficiencies_mx(1.5, 2.0)[0] > 0
    assert reloaded is miepython, "sys.modules was not restored"
