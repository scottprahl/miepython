"""Test which submodules a bare ``import miepython`` actually reaches.

``rayleigh``, ``vsh`` and ``monte_carlo`` used to need a separate import statement
that nothing documented, so ``miepython.rayleigh`` raised AttributeError after
``import miepython``.  The three are now handled differently, for reasons:

* ``rayleigh`` needs only NumPy, so the package imports it outright.
* ``vsh`` needs the optional scipy, so it is imported inside the same guard as
  ``field`` and is simply absent when scipy is not installed.
* ``monte_carlo`` imports ``miepython``, so importing it from the package would be
  circular.  It stays a separate import, and the module docstring says so.

Checking an attribute in-process proves little here, because another test file
importing ``miepython.monte_carlo`` sets that attribute on the package for everyone.
The claims below therefore run in a fresh interpreter, where the import really is
the first one.
"""

import subprocess
import sys

import numpy as np
import pytest

import miepython
import miepython.monte_carlo
import miepython.rayleigh
import miepython.vsh


def run(code):
    """Run code in a fresh interpreter and return its stdout, or fail loudly."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    return result.stdout.strip()


class TestReachableFromTheRoot:
    """Test the submodules a plain ``import miepython`` is enough to use."""

    def test_rayleigh_needs_no_second_import(self):
        """The whole point of the change."""
        out = run(
            "import miepython\n"
            "qext, qsca, qback, g = miepython.rayleigh.efficiencies_mx(1.5, 0.01)\n"
            "print(qsca > 0, abs(g) < 1e-9)\n"
        )
        assert out == "True True"

    def test_vsh_needs_no_second_import_when_scipy_is_installed(self):
        """Guarded, but bound on the healthy path."""
        out = run("import miepython\nprint(miepython.vsh.M_odd.__name__)\n")
        assert out == "M_odd"

    def test_the_bound_submodules_are_the_real_modules(self):
        """Not shims: what the package binds is the module the import system made."""
        assert miepython.rayleigh is sys.modules["miepython.rayleigh"]
        assert miepython.vsh is sys.modules["miepython.vsh"]

    def test_the_bound_submodules_expose_their_public_api(self):
        """Reaching them from the root has to give the whole module, not a fragment."""
        for module in (miepython.rayleigh, miepython.vsh):
            assert module.__all__, f"{module.__name__} declares no public names"
            for name in module.__all__:
                assert hasattr(module, name), f"{module.__name__}.{name} is missing"


class TestStarImport:
    """Test that ``from miepython import *`` stays sound."""

    def test_rayleigh_is_exported(self):
        """It is always bound, so it belongs in __all__."""
        assert "rayleigh" in miepython.__all__

    def test_vsh_is_not_exported(self):
        """It is bound only with scipy, and a star-import must not depend on that."""
        assert "vsh" not in miepython.__all__

    def test_every_exported_name_exists(self):
        """A name in __all__ that is not bound makes ``import *`` raise."""
        missing = [name for name in miepython.__all__ if not hasattr(miepython, name)]
        assert missing == [], f"__all__ promises names the package does not bind: {missing}"

    def test_star_import_succeeds_and_brings_rayleigh(self):
        """Run it for real rather than inferring from __all__."""
        out = run(
            "from miepython import *\n"
            "print(rayleigh.efficiencies_mx(1.5, 0.01)[1] > 0)\n"  # noqa: F821 - star import
        )
        assert out == "True"


class TestMonteCarlo:
    """Test the submodule that cannot be imported from the package root."""

    def test_it_imports_first_without_a_circular_import_error(self):
        """Importing it before anything else must work, cycle or no cycle."""
        out = run(
            "import numpy as np\n"
            "import miepython.monte_carlo as mc\n"
            "mu, cdf = mc.mu_with_uniform_cdf(1.33, 2.0, 100)\n"
            "print(len(mu), cdf[0] == 0.0, cdf[-1] == 1.0)\n"
        )
        assert out == "100 True True"

    def test_the_docstring_says_how_to_import_it(self):
        """The user has to be told, since the root import is not enough."""
        doc = miepython.monte_carlo.__doc__
        assert "import miepython.monte_carlo" in doc
        assert "circular" in doc, "the reason it is not imported for you should be stated"

    def test_the_package_docstring_covers_all_three_submodules(self):
        """Someone reading ``help(miepython)`` should not have to guess."""
        doc = miepython.__doc__
        for name in ("miepython.rayleigh", "miepython.vsh", "miepython.monte_carlo"):
            assert name in doc, f"{name} is undocumented in the package docstring"


class TestVshNoLongerImportsThePackage:
    """Test that vsh reaches the kernels directly instead of through ``miepython``.

    ``vsh`` used to do ``import miepython as mie`` for a single call to ``D_calc``,
    which is a cycle that happens to work only because the call is deferred to run
    time.  Importing vsh from ``__init__`` would have rested on that accident, so
    the import now goes to ``._backend``.
    """

    def test_vsh_holds_no_reference_to_the_package(self):
        """A ``miepython`` global here would mean the cycle is back."""
        package_refs = [
            name for name, value in vars(miepython.vsh).items() if value is miepython and not name.startswith("__")
        ]
        assert package_refs == [], f"vsh still imports the package as {package_refs}"

    def test_vsh_can_be_imported_on_its_own(self):
        """And it still computes: the D_calc it needs is reached through ._backend."""
        out = run(
            "import numpy as np\n"
            "import miepython.vsh as vsh\n"
            # r inside the sphere of diameter 1, which is the branch calling D_calc
            "N = vsh.N_odd(1, 1.0, 1.0, 1.5, 0.25, 0.7, 0.3)\n"
            "print(len(N), np.all(np.isfinite(np.asarray(N))))\n"
        )
        assert out == "3 True"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_internal_field_still_uses_d_calc(self, n):
        """The reroute must not change what the interior field evaluates to.

        Inside the sphere ``N_odd`` reaches for the logarithmic derivative, which is
        the one thing vsh used to import through the package, so a field that is
        finite and nonzero there is what proves the new import path works.
        """
        inside = np.asarray(miepython.vsh.N_odd(n, 1.0, 1.0, 1.5, 0.2, 0.6, 0.4))
        assert np.all(np.isfinite(inside))
        assert np.any(inside != 0)
