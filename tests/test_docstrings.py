"""Run the examples in the docstrings and check they still produce what they claim.

Two of the examples in ``core.py`` had rotted before this file existed: one called
a function name that never existed (``mie_coefficients``), another used the
signature of a different function and printed a made-up repr, and the array shown
still had the padding element that ``an_bn`` no longer returns.  Nothing executed
them, so nothing noticed.

Examples are written with explicit formatting or rounding rather than bare array
reprs, so they say what they mean without being hostage to NumPy's print
precision or a last-digit difference between platforms.
"""

import doctest
import importlib
import inspect

import pytest

# every module in the package; those without examples simply report zero
MODULES = [
    "miepython.core",
    "miepython.field",
    "miepython.util",
    "miepython.vsh",
    "miepython.bessel",
    "miepython.rayleigh",
    "miepython.monte_carlo",
    "miepython.mie_nojit",
    "miepython.mie_jit",
    "miepython._backend",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples(module_name):
    """Every ``>>>`` example in the module must still be accurate."""
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False, report=True)
    assert results.failed == 0, f"{results.failed} of {results.attempted} examples failed in {module_name}"


# modules whose docstring lists call signatures, which must match the real ones
SIGNATURE_LISTS = [
    ("miepython.vsh", ["M_odd", "M_even", "N_odd", "N_even", "mie_pi", "mie_tau"]),
]


@pytest.mark.parametrize("module_name, names", SIGNATURE_LISTS)
def test_documented_signatures_match_the_code(module_name, names):
    """A signature written out in prose is a claim, and it can be wrong.

    ``vsh``'s module docstring advertised ``M_odd(n, k, d_sphere, r, theta, phi)``
    for all four harmonics.  The real signature takes a wavelength rather than a
    wavenumber and one more argument besides -- ``(n, lambda0, d_sphere, m_index, r,
    theta, phi)`` -- so anyone following the docstring got a TypeError.  Doctests
    cannot catch this, because these are not examples.
    """
    module = importlib.import_module(module_name)
    doc = module.__doc__
    for name in names:
        signature = str(inspect.signature(getattr(module, name)))
        assert f"{name}{signature}" in doc, f"{module_name} docstring misstates {name}{signature}"


def test_the_documented_modules_are_worth_running():
    """Guard against this file silently checking nothing.

    If the examples in core.py and field.py ever disappear, that is a change worth
    noticing rather than a quietly passing test suite.
    """
    total = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        total += doctest.testmod(module, verbose=False).attempted
    assert total >= 30, f"only {total} docstring examples found; did they get deleted?"
