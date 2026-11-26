"""Time-harmonic sign convention for the complex values miepython returns.

A monochromatic field can be written with time dependence exp(+jωt) or
exp(-iωt).  The two descriptions carry the same physics and, for the
non-magnetic spheres handled here, every complex result in one is the complex
conjugate of the same result in the other:

``"+"`` (the default), exp(+jωt)
    The convention of van de Hulst, Kerker and Wiscombe's MIEV0, natural for the
    m = n - ik form of the refractive index that miepython has always used.
    Outgoing spherical waves go as h_n^(2)(kr), a plane wave travelling along +z
    as exp(-jkz), and ``S1_S2`` reproduces MIEV0's amplitudes directly.

``"-"``, exp(-iωt)
    The convention of Bohren & Huffman, Jackson and scattnlay.  Outgoing waves go
    as h_n^(1)(kr), a plane wave along +z as exp(+ikz), and the amplitudes,
    coefficients and phase matrix match Bohren & Huffman's equations and tables.

Only complex results depend on the choice: the Mie coefficients, S1 and S2, the
near and far fields, the vector spherical harmonics, and the S34 element of the
phase matrix (which changes sign, along with the handedness of the Stokes V).
Efficiencies, g, intensities and the other phase-matrix elements are identical.

The refractive index is accepted with either sign of its imaginary part and is
always taken to be absorbing, whatever convention is active.

The setting is global to the process.  Library code that needs a particular
convention should use the context manager, which restores the caller's choice::

    with miepython.time_convention("-"):
        a, b = miepython.coefficients(m, x)

The raw kernels (``an_bn``, ``cn_dn``, ``_S1_S2``, ...) ignore this setting and
always work in exp(-iωt); the public functions convert on the way out.
"""

import contextlib

import numpy as np

__all__ = ("get_time_convention", "set_time_convention", "time_convention")

_VALID = ("+", "-")
_current = "+"


def _validate(sign):
    """Return ``sign`` if it names a convention, else raise ``ValueError``."""
    if not isinstance(sign, str) or sign not in _VALID:
        raise ValueError(f"time convention must be '+' for exp(+jωt) or '-' for exp(-iωt), not {sign!r}")
    return sign


def get_time_convention():
    """Return the active time convention.

    Returns:
        str: ``"+"`` for exp(+jωt) or ``"-"`` for exp(-iωt).
    """
    return _current


def set_time_convention(sign):
    """Choose the time convention for every complex result miepython returns.

    Args:
        sign (str): ``"+"`` for exp(+jωt), the default, or ``"-"`` for exp(-iωt).

    Returns:
        str: The convention that was active before the call, so it can be restored.

    Raises:
        ValueError: if ``sign`` is neither ``"+"`` nor ``"-"``.
    """
    global _current  # pylint: disable=global-statement
    previous = _current
    _current = _validate(sign)
    return previous


@contextlib.contextmanager
def time_convention(sign):
    """Use a time convention inside a ``with`` block, then restore the previous one.

    Args:
        sign (str): ``"+"`` for exp(+jωt) or ``"-"`` for exp(-iωt).

    Yields:
        str: The convention in force inside the block.

    Raises:
        ValueError: if ``sign`` is neither ``"+"`` nor ``"-"``.
    """
    previous = set_time_convention(sign)
    try:
        yield sign
    finally:
        set_time_convention(previous)


def _convert(value):
    """Convert between exp(-iωt), used internally, and the active convention.

    Conjugation is its own inverse, so the same call takes an internal value out
    to the user and a user-supplied value (such as ``abcd``) in.
    """
    if _current == "+":
        return np.conjugate(value)
    return value
