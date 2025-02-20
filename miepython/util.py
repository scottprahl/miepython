"""Functions to format complex numbers or arrays of complex numbers."""

import numpy as np

_all_ = ("cs_scalar", "cs_vector", "cs")


def cs_scalar(z, N=5, include_parens=True):
    """Convert complex number to string for printing."""
    if z.imag < 0:
        form = "%% .%df - %%.%dfj" % (N, N)
    else:
        form = "%% .%df + %%.%dfj" % (N, N)
    if include_parens:
        form = "(" + form + ")"
    return form % (z.real, abs(z.imag))


def cs_vector(z, N=5):
    """Convert a tuple to string for printing."""
    s = "("
    for zz in z:
        if s != "(":
            s += ", "
        s += cs_scalar(zz, N, include_parens=False)
    s += ")"
    return s


def cs(z, N=5):
    """Convert complex number to string for printing."""
    if np.isscalar(z):
        return cs_scalar(z, N)

    s = ""
    for zz in z:
        if s != "":
            s += ", "
        s += cs_scalar(zz, N)

    return s
