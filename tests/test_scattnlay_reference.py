"""Check miepython against scattnlay, an independent Mie implementation.

scattnlay (https://github.com/ovidiopr/scattnlay) is a separate C++ code for
multilayered spheres.  Reference values were produced with its 2.4 Python package
at commit 211fc8d2168deff500b67042cea952fc74c84b64 by
``docs/data/generate_scattnlay_reference_fields.py``.

The comparison used to live only in ``docs/15_2D_fields.ipynb``, which prints and
plots the errors without asserting anything, so a regression in the near-field code
was invisible.  It also fetched the arrays over the network from the published
branch, meaning it validated released data rather than the working tree.

Two levels of check:

* the scalar efficiencies are inlined below, the same way this suite already
  inlines Wiscombe's MIEV0 numbers, so they need no data files and always run;
* the 2D field grids are 3 MB of ``.npy`` under ``docs/data``, which is outside
  the sdist, so those tests skip when the docs tree is absent.  A repository
  checkout, which is what CI uses, always has them.
"""

import json
import pathlib

import numpy as np
import pytest

import miepython as mie
from miepython.field import eh_near_cartesian

DATA = pathlib.Path(__file__).resolve().parent.parent / "docs" / "data"
METADATA = DATA / "scattnlay_reference_metadata.json"

# impedance of free space, used to put scattnlay's H into miepython's units
ETA0 = 376.730313667

# geometry shared by both reference cases
D_SPHERE = 2.0
LAMBDA0 = 1.0
N_ENV = 1.0
EXTENT = 3.0
NPTS = 121

# scattnlay scalar results, quoted from the metadata file
SCALARS = {
    "nonabs": {
        "m": 1.5 + 0.0j,
        "Qext": 2.351382357157882,
        "Qsca": 2.351382357157882,
        "Qbk": 2.53277025110356,
        "g": 0.583423159613144,
        "max_abs_E": 5.005583970849667,
    },
    "abs": {
        "m": 1.5 - 0.1j,
        "Qext": 2.5837269073859774,
        "Qsca": 1.3566707303067342,
        "Qbk": 0.1707308516638513,
        "g": 0.8266873213103347,
        "max_abs_E": 2.32745076357907,
    },
}

CASES = sorted(SCALARS)

grids_required = pytest.mark.skipif(not METADATA.exists(), reason=f"scattnlay reference grids not found under {DATA}")


def size_parameter():
    """Size parameter shared by the reference cases."""
    return np.pi * D_SPHERE * N_ENV / LAMBDA0


@pytest.mark.parametrize("case", CASES)
def test_efficiencies_match_scattnlay(case):
    """qext, qsca and g agree with an independent implementation."""
    ref = SCALARS[case]
    qext, qsca, qback, g = mie.efficiencies_mx(ref["m"], size_parameter())

    assert qext == pytest.approx(ref["Qext"], rel=1e-10)
    assert qsca == pytest.approx(ref["Qsca"], rel=1e-10)
    assert g == pytest.approx(ref["g"], rel=1e-10)
    # qback sums a series of alternating signs, so it carries the least precision
    assert qback == pytest.approx(ref["Qbk"], rel=1e-8)


@grids_required
def test_metadata_matches_the_inlined_scalars():
    """The inlined numbers must not drift from the metadata they were copied from."""
    meta = json.loads(METADATA.read_text())
    assert meta["d_sphere"] == D_SPHERE
    assert meta["lambda0"] == LAMBDA0
    assert meta["n_env"] == N_ENV
    assert meta["extent"] == EXTENT
    assert meta["npts"] == NPTS
    for case, ref in SCALARS.items():
        got = meta["cases"][case]
        assert complex(got["n_sphere_miepython"]) == ref["m"]
        assert got["max_abs_E"] == ref["max_abs_E"]
        for key in ("Qext", "Qsca", "Qbk", "g"):
            assert got["scattering"][key] == ref[key]


def load_case(case):
    """Return the reference grid and miepython's fields on the same points."""
    grids = {k: np.load(DATA / f"scattnlay_{case}_{k}.npy") for k in ("X", "Z", "E", "H")}
    e_mie, h_mie = eh_near_cartesian(
        LAMBDA0,
        D_SPHERE,
        SCALARS[case]["m"],
        N_ENV,
        grids["X"],
        np.zeros_like(grids["X"]),
        grids["Z"],
        include_incident=True,
    )
    return grids, e_mie, h_mie


def magnitudes(grids, e_mie, h_mie):
    """Field magnitudes for both codes, with scattnlay's H put into our units."""
    e_ref = np.sqrt(np.sum(np.abs(grids["E"]) ** 2, axis=0))
    h_ref = (ETA0 / N_ENV) * np.sqrt(np.sum(np.abs(grids["H"]) ** 2, axis=0))
    e_got = np.sqrt(np.sum(np.abs(e_mie) ** 2, axis=0))
    h_got = np.sqrt(np.sum(np.abs(h_mie) ** 2, axis=0))
    return e_ref, h_ref, e_got, h_got


@grids_required
@pytest.mark.parametrize("case", CASES)
def test_field_magnitude_scale(case):
    """The peak field magnitude matches the reference."""
    grids, e_mie, h_mie = load_case(case)
    e_ref, _, e_got, _ = magnitudes(grids, e_mie, h_mie)
    assert np.max(e_got) == pytest.approx(np.max(e_ref), rel=1e-6)
    assert np.max(e_ref) == pytest.approx(SCALARS[case]["max_abs_E"], rel=1e-9)


@grids_required
@pytest.mark.parametrize("case", CASES)
def test_field_agrees_across_the_grid(case):
    """Fields agree over the whole 121x121 slice, away from the surface.

    The radial E component is discontinuous at the boundary, so grid points that
    land exactly on it disagree by order unity depending on which side each code
    assigns them to.  There are 112 such points out of 14641; they are excluded
    and everything else is compared directly.
    """
    grids, e_mie, h_mie = load_case(case)
    e_ref, h_ref, e_got, h_got = magnitudes(grids, e_mie, h_mie)

    e_err = np.abs(e_got - e_ref) / np.maximum(e_ref, 1e-12)
    h_err = np.abs(h_got - h_ref) / np.maximum(h_ref, 1e-12)

    spacing = 2 * EXTENT / (NPTS - 1)
    radius = D_SPHERE / 2
    r = np.hypot(grids["X"], grids["Z"])
    off_surface = np.abs(r - radius) > spacing / 2
    assert off_surface.sum() > 0.98 * r.size, "excluded far too many points"

    # the medians sit near 3.5e-14.  This bound keeps ample headroom for other
    # platforms while still failing the pre-stability implementation, which
    # managed only 1e-11.  The exact term count is pinned in test_field.py, so
    # this does not need to discriminate one extra order from two.
    assert np.median(e_err) < 1e-12
    assert np.median(h_err) < 5e-9

    assert np.max(e_err[off_surface]) < 2e-3
    assert np.max(h_err[off_surface]) < 2e-3
