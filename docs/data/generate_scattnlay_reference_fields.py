#!/usr/bin/env python3
"""Generate scattnlay near-field reference arrays for notebook validation.

This script uses the Python bindings from scattnlay (`fieldnlay`, `scattnlay`)
for two homogeneous-sphere cases and saves `.npy` outputs under `docs/data/`:

- nonabs: m_scatt = 1.5 + 0.0j
- abs:    m_scatt = 1.5 + 0.1j  (mapped to physical 1.5 - 0.1j in miepython)

Saved files per case:
- scattnlay_<case>_X.npy
- scattnlay_<case>_Z.npy
- scattnlay_<case>_E.npy  (shape: 3 x nx x nz)
- scattnlay_<case>_H.npy  (shape: 3 x nx x nz)
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from scattnlay import fieldnlay, scattnlay


SCATTNLAY_COMMIT = "211fc8d2168deff500b67042cea952fc74c84b64"


@contextlib.contextmanager
def _suppress_native_output() -> None:
    """Suppress verbose stdout/stderr emitted by the C++ extension."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _reshape_field_components(raw: np.ndarray, npts: int) -> np.ndarray:
    """Return field components as (3, npts, npts)."""
    if raw.shape == (npts * npts, 3):
        return raw.T.reshape(3, npts, npts, order="C")
    if raw.shape == (3, npts * npts):
        return raw.reshape(3, npts, npts, order="C")
    raise RuntimeError(f"Unexpected field array shape from scattnlay: {raw.shape}")


def run_fieldnlay_case(
    d_sphere: float,
    n_env: float,
    lambda0: float,
    m_scatt: complex,
    extent: float,
    npts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Run fieldnlay for one case and return X,Z,E,H on physical coordinates."""
    x_size_parameter = np.pi * d_sphere * n_env / lambda0
    k_env = 2.0 * np.pi * n_env / lambda0

    # scattnlay field coordinates are dimensionless and consistent with x = k*r.
    lo_kr = -extent * k_env
    hi_kr = extent * k_env

    x_scan = np.linspace(lo_kr, hi_kr, npts)
    z_scan = np.linspace(lo_kr, hi_kr, npts)
    Xkr, Zkr = np.meshgrid(x_scan, z_scan, indexing="ij")
    Ykr = np.zeros_like(Xkr)

    xp = Xkr.ravel(order="C")
    yp = Ykr.ravel(order="C")
    zp = Zkr.ravel(order="C")

    x_layers = np.array([x_size_parameter], dtype=float)
    m_layers = np.array([m_scatt], dtype=complex)

    with _suppress_native_output():
        terms, e_raw, h_raw = fieldnlay(x_layers, m_layers, xp, yp, zp)
        (
            terms_scatt,
            qext,
            qsca,
            qabs,
            qbk,
            qpr,
            asymmetry,
            albedo,
            _s1,
            _s2,
        ) = scattnlay(x_layers, m_layers)

    E = _reshape_field_components(np.asarray(e_raw), npts)
    H = _reshape_field_components(np.asarray(h_raw), npts)

    X = Xkr / k_env
    Z = Zkr / k_env

    scattering = {
        "terms_fieldnlay": int(terms),
        "terms_scattnlay": int(terms_scatt),
        "Qext": float(qext),
        "Qsca": float(qsca),
        "Qabs": float(qabs),
        "Qbk": float(qbk),
        "Qpr": float(qpr),
        "g": float(asymmetry),
        "albedo": float(albedo),
    }
    return X, Z, E, H, scattering


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)

    d_sphere = 2.0
    n_env = 1.0
    lambda0 = 1.0
    extent = 3.0
    npts = 121

    cases = {
        "nonabs": {
            "m_scatt": 1.5 + 0.0j,
            "n_sphere_physical": "1.5+0.0j",
            "n_sphere_miepython": "1.5+0.0j",
        },
        "abs": {
            "m_scatt": 1.5 + 0.1j,
            "n_sphere_physical": "1.5-0.1j",
            "n_sphere_miepython": "1.5-0.1j",
        },
    }

    metadata = {
        "scattnlay_python_package": importlib.metadata.version("scattnlay"),
        "d_sphere": d_sphere,
        "n_env": n_env,
        "lambda0": lambda0,
        "extent": extent,
        "npts": npts,
        "cases": {},
        "scattnlay_commit": SCATTNLAY_COMMIT,
    }

    for case, cfg in cases.items():
        X, Z, E, H, scattering = run_fieldnlay_case(
            d_sphere=d_sphere,
            n_env=n_env,
            lambda0=lambda0,
            m_scatt=cfg["m_scatt"],
            extent=extent,
            npts=npts,
        )

        fx = f"scattnlay_{case}_X.npy"
        fz = f"scattnlay_{case}_Z.npy"
        fe = f"scattnlay_{case}_E.npy"
        fh = f"scattnlay_{case}_H.npy"

        np.save(data_dir / fx, X)
        np.save(data_dir / fz, Z)
        np.save(data_dir / fe, E)
        np.save(data_dir / fh, H)

        e_abs = np.sqrt(np.sum(np.abs(E) ** 2, axis=0))
        h_abs = np.sqrt(np.sum(np.abs(H) ** 2, axis=0))
        metadata["cases"][case] = {
            "scattering": scattering,
            "n_sphere_scatt": [cfg["m_scatt"].real, cfg["m_scatt"].imag],
            "n_sphere_physical": cfg["n_sphere_physical"],
            "n_sphere_miepython": cfg["n_sphere_miepython"],
            "max_abs_E": float(np.max(e_abs)),
            "max_abs_H_raw": float(np.max(h_abs)),
            "files": {"X": fx, "Z": fz, "E": fe, "H": fh},
        }

        print(
            f"[{case}] wrote {fx}, {fz}, {fe}, {fh}; "
            f"max|E|={np.max(e_abs):.6g}, max|H|={np.max(h_abs):.6g}"
        )

    meta_path = data_dir / "scattnlay_reference_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
