#!/usr/bin/env python3
"""Generate SVG images from example scripts for README/docs.

This runs each numbered example script in ``miepython/examples`` and saves the
last matplotlib figure as ``docs/images/NN.svg`` where ``NN`` is the two-digit
prefix from the example filename (for example, ``01_dielectric.py`` -> ``01.svg``).
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXAMPLES_DIR = ROOT / "miepython" / "examples"
IMAGES_DIR = ROOT / "docs" / "images"


def _numbered_examples() -> list[Path]:
    """Return sorted numbered example scripts."""
    return sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))


def _save_last_figure(svg_path: Path) -> None:
    """Save the latest matplotlib figure to SVG."""
    fig_numbers = plt.get_fignums()
    if not fig_numbers:
        raise RuntimeError(f"No matplotlib figure produced for {svg_path.name}")

    fig = plt.figure(fig_numbers[-1])
    fig.savefig(svg_path, format="svg", bbox_inches="tight")


def main() -> None:
    """Run example scripts and write SVG images into docs/images."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    scripts = _numbered_examples()
    if not scripts:
        raise RuntimeError(f"No numbered examples found in {EXAMPLES_DIR}")

    for script in scripts:
        prefix = script.stem.split("_", 1)[0]
        out_svg = IMAGES_DIR / f"{prefix}.svg"

        plt.close("all")
        runpy.run_path(str(script), run_name="__main__")
        _save_last_figure(out_svg)
        print(f"Wrote {out_svg}")

    plt.close("all")


if __name__ == "__main__":
    main()

