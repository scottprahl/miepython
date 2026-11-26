"""Run all the scripts in the examples directory."""

import importlib
import time
import sys
import pathlib

import pytest
import matplotlib

# The examples call plt.show().  Without a non-interactive backend matplotlib
# opens real Tk windows, and on the Windows runners Tcl/Tk intermittently fails
# to start ("invalid command name tcl_findLibrary"), failing a random example.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

# anchored on this file, not the working directory, so a run started from
# somewhere else cannot silently collect zero examples
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
examples = sorted((REPO_ROOT / "miepython" / "examples").glob("*.py"))
assert examples, f"no example scripts found under {REPO_ROOT / 'miepython' / 'examples'}"
ids = [p.name for p in examples]


@pytest.mark.parametrize("path", examples, ids=ids)
def test_example_runs(path):
    """Test each example script."""
    sys.path.append(str(path.parent))
    try:
        importlib.import_module(path.stem)
    finally:
        plt.close("all")
    time.sleep(0.2)
