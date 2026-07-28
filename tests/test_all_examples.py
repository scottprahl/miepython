"""Run all the scripts in the examples directory."""

import importlib
import time
import sys
import pathlib

import pytest

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
    importlib.import_module(path.stem)
    time.sleep(0.2)
