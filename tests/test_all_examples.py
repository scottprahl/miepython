# pylint: disable=invalid-name
"""Run all the scripts in the examples directory."""
import importlib
import time
import sys
import pathlib

import pytest

examples = list(pathlib.Path("miepython/examples").glob("*.py"))
ids = [p.as_posix() for p in examples]

@pytest.mark.parametrize("path", examples, ids=ids)
def test_example_runs(path):
    """Test each example script."""
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    time.sleep(0.2)
