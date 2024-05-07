"""Run all the scripts in the examples directory."""

import importlib
import sys
import pathlib

from unittest.mock import patch
import matplotlib.pyplot as plt
import pytest

examples = list(pathlib.Path("miepython/examples").glob("*.py"))
ids = [example.as_posix() for example in examples]


@pytest.mark.parametrize("path", examples, ids=ids)
def test_example_runs(path):
    """Test all examples."""
    with patch.object(plt, 'show'):
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
