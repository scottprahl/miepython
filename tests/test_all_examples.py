
from pathlib import Path

import pytest


examples = list(Path("miepython/examples").glob("*.py"))
ids = [p.as_posix() for p in examples]

@pytest.mark.parametrize("path", examples, ids=ids)
def test_example_runs(path):
    import importlib
    import time
    import sys

    import matplotlib.pyplot as plt

    sys.path.append(str(path.parent))

    importlib.import_module(path.stem)
    time.sleep(0.2)
