"""
This script will test all the jupyter notebooks..

    pytest --verbose tests/test_all_notebooks.py

It will recursively find all .ipynb files in the ../docs directory, ignoring
directories starting with . and any files matching patterns found in .testignore

Original script is GPL 3.0 licensed so this one is too. See
    https://github.com/alchemyst/Dynamics-and-Control/test_all_notebooks.py
"""

import pathlib
import pytest
import nbformat
import nbconvert.preprocessors

# Default search path is the current directory
# searchpath = pathlib.Path('.')
# anchored on this file rather than the working directory
searchpath = pathlib.Path(__file__).resolve().parent.parent / "docs"

# Read patterns from .testignore file
ignores = ""
testignore = searchpath.parent / ".testignore"
if testignore.exists():
    with open(testignore, encoding="utf-8") as ff:
        ignores = [line.strip() for line in ff if line.strip()]

# Ignore hidden folders (startswith('.')) and files matching ignore patterns
notebooks = [
    notebook
    for notebook in searchpath.glob("*.ipynb")
    if not (
        any(parent.startswith(".") for parent in notebook.parent.parts)
        or any(notebook.match(pattern) for pattern in ignores)
    )
]

notebooks.sort()
print(notebooks)

ids = [n.as_posix() for n in notebooks]

for n in notebooks:
    print(n)


@pytest.mark.parametrize("notebook", notebooks, ids=ids)
def test_run_notebook(notebook):
    """Read and execute notebook.

    The method here is directly from the nbconvert docs

    Note that there is no error handling in this file as any errors will be
    caught by pytest

    """
    with open(notebook, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": notebook.parent}})
