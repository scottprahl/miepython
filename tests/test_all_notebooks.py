"""
This file is intended to be the target of a pytest run.

It will recursively find all .ipynb files in the ../docs directory, ignoring
directories that start with . and any files matching patterins found in the file
.testignore

List patterns to skip in .testignore file:

    under_construction/*

Sample invocations of pytest which make the output nicely readable:

    pytest --verbose --durations=5 test_all_notebooks.py

If you install pytest-xdist you can run tests in parallel with

    pytest --verbose --durations=5 -n 4 test_all_notebooks.py

Original version is licensed under GPL 3.0 so this one is too.
The original can be located at
    https://github.com/alchemyst/Dynamics-and-Control/test_all_notebooks.py
"""
import os.path
import pathlib
import pytest
import nbformat
import nbconvert.preprocessors

# Default search path is the current directory
# searchpath = pathlib.Path('.')
searchpath = pathlib.Path('docs/')  # all notebooks are in here

# Read patterns from .testignore file
ignores = ''
if os.path.exists('.testignore'):
    with open('.testignore', encoding='utf-8') as ff:
        ignores = [line.strip() for line in ff if line.strip()]

# Ignore hidden folders (startswith('.')) and files matching ignore patterns
notebooks = [notebook for notebook in searchpath.glob('*.ipynb')
             if not (any(parent.startswith('.')
                         for parent in notebook.parent.parts)
                     or any(notebook.match(pattern)
                            for pattern in ignores))]

notebooks.sort()
print(notebooks)

ids = [n.as_posix() for n in notebooks]

for n in notebooks:
    print(n)


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", notebooks, ids=ids)
def test_run_notebook(notebook):
    """Read and execute notebook.

    The method here is directly from the nbconvert docs

    Note that there is no error handling in this file as any errors will be
    caught by pytest

    """
    with open(notebook, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': notebook.parent}})
