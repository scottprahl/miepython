SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make check, make html, or make clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

notecheck:
	make clean
	pytest --verbose --notebooks tests/test_all_notebooks.py
	rm -rf __pycache__

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodule docs/miepython.rst

lintcheck:
	-pylint miepython/miepython.py
	-pylint miepython/miepython_nojit.py
	-pylint miepython/__init__.py
	-pylint tests/test_all_examples.py
	-pylint tests/test_all_notebooks.py
	-pylint tests/test_jit.py
	-pylint tests/test_nojit.py

doccheck:
	-pydocstyle miepython/miepython.py
	-pydocstyle miepython/miepython_nojit.py
	-pydocstyle miepython/__init__.py
	-pydocstyle tests/test_all_examples.py
	-pydocstyle tests/test_all_notebooks.py
	-pydocstyle --ignore D100,D101,D102 tests/test_jit.py
	-pydocstyle --ignore D100,D101,D102 tests/test_nojit.py

rcheck:
	make notecheck
	make rstcheck
	make lintcheck
	make doccheck
	-pyroma -d .
	-check-manifest
	make jittest
	make html


jittest:
	python3 -m pytest tests/test_nojit.py
	python3 -m pytest tests/test_jit.py

test:
	tox

clean:
	rm -rf dist
	rm -rf miepython.egg-info
	rm -rf miepython/__pycache__
	rm -rf docs/api
	rm -rf docs/_build
	rm -rf tests/__pycache__
	rm -rf .tox
	rm -rf 04_plot.png
	rm -rf .pytest_cache

realclean:
	make clean

.PHONY: clean html test realclean \
        rcheck doccheck lintcheck rstcheck notecheck