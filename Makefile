SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make rcheck, make html, or make clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

notecheck:
	make clean
	python -m pytest --verbose --notebooks tests/test_all_notebooks.py
	rm -rf __pycache__

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck --ignore-directives automodapi docs/miepython.rst

lintcheck:
	-pylint miepython/miepython.py
	-pylint miepython/miepython_nojit.py
	-pylint miepython/__init__.py
	-pylint tests/test_all_examples.py
	-pylint tests/test_all_notebooks.py
	-pylint tests/test_jit.py
	-pylint tests/test_nojit.py
	-pylint docs/conf.py

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
	- flake8 .
	pyroma -d .
	check-manifest
	make html
	make test

test:
	python -m pytest tests/test_nojit.py
	python -m pytest tests/test_jit.py
	python -m pytest tests/test_all_examples.py

clean:
	rm -rf dist
	rm -rf miepython.egg-info
	rm -rf miepython/__pycache__
	rm -rf miepython/examples/__pycache__
	rm -rf docs/.DS_Store
	rm -rf docs/api
	rm -rf docs/_build
	rm -rf docs/.ipynb_checkpoints
	rm -rf docs/.pytest_cache
	rm -rf tests/__pycache__
	rm -rf 04_plot.png
	rm -rf .pytest_cache
	rm -rf build

realclean:
	make clean

.PHONY: clean html test realclean \
        rcheck doccheck lintcheck rstcheck notecheck