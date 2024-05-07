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

lint:
	-pylint miepython/miepython.py
	-pylint miepython/miepython_nojit.py
	-pylint miepython/__init__.py
	-pylint tests/test_all_examples.py
	-pylint tests/test_all_notebooks.py
	-pylint tests/test_mie.py
	-pylint docs/conf.py

rcheck:
	make clean
	ruff check
	make rstcheck
	make lint
	make html
	make test
	make notecheck
	pyroma -d .
	check-manifest

test:
	python -m pytest tests/test_mie.py
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
	rm -rf tests/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf build

.PHONY: clean html test rcheck lint rstcheck notecheck
