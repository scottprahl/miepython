SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make check, make html, or make clean

rcheck:
	make clean
	make notecheck
	make rstcheck
	-pyroma -d .
	-check-manifest
	-pylint miepython/miepython.py
	-pydocstyle miepython/miepython.py
	-pylint miepython/__init__.py
	-pydocstyle miepython/__init__.py

notecheck:
	pytest --verbose -n 4 test_all_notebooks.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodule docs/miepython.rst

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

test:
	tox

clean:
	rm -rf dist
	rm -rf miepython.egg-info
	rm -rf miepython/__pycache__
	rm -rf docs/api
	rm -rf docs/_build
	rm -rf __pycache__
	rm -rf .tox
	rm -rf 04_plot.png

	
realclean:
	make clean

.PHONY: clean rcheck html test realclean