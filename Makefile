SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make check, make html, or make clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

notecheck:
	make clean
	pytest --verbose test_all_notebooks.py
	rm -rf __pycache__

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodule docs/miepython.rst

lintcheck:
	-pylint miepython/miepython.py
	-pylint miepython/__init__.py

doccheck:
	-pydocstyle miepython/miepython.py
	-pydocstyle miepython/__init__.py

rcheck:
	make notecheck
	make rstcheck
	make lintcheck
	make doccheck
	-pyroma -d .
	-check-manifest

jittest:
	python3 -m pytest test.py
	python3 -m pytest test_jit.py

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

.PHONY: clean html test realclean \
        rcheck doccheck lintcheck rstcheck notecheck