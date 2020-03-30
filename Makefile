SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

default:
	@echo Type: make check, make html, or make clean

check:
	-pyroma -d .
	-check-manifest
	-pylint miepython/miepython.py
	-pep257 miepython/miepython.py

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

test:
	tox

clean:
	rm -rf dist
	rm -rf miepython.egg-info
	rm -rf miepython/__pycache__
	rm -rf docs/_build/*
	rm -rf docs/api/*
	rm -rf .tox
	
.PHONY: clean check html test