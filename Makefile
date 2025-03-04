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
	-pylint miepython/__init__.py
	-pylint miepython/bessel.py
	-pylint miepython/mie_jit.py
	-pylint miepython/mie_nojit.py
	-pylint miepython/core.py
	-pylint miepython/monte_carlo.py
	-pylint miepython/util.py
	-pylint miepython/vsh.py
	-pylint tests/test_all_examples.py
	-pylint tests/test_all_notebooks.py
	-pylint tests/test_jit.py
	-pylint tests/test_nojit.py
	-pylint docs/conf.py

doccheck:
	-ruff check miepython/__init__.py
	-ruff check miepython/bessel.py
	-ruff check miepython/core.py
	-ruff check miepython/mie_jit.py
	-ruff check miepython/mie_nojit.py
	-ruff check miepython/monte_carlo.py
	-ruff check miepython/util.py
	-ruff check miepython/vsh.py
	-ruff check tests/test_all_examples.py
	-ruff check tests/test_bessel.py
	-ruff check tests/test_jit.py
	-ruff check tests/test_jit_D.py
	-ruff check tests/test_jit_abcd.py
	-ruff check tests/test_nojit.py
	-ruff check tests/test_nojit_D.py
	-ruff check tests/test_nojit_abcd.py
	-ruff check tests/test_vsh_basic.py
	-ruff check tests/test_all_notebooks.py

rcheck:
	make notecheck
	make rstcheck
	make lintcheck
	make doccheck
	ruff check .
	pyroma -d .
	check-manifest
	make html
	make test

test:
	-pytest -v tests/test_bessel.py
	-pytest -v tests/test_rayleigh.py

	-pytest -v tests/test_nojit_D.py
	-pytest -v tests/test_nojit_abcd.py
	-pytest -v tests/test_nojit.py

	-pytest -v tests/test_jit_D.py
	-pytest -v tests/test_jit_abcd.py
	-pytest -v tests/test_jit.py
	-pytest -v tests/test_vsh_basic.py
	-pytest -v tests/test_field.py

	-pytest -v tests/test_all_examples.py
	-pytest -v --notebooks tests/test_all_notebooks.py

clean:
	rm -rf .ruff_cache
	rm -rf miepython.egg-info
	rm -rf miepython/__pycache__
	rm -rf miepython/.ipynb_checkpoints
	rm -rf miepython/examples/__pycache__
	rm -rf docs/.DS_Store
	rm -rf docs/api
	rm -rf docs/_build
	rm -rf docs/.ipynb_checkpoints
	rm -rf docs/.pytest_cache
	rm -rf tests/__pycache__
	rm -rf tests/.ipynb_checkpoints
	rm -rf 04_plot.png
	rm -rf .pytest_cache
	rm -rf build

realclean:
	make clean

.PHONY: clean html test realclean \
        rcheck doccheck lintcheck rstcheck notecheck