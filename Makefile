PACKAGE         := miepython
GITHUB_USER     := scottprahl

# -------- venv config --------
PY_VERSION      ?= 3.12
VENV            ?= .venv
PY              := /opt/homebrew/opt/python@$(PY_VERSION)/bin/python$(PY_VERSION)
PYTHON          := $(VENV)/bin/python
SERVE_PY        := $(abspath $(PYTHON))
PIP             := $(VENV)/bin/pip
PYPROJECT       := pyproject.toml

BUILD_APPS      := lab
DOCS_DIR        := docs
HTML_DIR        := $(DOCS_DIR)/_build/html

ROOT            := $(abspath .)
PIP_CACHE_DIR   := $(ROOT)/.cache/pip
OUT_ROOT        := $(ROOT)/_site
OUT_DIR         := $(OUT_ROOT)/$(PACKAGE)
STAGE_DIR       := $(ROOT)/.lite_src
DOIT_DB         := $(ROOT)/.jupyterlite.doit.db
LITE_CONFIG     := $(ROOT)/$(PACKAGE)/jupyter_lite_config.json

# --- GitHub Pages deploy config ---
PAGES_BRANCH    := gh-pages
WORKTREE        := .gh-pages
REMOTE          := origin

# --- server config (override on CLI if needed) ---
HOST            := 127.0.0.1
PORT            := 8000

PYTEST          := $(VENV)/bin/pytest
PYLINT          := $(VENV)/bin/pylint
SPHINX          := $(VENV)/bin/sphinx-build
RUFF            := $(VENV)/bin/ruff
BLACK           := $(VENV)/bin/black
CHECKMANIFEST   := $(VENV)/bin/check-manifest
PYROMA          := $(PYTHON) -m pyroma
RSTCHECK        := $(PYTHON) -m rstcheck
YAMLLINT        := $(PYTHON) -m yamllint

PYTEST_OPTS     := -q
SPHINX_OPTS     := -T -E -b html -d $(DOCS_DIR)/_build/doctrees -D language=en
NOTEBOOK_RUN    := $(PYTEST) --verbose tests/all_test_notebooks.py

.PHONY: help
help:
	@echo "Build Targets:"
	@echo "  dist           - Build sdist+wheel locally"
	@echo "  html           - Build Sphinx HTML documentation"
	@echo "  lab            - Start jupyterlab"
	@echo "  speed          - Quick test of jit and no-jit speeds"
	@echo "  readme_images  - Regenerate docs/images/*.svg from examples"
	@echo "  venv           - Create/provision the virtual environment ($(VENV))"
	@echo ""
	@echo "Test Targets:"
	@echo "  test           - Run pytest on python files"
	@echo "  note-test      - Test all notebooks for errors"
	@echo ""
	@echo "Packaging Targets:"
	@echo "  rcheck         - Distribution release checks"
	@echo "  manifest-check - Validate MANIFEST"
	@echo "  pylint-check   - Same as lint above"
	@echo "  pyroma-check   - Validate overall packaging"
	@echo "  rst-check      - Validate all RST files"
	@echo "  ruff-check     - Lint all .py and .ipynb files"
	@echo "  yaml-check     - Validate YAML files"
	@echo ""
	@echo "JupyterLite Targets:"
	@echo "  lite           - Build JupyterLite site into $(OUT_DIR)"
	@echo "  lite-serve     - Serve $(OUT_DIR) at http://$(HOST):$(PORT)"
	@echo "  lite-deploy    - Upload to github"
	@echo ""
	@echo "Clean Targets:"
	@echo "  clean          - Remove build caches and docs output"
	@echo "  lite-clean     - Remove JupyterLite outputs"
	@echo "  realclean      - clean + remove $(VENV)"

# venv bootstrap
$(VENV)/.ready: Makefile $(PYPROJECT)
	@echo "==> Ensuring venv at $(VENV) using $(PY)"
	@if [ ! -x "$(PY)" ]; then \
		echo "❌ Homebrew Python $(PY_VERSION) not found at $(PY)"; \
		echo "   Try: brew install python@$(PY_VERSION)"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV)" ]; then \
		"$(PY)" -m venv "$(VENV)"; \
	fi
	@mkdir -p "$(PIP_CACHE_DIR)"
	@PIP_CACHE_DIR="$(PIP_CACHE_DIR)" $(PYTHON) -m pip -q install --upgrade pip wheel
	@echo "==> Installing miepython + dev extras"
	@PIP_CACHE_DIR="$(PIP_CACHE_DIR)" $(PYTHON) -m pip install -e ".[dev,docs,lite]"
	@touch "$(VENV)/.ready"
	@echo "✅ venv ready"

.PHONY: venv
venv: $(VENV)/.ready
	@:

.PHONY: dist
dist: $(VENV)/.ready
	$(PYTHON) -m build

.PHONY: readme_images
readme_images: $(VENV)/.ready
	@echo "==> Generating README/example SVG images in docs/images"
	@$(PYTHON) docs/images/make_readme_images.py
	
.PHONY: test
test: $(VENV)/.ready
	$(PYTEST) $(PYTEST_OPTS) tests

.PHONY: note-test
note-test: $(VENV)/.ready
	$(PYTEST) --verbose tests/test_all_notebooks.py
	@echo "✅ Notebook check complete"

.PHONY: html
html: $(VENV)/.ready
	@mkdir -p "$(HTML_DIR)"
	$(SPHINX) $(SPHINX_OPTS) "$(DOCS_DIR)" "$(HTML_DIR)"
	@command -v open >/dev/null 2>&1 && open "$(HTML_DIR)/index.html" || true

.PHONY: pylint-check
pylint-check: $(VENV)/.ready
	-@$(PYLINT) miepython/__init__.py
	-@$(PYLINT) miepython/bessel.py
	-@$(PYLINT) miepython/core.py
	-@$(PYLINT) miepython/mie_jit.py
	-@$(PYLINT) miepython/mie_nojit.py
	-@$(PYLINT) miepython/monte_carlo.py
	-@$(PYLINT) miepython/rayleigh.py
	-@$(PYLINT) miepython/util.py
	-@$(PYLINT) miepython/vsh.py
	-@$(PYLINT) tests/test_all_examples.py
	-@$(PYLINT) tests/test_all_notebooks.py
	-@$(PYLINT) tests/test_jit.py
	-@$(PYLINT) tests/test_nojit.py
	-@$(PYLINT) docs/conf.py
	-@$(PYLINT) .github/scripts/update_citation.py

.PHONY: yaml-check
yaml-check: $(VENV)/.ready
	-@$(PYTHON) -m yamllint .github/workflows/citation.yaml
	-@$(PYTHON) -m yamllint .github/workflows/pypi.yaml
	-@$(PYTHON) -m yamllint .github/workflows/test.yaml
	-@$(PYTHON) -m yamllint .readthedocs.yaml

.PHONY: rst-check
rst-check: $(VENV)/.ready
	-@$(RSTCHECK) README.rst
	-@$(RSTCHECK) CHANGELOG.rst
	-@$(RSTCHECK) $(DOCS_DIR)/index.rst
	-@$(RSTCHECK) $(DOCS_DIR)/changelog.rst
	-@$(RSTCHECK) --ignore-directives automodapi $(DOCS_DIR)/$(PACKAGE).rst

.PHONY: ruff-check
ruff-check: $(VENV)/.ready
	$(RUFF) check

.PHONY: manifest-check
manifest-check: $(VENV)/.ready
	$(CHECKMANIFEST)

.PHONY: pyroma-check
pyroma-check: $(VENV)/.ready
	$(PYROMA) -d .

.PHONY: rcheck
rcheck:
	@echo "Running all release checks..."
	@$(MAKE) realclean
	@$(MAKE) ruff-check
	@$(MAKE) pylint-check
	@$(MAKE) rst-check
	@$(MAKE) manifest-check
	@$(MAKE) pyroma-check
	@$(MAKE) html
	@$(MAKE) lite
	@$(MAKE) dist
	@$(MAKE) test
	@$(MAKE) note-test
	@echo "✅ Release checks complete"
	
.PHONY: lite
lite: $(VENV)/.ready $(LITE_CONFIG)
	@echo "==> Building package wheel for PyOdide"
	@$(PYTHON) -m build

	@echo "==> Checking for .gh-pages worktree"
	@if [ -d "$(WORKTREE)" ]; then \
		echo "    Found .gh-pages worktree, removing..."; \
		git worktree remove "$(WORKTREE)" --force 2>/dev/null || true; \
		git worktree prune; \
		rm -rf "$(WORKTREE)"; \
		echo "    ✓ Removed"; \
	else \
		echo "    No .gh-pages worktree found"; \
	fi

	@echo "==> Cleaning previous builds"
	@/bin/rm -rf "$(OUT_ROOT)"
	@/bin/rm -rf "$(DOIT_DB)"
	@/bin/rm -rf ".doit.db"
	@/bin/rm -rf ".jupyterlite.doit.db.db"
	@echo "    ✓ Cleaned"

	@echo "==> Staging notebooks from docs -> $(STAGE_DIR)"
	@/bin/rm -rf "$(STAGE_DIR)"; mkdir -p "$(STAGE_DIR)"
	@if ls docs/*.ipynb 1> /dev/null 2>&1; then \
		/bin/cp docs/*.ipynb "$(STAGE_DIR)"; \
		/bin/rm $(STAGE_DIR)/x_*.ipynb; \
		mkdir -p "$(STAGE_DIR)/examples"; \
		/bin/cp $(PACKAGE)/examples/*.py "$(STAGE_DIR)/examples"; \
		if ls docs/data/*.npy 1> /dev/null 2>&1; then \
			echo "==> Staging near-field reference arrays into $(STAGE_DIR)/data"; \
			mkdir -p "$(STAGE_DIR)/data"; \
			/bin/cp docs/data/*.npy "$(STAGE_DIR)/data"; \
		else \
			echo "⚠️  No docs/data/*.npy files found (15_2D_fields.ipynb may fail)"; \
		fi; \
		if [ -f docs/data/scattnlay_reference_metadata.json ]; then \
			/bin/cp docs/data/scattnlay_reference_metadata.json "$(STAGE_DIR)/data"; \
		fi; \
		echo "==> Clearing outputs from staged notebooks"; \
		"$(PYTHON)" -m jupyter nbconvert --clear-output --inplace "$(STAGE_DIR)"/*.ipynb; \
	else \
		echo "⚠️  No notebooks found in docs/"; \
	fi

	@echo "==> Building JupyterLite"
	@"$(PYTHON)" -m jupyter lite build \
		--config="$(LITE_CONFIG)" \
		--contents="$(STAGE_DIR)" \
		--output-dir="$(OUT_DIR)"

	@echo "==> Adding .nojekyll for GitHub Pages"
	@touch "$(OUT_DIR)/.nojekyll"
	
	@echo "✅ Build complete -> $(OUT_DIR)"

.PHONY: lite-serve
lite-serve: $(VENV)/.ready
	@test -d "$(OUT_DIR)" || { echo "❌ run 'make lite' first"; exit 1; }
	@echo "Serving at"
	@echo "   http://$(HOST):$(PORT)/$(PACKAGE)/?disableCache=1"
	@echo ""
	"$(PYTHON)" -m http.server -d "$(OUT_ROOT)" --bind $(HOST) $(PORT)

.PHONY: lite-deploy
lite-deploy: 
	@echo "==> Sanity check"
	@test -d "$(OUT_DIR)" || { echo "❌ Run 'make lite' first"; exit 1; }

	@echo "==> Ensure $(PAGES_BRANCH) branch exists"
	@if ! git show-ref --verify --quiet refs/heads/$(PAGES_BRANCH); then \
	  CURRENT=$$(git branch --show-current); \
	  git switch --orphan $(PAGES_BRANCH); \
	  git commit --allow-empty -m "Initialize $(PAGES_BRANCH)"; \
	  git switch $$CURRENT; \
	fi

	@echo "==> Setup deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@git worktree prune || true
	@rm -rf "$(WORKTREE)"
	@git worktree add "$(WORKTREE)" "$(PAGES_BRANCH)"
	@git -C "$(WORKTREE)" pull "$(REMOTE)" "$(PAGES_BRANCH)" 2>/dev/null || true

	@echo "==> Deploy $(OUT_DIR) -> $(WORKTREE)"
	@rsync -a --delete --exclude ".git*" "$(OUT_DIR)/" "$(WORKTREE)/"
	@touch "$(WORKTREE)/.nojekyll"
	@date -u +"%Y-%m-%d %H:%M:%S UTC" > "$(WORKTREE)/.pages-ping"

	@echo "==> Commit & push"
	@cd "$(WORKTREE)" && \
	  git add -A && \
	  if git diff --quiet --cached; then \
	    echo "✅ No changes to deploy"; \
	  else \
	    git commit -m "Deploy $$(date -u +'%Y-%m-%d %H:%M:%S UTC')" && \
	    git push "$(REMOTE)" "$(PAGES_BRANCH)" && \
	    echo "✅ Deployed to https://$(GITHUB_USER).github.io/$(PACKAGE)/"; \
	  fi

.PHONY: kernelspec
kernelspec: $(VENV)/.ready
	@$(PYTHON) -m ipykernel install --user \
	  --name miepython-venv \
	  --display-name "Python (miepython venv)" >/dev/null

.PHONY: lab
lab: kernelspec
	@echo "==> Launching JupyterLab using venv ($(PYTHON))"
	@$(PYTHON) -m jupyter lab --ServerApp.root_dir="$(CURDIR)"

.PHONY: speed
speed:
	-python tests/test_nojit_speed.py
	-python tests/test_jit_speed.py

.PHONY: clean
clean:
	@echo "==> Cleaning build artifacts"	
	@find . -name '__pycache__' -type d -exec rm -rf {} +
	@find . -name '.DS_Store' -type f -delete
	@find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
	@find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@/bin/rm -rf .cache
	@/bin/rm -rf .ruff_cache
	@/bin/rm -rf $(PACKAGE).egg-info
	@/bin/rm -rf docs/api
	@/bin/rm -rf docs/_build
	@/bin/rm -rf docs/.jupyter
	@/bin/rm -rf dist

.PHONY: lite-clean
lite-clean:
	@echo "==> Cleaning JupyterLite build artifacts"
	@/bin/rm -rf "$(STAGE_DIR)"
	@/bin/rm -rf "$(OUT_ROOT)"
	@/bin/rm -rf ".lite_root"
	@/bin/rm -rf "$(DOIT_DB)"
	@/bin/rm -rf "_output"
	@/bin/rm -rf "_site"

.PHONY: realclean
realclean: lite-clean clean
	@echo "==> Deep cleaning: removing venv and deployment worktree"
#	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@/bin/rm -rf .cache
	@/bin/rm -rf .tmp
	@/bin/rm -rf "$(WORKTREE)"
	@/bin/rm -rf "$(VENV)"
	@/bin/rm -rf "docs/api"
	@/bin/rm -rf "docs/_templates"
