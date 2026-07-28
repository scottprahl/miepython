PACKAGE         := miepython
GITHUB_USER     := scottprahl

PY_VERSION      ?= 3.14
VENV            ?= .venv
UV              ?= uv
RUN             := $(UV) run --extra dev
RUN_DOCS        := $(UV) run --extra docs
RUN_LITE        := $(UV) run --extra lite
RM              ?= rm -f
RMR             ?= rm -rf

DOCS_DIR        := docs
HTML_DIR        := $(DOCS_DIR)/_build/html
OUT_ROOT        := _site
OUT_DIR         := $(OUT_ROOT)/$(PACKAGE)
STAGE_DIR       := .lite_src
DOIT_DB         := .jupyterlite.doit.db
LITE_CONFIG     := $(PACKAGE)/jupyter_lite_config.json

# --- GitHub Pages deploy config ---
PAGES_BRANCH    := gh-pages
WORKTREE        := .gh-pages
REMOTE          := origin

# --- server config (override on CLI if needed) ---
HOST            := 127.0.0.1
PORT            := 8000

PYTEST_OPTS     := 
SPHINX_OPTS     := -T -E -b html -d $(DOCS_DIR)/_build/doctrees -D language=en
PYLINT_TARGETS  := $(PACKAGE)/*.py tests/*.py .github/scripts/update_citation.py
YAML_TARGETS    := .github/workflows/citation.yaml .github/workflows/pypi.yaml .github/workflows/test.yaml .readthedocs.yaml
RST_TARGETS     := README.rst CHANGELOG.rst $(DOCS_DIR)/changelog.rst $(DOCS_DIR)/index.rst

.PHONY: help
help:
	@echo "Build Targets:"
	@echo "  dist           - Build sdist+wheel locally"
	@echo "  html           - Build Sphinx HTML documentation"
	@echo "  lab            - Start jupyterlab"
	@echo "  speed          - Quick test of jit and no-jit speeds"
	@echo "  readme_images  - Regenerate docs/images/*.svg from examples"
	@echo "  sync           - Sync uv environment with dev/docs/lite extras"
	@echo ""
	@echo "Test Targets:"
	@echo "  test           - Run pytest once per backend (no-JIT then JIT)"
	@echo "  test-nojit     - Run the suite with the pure python backend"
	@echo "  test-jit       - Run the suite with the numba backend"
	@echo "  coverage       - Both backends, combined coverage report"
	@echo "  note-test      - Test all notebooks for errors"
	@echo ""
	@echo "Lint Targets:"
	@echo "  lint           - Run every static check below"
	@echo ""
	@echo "Packaging Targets:"
	@echo "  rcheck         - Distribution release checks"
	@echo "  manifest-check - Validate MANIFEST"
	@echo "  pylint-check   - Same as lint above"
	@echo "  pyroma-check   - Validate overall packaging"
	@echo "  black-check    - Check formatting with black"
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

.PHONY: venv
venv:
	@$(UV) sync --python $(PY_VERSION) --extra dev --extra docs --extra lite

.PHONY: dist
dist:
	$(RUN) python -m build

.PHONY: readme_images
readme:
	cd docs/images && $(RUN) python make_readme_images.py

# The backend is chosen when miepython is first imported, so one pytest process
# can only exercise one of them.  Run the suite once per backend; conftest.py
# skips the files belonging to the other one.
.PHONY: test
test: test-nojit test-jit

.PHONY: test-nojit
test-nojit:
	MIEPYTHON_USE_JIT=0 $(RUN) pytest $(PYTEST_OPTS) tests --ignore=tests/test_all_notebooks.py

.PHONY: test-jit
test-jit:
	MIEPYTHON_USE_JIT=1 $(RUN) pytest $(PYTEST_OPTS) tests --ignore=tests/test_all_notebooks.py

# Coverage needs both backends, and NUMBA_DISABLE_JIT lets coverage.py see inside
# the njit bodies: without it mie_jit.py reports 11% however well it is tested.
.PHONY: coverage
coverage:
	@$(RM) .coverage
	MIEPYTHON_USE_JIT=0 $(RUN) pytest $(PYTEST_OPTS) tests \
	    --cov --cov-report= --cov-fail-under=0
	MIEPYTHON_USE_JIT=1 NUMBA_DISABLE_JIT=1 $(RUN) pytest $(PYTEST_OPTS) tests \
	    --cov --cov-append --cov-report= --cov-fail-under=0
	@$(RUN) coverage report
	@$(RUN) coverage html
	@echo "HTML report in htmlcov/index.html"

.PHONY: note-test
note-test:
	$(RUN) pytest --verbose tests/test_all_notebooks.py

.PHONY: html
html:
	@mkdir -p "$(HTML_DIR)"
	$(RUN_DOCS) sphinx-build $(SPHINX_OPTS) "$(DOCS_DIR)" "$(HTML_DIR)"
	@command -v open >/dev/null 2>&1 && open "$(HTML_DIR)/index.html" || true

# Every static check, in one target so CI and `make rcheck` cannot drift apart.
.PHONY: lint
lint:
	@$(MAKE) ruff-check
	@$(MAKE) black-check
	@$(MAKE) pylint-check
	@$(MAKE) rst-check
	@$(MAKE) yaml-check
	@$(MAKE) manifest-check
	@$(MAKE) pyroma-check
	@echo "✅ Lint checks complete"

.PHONY: pylint-check
pylint-check:
	@$(RUN) pylint $(PYLINT_TARGETS)

.PHONY: yaml-check
yaml-check:
	@$(RUN) yamllint $(YAML_TARGETS)

.PHONY: rst-check
rst-check:
	@$(RUN) rstcheck $(RST_TARGETS)
	@$(RUN) rstcheck --ignore-directives automodapi $(DOCS_DIR)/$(PACKAGE).rst

.PHONY: ruff-check
ruff-check:
	@$(RUN) ruff check

.PHONY: black-check
black-check:
	$(RUN) black --check .

.PHONY: manifest-check
manifest-check:
	$(RUN) check-manifest

.PHONY: pyroma-check
pyroma-check:
	$(RUN) pyroma -d .

.PHONY: rcheck
rcheck:
	@echo "Running all release checks..."
	@$(MAKE) realclean
	@$(MAKE) lint
	@$(MAKE) html
	@$(MAKE) lite
	@$(MAKE) dist
	@$(MAKE) test
	@$(MAKE) note-test
	@echo "✅ Release checks complete"
	
.PHONY: lite
lite: lite-clean $(LITE_CONFIG) dist
	@echo "==> Staging notebooks from docs -> $(STAGE_DIR)"
	mkdir -p "$(STAGE_DIR)"
	cp $(DOCS_DIR)/*.ipynb "$(STAGE_DIR)"
	$(RUN) python -m jupyter nbconvert --clear-output --inplace "$(STAGE_DIR)"/*.ipynb
	mkdir -p "$(STAGE_DIR)/examples"
	cp $(PACKAGE)/examples/*.py "$(STAGE_DIR)/examples"
	mkdir -p "$(STAGE_DIR)/data"
	cp docs/data/*.npy "$(STAGE_DIR)/data"
	cp docs/data/scattnlay_reference_metadata.json "$(STAGE_DIR)/data"

	@echo "==> Building JupyterLite"
	@$(RUN_LITE) jupyter lite build \
		--config="$(LITE_CONFIG)" \
		--contents="$(STAGE_DIR)" \
		--output-dir="$(OUT_DIR)"
	@touch "$(OUT_DIR)/.nojekyll"  # for GitHub pages

.PHONY: lite-serve
lite-serve:
	@test -d "$(OUT_DIR)" || { echo "❌ run 'make lite' first"; exit 1; }
	@echo "Serving at"
	@echo "   http://$(HOST):$(PORT)/$(PACKAGE)/?disableCache=1"
	@echo ""
	$(RUN) python -m http.server -d "$(OUT_ROOT)" --bind $(HOST) $(PORT)

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
	@$(RMR) "$(WORKTREE)"
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

.PHONY: lab
lab:
	@echo "==> Launching JupyterLab with uv-managed environment"
	$(RUN) python -m jupyter lab --ServerApp.root_dir="$(CURDIR)"

.PHONY: speed
speed:
	-MIEPYTHON_USE_JIT=0 $(RUN) python tests/benchmark_efficiencies.py
	-MIEPYTHON_USE_JIT=1 $(RUN) python tests/benchmark_efficiencies.py
	-$(RUN) python tests/benchmark_efficiencies.py --compare

.PHONY: lite-clean
lite-clean:
	@echo "==> Cleaning JupyterLite build artifacts"
	@$(RMR) "$(STAGE_DIR)"
	@$(RMR) "$(OUT_ROOT)"
	@$(RMR) "$(DOIT_DB)"
	@$(RMR) .cache dist $(PACKAGE).egg-info

.PHONY: clean
clean: lite-clean
	@echo "==> Cleaning build artifacts"	
	@find . -name '__pycache__' -type d -exec $(RMR) {} +
	@find . -name '.DS_Store' -type f -delete
	@find . -name '.ipynb_checkpoints' -type d -prune -exec $(RMR) {} +
	@find . -name '.pytest_cache' -type d -prune -exec $(RMR) {} +
	@$(RMR) .ruff_cache
	@$(RMR) docs/api docs/_build docs/.jupyter

.PHONY: realclean
realclean: clean
	@echo "==> Deep cleaning: removing venv and deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@git worktree prune || true
	$(RMR) "$(WORKTREE)"
	$(RMR) .venv
	@$(RM) uv.lock
