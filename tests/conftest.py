"""Configuration file for pytest."""


def pytest_addoption(parser):
    """Add --notebooks option to pytest parser."""
    parser.addoption(
        "--notebooks", action="store_true", default=False, dest="notebooks",
        help="test notebooks by running them"
    )


def pytest_collection_modifyitems(config, items):
    """Modify files to automatically be tested by pytest."""
    import pytest

    # Skip items marked with `notebooks` by default
    if config.option.notebooks:
        # --notebooks was set, do not skip
        return

    skip_notebooks = pytest.mark.skip(reason="--notebooks option not used")
    for item in items:
        if "notebooks" in item.keywords:
            item.add_marker(skip_notebooks)
