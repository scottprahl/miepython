"""Test that the Makefile's help text and declarations match the rules it defines.

``make readme_images`` was broken for as long as it had existed: ``.PHONY`` declared
that name but the rule underneath was written ``readme:``, so the advertised command
failed with "No rule to make target" while an undocumented one worked.  ``make help``
also offered a ``sync`` target that was never written, and described ``pylint-check``
as "Same as lint above" when it runs pylint alone.

Nothing could catch that, because the help text is a pile of ``@echo`` lines with no
connection to the rules below it.  These tests make that connection, so a target
cannot be advertised, or declared phony, without existing.

The Makefile is parsed rather than run: ``make`` is not available everywhere the test
suite runs, and the questions here are about the file's contents anyway.
"""

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[1] / "Makefile"

# rules that exist for `make` itself or as internal plumbing, and so need no help entry
UNADVERTISED_BY_DESIGN = {"help"}


def makefile_text():
    """The Makefile source, or a skip when running somewhere it was not installed."""
    if not MAKEFILE.is_file():
        pytest.skip(f"no Makefile at {MAKEFILE}; not running from a source checkout")
    return MAKEFILE.read_text()


def rule_names():
    """Names of every rule the Makefile defines."""
    found = re.findall(r"^([A-Za-z][\w.-]*)\s*:(?!=)", makefile_text(), re.M)
    names = {name for name in found if name != ".PHONY"}
    assert names, "parsed no rules at all; the regex or the Makefile layout changed"
    return names


def phony_names():
    """Names listed in .PHONY declarations."""
    names = set()
    for line in re.findall(r"^\.PHONY:\s*(.+)$", makefile_text(), re.M):
        names.update(line.split())
    assert names, "parsed no .PHONY names"
    return names


def advertised_names():
    """Target names printed by the help target."""
    names = set(re.findall(r'^\t@echo "  (\S+)\s+-', makefile_text(), re.M))
    assert len(names) > 10, f"parsed only {len(names)} help entries; the help format changed"
    return names


def test_every_advertised_target_exists():
    """`make help` must not offer a command that fails.

    This is the ``readme_images`` and ``sync`` bug: both were advertised, neither
    could be run.
    """
    missing = sorted(advertised_names() - rule_names())
    assert missing == [], f"make help advertises targets with no rule: {missing}"


def test_every_phony_declaration_has_a_rule():
    """A ``.PHONY`` name with no rule underneath is a typo, and reads as if it works."""
    missing = sorted(phony_names() - rule_names())
    assert missing == [], f".PHONY declares targets with no rule: {missing}"


def test_every_rule_is_declared_phony():
    """None of these targets produce a file of their own name.

    If one ever does, this test should be updated to allow it rather than deleted --
    the point is that the choice stays deliberate.
    """
    not_phony = sorted(rule_names() - phony_names())
    assert not_phony == [], f"rules missing from .PHONY: {not_phony}"


def test_every_rule_is_advertised():
    """A target nobody can discover may as well not exist.

    ``venv`` was in this position: it was the target that syncs the environment, but
    help listed a nonexistent ``sync`` instead.
    """
    undocumented = sorted(rule_names() - advertised_names() - UNADVERTISED_BY_DESIGN)
    assert undocumented == [], f"rules missing from make help: {undocumented}"


@pytest.mark.parametrize("variable", ["YAML_TARGETS", "RST_TARGETS", "PYLINT_TARGETS"])
def test_lint_target_lists_name_files_that_exist(variable):
    """The lint targets name files one by one, so a rename silently drops coverage."""
    match = re.search(rf"^{variable}\s*:?=\s*(.+)$", makefile_text(), re.M)
    assert match, f"{variable} is no longer defined"

    root = MAKEFILE.parent
    entries = match.group(1).split()
    assert entries, f"{variable} is empty"

    for entry in entries:
        # expand the two variables these lists use, then resolve globs
        path = entry.replace("$(DOCS_DIR)", "docs").replace("$(PACKAGE)", "miepython")
        assert "$" not in path, f"{variable} uses an unexpanded variable: {entry}"
        assert list(root.glob(path)), f"{variable} names {path}, which does not exist"
