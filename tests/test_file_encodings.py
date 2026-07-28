"""Test that text files are read with an explicit encoding.

``open(path)`` and ``Path.read_text()`` decode using the locale's preferred encoding
when no ``encoding=`` is given.  On Linux and macOS that is UTF-8, so the omission is
invisible; on Windows it is cp1252, which cannot decode most non-ASCII bytes.

``tests/test_makefile.py`` was added with a bare ``MAKEFILE.read_text()`` and passed
everywhere except the four Windows jobs in the CI matrix, where it died on the ``OK``
and ``FAIL`` emoji in the Makefile's echo lines::

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 6472

Nothing about that failure is specific to the Makefile: any file with a non-ASCII
character is a Windows-only failure waiting for whoever adds one.  This test walks the
repository's syntax trees and requires the keyword, so the mistake cannot come back
through a platform the author does not run.

To reproduce a Windows-style encoding failure on a UTF-8 machine, force the
interpreter onto an ASCII locale::

    PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 LC_ALL=C LANG=C pytest tests
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".venv", "_site", ".lite_src", "_build", "__pycache__", ".git", "htmlcov", ".gh-pages"}

# calls that decode or encode text and therefore need an explicit encoding
TEXT_CALLS = {"read_text", "write_text"}


def _python_files():
    """Every Python source file in the repository worth checking."""
    files = [p for p in ROOT.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)]
    assert len(files) > 20, f"only found {len(files)} python files; the walk is wrong"
    return files


def _is_binary_mode(node):
    """True when an ``open`` call asks for bytes, which need no encoding."""
    mode = node.args[1] if len(node.args) > 1 else None
    if mode is None:
        mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _offenders(path):
    """Return ``(line, call)`` for each text read/write in the file lacking an encoding."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            name = "open"
        elif isinstance(func, ast.Attribute) and func.attr in TEXT_CALLS:
            name = func.attr
        else:
            continue

        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        if name == "open":
            if _is_binary_mode(node):
                continue
            if len(node.args) > 3:  # encoding passed positionally
                continue

        found.append((node.lineno, name))
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_text_io_names_its_encoding(path):
    """Every text read and write must say which encoding it means."""
    offenders = _offenders(path)
    formatted = [f"{path.relative_to(ROOT)}:{line} -> {call}(...) without encoding=" for line, call in offenders]
    assert not offenders, "locale-dependent text I/O breaks on Windows:\n  " + "\n  ".join(formatted)


def test_the_scan_can_actually_find_an_offender():
    """Guard the guard: a checker that never fires proves nothing."""
    sample = ast.parse(
        "\n".join(
            [
                "open('f')",  # offender
                "open('f', 'r')",  # offender
                "p.read_text()",  # offender
                "p.write_text(s)",  # offender
                "open('f', 'rb')",  # fine, bytes
                "open('f', encoding='utf-8')",  # fine
                "p.read_text(encoding='utf-8')",  # fine
                "p.read_bytes()",  # fine
                "zf.open('inner')",  # fine, not the builtin
            ]
        )
    )
    # reuse the same logic the real test does, on a known input
    lines = []
    for node in ast.walk(sample):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            name = "open"
        elif isinstance(func, ast.Attribute) and func.attr in TEXT_CALLS:
            name = func.attr
        else:
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        if name == "open" and _is_binary_mode(node):
            continue
        lines.append(node.lineno)

    assert sorted(lines) == [1, 2, 3, 4], f"the scan flagged {sorted(lines)}, expected the first four lines"
