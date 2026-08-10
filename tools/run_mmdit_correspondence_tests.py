#!/usr/bin/env python3
"""Run the correspondence unit tests without requiring Pytest.

The production container only needs the test files' small ``raises`` and
``monkeypatch`` subset.  When Pytest is installed the Slurm launcher still
uses it; this runner is the deterministic fallback used by the repository's
existing DiffSynth image.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Raises(contextlib.AbstractContextManager[None]):
    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected = expected
        self.match = match

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> bool:
        del traceback
        if kind is None:
            raise AssertionError(f"{self.expected.__name__} was not raised")
        if not issubclass(kind, self.expected):
            return False
        if self.match is not None and self.match not in str(value):
            raise AssertionError(
                f"expected exception text containing {self.match!r}, got {value!r}"
            )
        return True


class _MonkeyPatch:
    def __init__(self) -> None:
        self._changes: list[tuple[Any, str, Any]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        self._changes.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, original in reversed(self._changes):
            setattr(target, name, original)
        self._changes.clear()


def _install_pytest_shim() -> None:
    try:
        import pytest  # noqa: F401

        return
    except ImportError:
        shim = types.ModuleType("pytest")
        shim.raises = lambda expected, match=None: _Raises(expected, match)  # type: ignore[attr-defined]
        sys.modules["pytest"] = shim


def _load_module(path: Path) -> types.ModuleType:
    name = f"_mmdit_selftest_{path.stem}"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import test module {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _run_test(function: Callable[..., None]) -> None:
    parameters = inspect.signature(function).parameters
    monkeypatch = _MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory(prefix="mmdit-selftest-") as directory:
            fixtures: dict[str, Any] = {
                "tmp_path": Path(directory),
                "monkeypatch": monkeypatch,
            }
            unsupported = sorted(set(parameters) - set(fixtures))
            if unsupported:
                raise RuntimeError(
                    f"{function.__name__} requests unsupported fixtures {unsupported}"
                )
            function(**{name: fixtures[name] for name in parameters})
    finally:
        monkeypatch.undo()


def main() -> None:
    _install_pytest_shim()
    failures: list[tuple[str, BaseException]] = []
    count = 0
    for relative in (
        "tests/test_mmdit_correspondence.py",
        "tests/test_mmdit_correspondence_report.py",
    ):
        module = _load_module(PROJECT_ROOT / relative)
        for name in sorted(value for value in dir(module) if value.startswith("test_")):
            function = getattr(module, name)
            if not callable(function):
                continue
            count += 1
            try:
                _run_test(function)
                print(f"PASS {name}")
            except BaseException as exc:  # keep running to report every failure
                failures.append((name, exc))
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"MMDiT correspondence self-test: {count - len(failures)}/{count} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
