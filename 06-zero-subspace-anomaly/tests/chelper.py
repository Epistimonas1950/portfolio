"""Make the compiled C available to the test suite, building it if necessary.

The claims this repo makes about the C are claims about the C, so the tests that check
them shell out to the real binary rather than to a Python re-implementation. That means
the suite needs the binary to exist. `make test` depends on `make host`, but the suite
also has to work when run directly as

    python3 -m unittest discover -s tests -t . -v

so this module builds it on demand. If gcc is missing the C tests skip with a message
naming what is missing -- they do not silently pass.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oracle.chost import TRACKER, TRACKER32                        # noqa: E402

_BUILD_ATTEMPTED = False
_BUILD_ERROR = ""


def ensure_built() -> None:
    """Build build/tracker and build/tracker32 once per process."""
    global _BUILD_ATTEMPTED, _BUILD_ERROR
    if _BUILD_ATTEMPTED:
        return
    _BUILD_ATTEMPTED = True
    if TRACKER.exists() and TRACKER32.exists():
        return
    try:
        proc = subprocess.run(["make", "host"], cwd=ROOT, capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0:
            _BUILD_ERROR = f"`make host` failed:\n{proc.stdout}\n{proc.stderr}"
    except FileNotFoundError as exc:
        _BUILD_ERROR = f"cannot run make: {exc}"


def require_c(test: unittest.TestCase) -> None:
    """Skip a test with a specific message if the C could not be built."""
    ensure_built()
    if _BUILD_ERROR:
        test.skipTest(_BUILD_ERROR)
    if not TRACKER.exists():
        test.skipTest(f"{TRACKER} missing; run `make host` (needs gcc)")


def require_data(test: unittest.TestCase, *names: str) -> None:
    missing = [n for n in names if not (ROOT / "data" / n).exists()]
    if missing:
        test.skipTest(f"missing {missing}; run `make data`")
