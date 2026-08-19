"""Shared test plumbing: find the compiled binaries, run them, parse their output.

The suite deliberately tests the COMPILED C, not a Python re-implementation of it. A
Python model of fixed-point arithmetic would agree with itself and prove nothing; the
claims in the README are about what gcc produced, so the tests shell out to it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
KFHOST = BUILD / "kfhost"
QTEST = BUILD / "qtest"
TRACE = ROOT / "data" / "imu_capture.csv"

sys.path.insert(0, str(ROOT))


def require(path: pathlib.Path, how: str) -> pathlib.Path:
    if not path.exists():
        raise unittest.SkipTest(f"{path} missing. {how}")
    return path


def need_binaries() -> None:
    for path in (KFHOST, QTEST):
        if not path.exists():
            raise AssertionError(
                f"{path} has not been built. Run:\n    make host\n"
                "(or `make test`, which builds it first). The suite tests the compiled "
                "C on purpose and will not silently fall back to a Python model.")


def need_trace() -> None:
    if not TRACE.exists():
        raise AssertionError(
            f"{TRACE} has not been generated. Run:\n"
            "    python3 reference/generate_trace.py\n(or `make test`).")


def kfhost(variant: str = "joseph", trace: pathlib.Path | None = None,
           out: pathlib.Path | None = None, **flags) -> dict[str, str]:
    """Run the compiled filter and return its key=value summary."""
    need_binaries()
    cmd = [str(KFHOST), "--trace", str(trace or TRACE), "--variant", variant]
    if out is not None:
        cmd += ["--out", str(out)]
    for key, val in flags.items():
        cmd += ["--" + key.replace("_", "-"), str(val)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)} exited {proc.returncode}\n{proc.stderr}")
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def steps(path: pathlib.Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)
