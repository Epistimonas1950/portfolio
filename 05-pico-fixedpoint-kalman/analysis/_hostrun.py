"""Shared plumbing: build-check and one call into the compiled fixed-point filter.

Kept in one place so every sweep fails the same way -- with the make target to run --
rather than each script inventing its own message about a missing binary.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "kfhost"
TRACE = ROOT / "data" / "imu_capture.csv"
RESULTS = ROOT / "results"
BUILD = ROOT / "build"


def require_binary() -> pathlib.Path:
    if not BINARY.exists():
        sys.exit(f"{BINARY} not found. Build the host target first:\n    make host")
    return BINARY


def require_trace(path: pathlib.Path = TRACE) -> pathlib.Path:
    if not path.exists():
        sys.exit(f"{path} not found. Generate it first:\n"
                 f"    python3 reference/generate_trace.py")
    return path


def run(variant: str, trace: pathlib.Path = TRACE, out: pathlib.Path | None = None,
        **flags) -> tuple[dict, np.ndarray | None]:
    """Invoke kfhost. Returns (summary dict, per-step array or None)."""
    cmd = [str(require_binary()), "--trace", str(trace), "--variant", variant]
    if out is not None:
        cmd += ["--out", str(out)]
    for key, val in flags.items():
        cmd += ["--" + key.replace("_", "-"), str(val)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    summary = {}
    for line in proc.stdout.splitlines():
        key, _, val = line.partition("=")
        summary[key] = val
    steps = np.genfromtxt(out, delimiter=",", names=True) if out is not None else None
    return summary, steps
