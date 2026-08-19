#!/usr/bin/env python3
"""Thin wrapper around the compiled C tracker, so Python can score what the C produced.

The C binary is the deliverable -- it is the thing that would run on the board -- and
every number attributed to it in the README comes through this module by actually
executing it. Nothing here re-implements the algorithm; if `build/tracker` is missing,
the C rows are omitted rather than silently substituted with the numpy ones.

The binary prints its per-sample output to a CSV and a `key=value` summary block to
stdout. Parsing key/value pairs rather than positional output means adding a field to
the C does not break the Python.
"""

from __future__ import annotations

import pathlib
import subprocess
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
TRACKER = BUILD / "tracker"
TRACKER32 = BUILD / "tracker32"


def c_available(binary: pathlib.Path = TRACKER) -> bool:
    return binary.exists()


def require(binary: pathlib.Path = TRACKER) -> pathlib.Path:
    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} not built. Run `make host` (plain gcc, no dependencies).")
    return binary


@dataclass
class CDefaults:
    """Tracker parameters. The defaults mirror oracle/incremental.py exactly."""
    lam: float = 1.0
    reorth: bool = True
    reorth_tol: float = 100.0 * 2.220446049250313e-16
    check_every: int = 20
    rank_max: int = 8
    oversampling: int = 6
    power_iters: int = 1
    energy: float = 0.95
    rank_mode: str = "energy"
    quantile: float = 0.99
    warmup: int = 300
    calibration: int = 300
    repeat: int = 1
    seed: int = 0

    def argv(self) -> list[str]:
        return [
            "--lambda", repr(self.lam),
            "--reorth", "on" if self.reorth else "off",
            "--reorth-tol", repr(self.reorth_tol),
            "--check-every", str(self.check_every),
            "--rank-max", str(self.rank_max),
            "--oversampling", str(self.oversampling),
            "--power-iters", str(self.power_iters),
            "--energy", repr(self.energy),
            "--rank-mode", self.rank_mode,
            "--quantile", repr(self.quantile),
            "--warmup", str(self.warmup),
            "--calibration", str(self.calibration),
            "--repeat", str(self.repeat),
            "--seed", str(self.seed),
        ]


@dataclass
class CResult:
    scores: np.ndarray
    drift: np.ndarray
    basis: np.ndarray | None
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.summary["rank"])

    @property
    def threshold(self) -> float:
        return float(self.summary["threshold"])

    @property
    def max_drift(self) -> float:
        return float(self.summary["max_drift"])

    @property
    def final_drift(self) -> float:
        return float(self.summary["final_drift"])

    @property
    def seconds(self) -> float:
        return float(self.summary["seconds"])


def _parse_summary(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def run_c_tracker(stream: pathlib.Path, opts: CDefaults | None = None,
                  binary: pathlib.Path = TRACKER,
                  out_csv: pathlib.Path | None = None,
                  want_basis: bool = False) -> CResult:
    """Run `tracker track` on a stream CSV and return its scores, drift and basis."""
    require(binary)
    opts = opts or CDefaults()
    BUILD.mkdir(exist_ok=True)
    out_csv = out_csv or (BUILD / f"{stream.stem}_scores.csv")
    basis_path = BUILD / f"{stream.stem}_basis.csv" if want_basis else None

    argv = [str(binary), "track", "--input", str(stream), "--output", str(out_csv)]
    argv += opts.argv()
    if basis_path is not None:
        argv += ["--basis", str(basis_path)]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)}\nexit {proc.returncode}\n{proc.stderr}")

    table = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    basis = None
    if basis_path is not None:
        basis = np.atleast_2d(np.loadtxt(basis_path, delimiter=",", skiprows=1))
    return CResult(scores=table[:, 1], drift=table[:, 2], basis=basis,
                   summary=_parse_summary(proc.stdout))


def run_c_rangefinder(stream: pathlib.Path, rank: int, oversampling: int,
                      power_iters: int, seed: int = 0,
                      n_cols: int | None = None,
                      binary: pathlib.Path = TRACKER) -> float:
    """Run `tracker rangefind` and return || A - Q Q^T A ||_F as the C computed it."""
    require(binary)
    argv = [str(binary), "rangefind", "--input", str(stream), "--rank", str(rank),
            "--oversampling", str(oversampling), "--power-iters", str(power_iters),
            "--seed", str(seed)]
    if n_cols is not None:
        argv += ["--columns", str(n_cols)]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)}\nexit {proc.returncode}\n{proc.stderr}")
    return float(_parse_summary(proc.stdout)["proj_error"])


def run_c_selftest(binary: pathlib.Path = TRACKER) -> tuple[int, str]:
    """Run the C's internal linear-algebra checks. Returns (exit code, stdout)."""
    require(binary)
    proc = subprocess.run([str(binary), "selftest"], capture_output=True, text=True,
                          check=False)
    return proc.returncode, proc.stdout + proc.stderr
