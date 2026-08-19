#!/usr/bin/env python3
"""The batch full-SVD oracle: the accuracy ceiling the streaming tracker is measured against.

This is what you do when memory is not a constraint. Load the whole known-normal
recording, take an exact SVD of it, keep the top r left singular vectors, and score
every sample by residual energy against that fixed subspace:

    A = U Sigma V^T           (numpy.linalg.svd, LAPACK, O(m^2 n))
    score(a) = || a - U_r U_r^T a ||^2 / || a ||^2

No approximation, no forgetting, no rank-one updates and therefore no orthogonality
drift. Whatever the incremental tracker loses relative to this number is the price of
running in bounded memory on a board that cannot hold the data.

Two oracles are computed, and the distinction matters:

`warm-up`     the SVD sees only the known-normal prefix of the stream -- the same
              samples the streaming tracker warms up on. This is the fair comparison:
              same information, exact arithmetic.
`all-normal`  the SVD sees every sample the labels call normal, including ones that
              arrive after the anomalies. It uses the answer key, so it is not
              achievable in deployment. It is reported as an upper bound only, and is
              labelled as such everywhere it appears.

Neither oracle adapts. On a stationary stream that costs nothing; on the rotating
segment it is the whole story, and the comparison against `analysis/forgetting_study.py`
is where the case for exponential forgetting is actually made.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.incremental import CALIBRATION, WARMUP                       # noqa: E402,F401
from oracle.rangefinder import rank_by_energy, rank_by_gap              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def read_stream(path: pathlib.Path) -> np.ndarray:
    """Read a channels-in-columns CSV and return the (m, n) data matrix.

    The file stores one sample per row; the algorithms want one sample per column, so
    this transposes. Stated in generate_data.py too, because it is the single easiest
    thing to get backwards.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` "
                                f"(oracle/generate_data.py) first")
    return np.loadtxt(path, delimiter=",", skiprows=1).T


def read_labels(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` first")
    label, kind = [], []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label.append(int(row["label"]))
            kind.append(row["kind"])
    return np.array(label, dtype=int), kind


def batch_subspace(block: np.ndarray, energy: float = 0.95, r_max: int = 8
                   ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Exact top-r left singular vectors of `block`, with r chosen from the spectrum.

    Returns (U_r, full spectrum, r_energy, r_gap).
    """
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    r_energy = rank_by_energy(s, energy)
    r_gap = rank_by_gap(s, r_max=r_max)
    r = int(np.clip(r_energy, 1, min(r_max, u.shape[1])))
    return np.ascontiguousarray(u[:, :r]), s, r_energy, r_gap


def residual_scores(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """|| a - U U^T a ||^2 / || a ||^2 for every column of x, vectorized."""
    proj = u @ (u.T @ x)
    resid = x - proj
    num = np.einsum("ij,ij->j", resid, resid)
    den = np.einsum("ij,ij->j", x, x)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def oracle_scores(x: np.ndarray, label: np.ndarray | None = None
                  ) -> dict[str, np.ndarray]:
    """Both oracles' score vectors, keyed by name."""
    out: dict[str, np.ndarray] = {}
    u_warm, _, _, _ = batch_subspace(x[:, :WARMUP + CALIBRATION])
    out["batch full SVD (warm-up)"] = residual_scores(x, u_warm)
    if label is not None:
        u_all, _, _, _ = batch_subspace(x[:, label == 0])
        out["batch full SVD (all-normal, uses labels)"] = residual_scores(x, u_all)
    return out


def main() -> None:
    """Write the singular-value spectrum and the rank each criterion selects."""
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for name in ("normal.csv", "anomalous.csv", "multimode.csv", "manymode.csv",
                 "rotating.csv"):
        x = read_stream(DATA / name)
        block = x[:, :WARMUP + CALIBRATION]
        _, s, r_energy, r_gap = batch_subspace(block)
        total = float(np.sum(s ** 2))
        cum = np.cumsum(s ** 2) / total
        for i, sigma in enumerate(s, start=1):
            rows.append({
                "stream": name,
                "index": i,
                "sigma": round(float(sigma), 8),
                "cumulative_energy": round(float(cum[i - 1]), 8),
                "gap_ratio": round(float(s[i - 1] / s[i]), 6) if i < len(s) else "",
                "r_energy_95": r_energy,
                "r_gap": r_gap,
            })
        print(f"{name:18s} r_energy(0.95)={r_energy}  r_gap={r_gap}  "
              f"sigma[:6]={np.array2string(s[:6], precision=2)}")
    out = RESULTS / "spectrum.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
