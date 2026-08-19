#!/usr/bin/env python3
"""Singular-value spectra of W against W S, and per-layer conditioning.

This is the figure the method is easiest to explain from. W and W S are the same
matrix up to a change of variable, but their spectra are not remotely the same shape:
W's is flat-ish, which is exactly why plain truncated SVD has nothing good to throw
away, while W S inherits the decay of the activation covariance and is genuinely
low-rank. Truncating the flat spectrum discards directions the layer actually uses;
truncating the decaying one discards directions it barely visits.

The scalar that summarises the difference is the stable rank

    srank(A) = ||A||_F^2 / ||A||_2^2 = sum_i sigma_i^2 / sigma_1^2

which is the "effective number of directions carrying energy" and needs no threshold.
A large drop from srank(W) to srank(W S) is precisely the statement that whitening
made the layer compressible.

Also reported per layer: cond(M) before damping and cond(M + lambda I) after, since
the brief's numerical-judgment question is what the ridge actually buys, and
"it took cond(M) from 3e6 to 1e5" is an answer while "I added a ridge" is not.

Writes results/spectra.csv (long format, one row per singular value per layer, with
the per-layer scalars carried along so either table can be read straight out of it).
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.factorize import whitened_spectrum          # noqa: E402
from src.rebuild import break_even_rank              # noqa: E402
from src.synth import make_stack                     # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_SAMPLES = 512
RIDGE = 1e-6      # Small enough to leave the spectra alone -- the point of this
                  # script is what the spectra look like, so the damping must not be
                  # part of the answer. Every layer of this stack does in fact factor
                  # at ridge=0 (512 samples for at most 320 channels); the ridge is
                  # margin for a stack where that is not true, not a necessity here.
SEED = 0


def stable_rank(sigma: np.ndarray) -> float:
    """sum sigma_i^2 / sigma_1^2 -- effective number of energy-carrying directions."""
    return float(np.sum(sigma ** 2) / sigma[0] ** 2)


def run() -> list[dict]:
    rows: list[dict] = []
    for layer in make_stack(n_samples=N_SAMPLES, seed=SEED):
        m, n = layer.w.shape
        sigma_w = np.linalg.svd(layer.w, compute_uv=False)
        sigma_ws, wh = whitened_spectrum(layer.w, layer.x, ridge=RIDGE)
        srank_w, srank_ws = stable_rank(sigma_w), stable_rank(sigma_ws)
        print(f"{layer.name}  {m}x{n}  cond(M)={wh.cond_raw:.3e} -> "
              f"{wh.cond_damped:.3e}   stable rank {srank_w:.1f} -> {srank_ws:.1f}")
        for i, (sw, sws) in enumerate(zip(sigma_w, sigma_ws), start=1):
            rows.append({
                "layer": layer.name,
                "m": m,
                "n": n,
                "cond_target": layer.cond_target,
                "cond_m": f"{wh.cond_raw:.6e}",
                "cond_m_damped": f"{wh.cond_damped:.6e}",
                "ridge_ratio": RIDGE,
                "ridge_lambda": f"{wh.lam:.6e}",
                "break_even_rank": break_even_rank(m, n),
                "stable_rank_w": round(srank_w, 4),
                "stable_rank_ws": round(srank_ws, 4),
                "index": i,
                "sigma_w": f"{sw:.8e}",
                "sigma_w_relative": round(float(sw / sigma_w[0]), 8),
                "sigma_ws": f"{sws:.8e}",
                "sigma_ws_relative": round(float(sws / sigma_ws[0]), 8),
            })
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = run()
    out = RESULTS / "spectra.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
