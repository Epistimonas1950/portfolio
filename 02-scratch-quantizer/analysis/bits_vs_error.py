#!/usr/bin/env python3
"""Bits per weight against activation-weighted error, for every method.

Produces results/bits_vs_error.csv -- the table the README leads with. Writes CSV
unconditionally; plotting lives in analysis/plot_results.py so that the numbers exist
whether or not matplotlib is installed.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.grid import Grid, effective_bits            # noqa: E402
from src.rtn import rtn_quantize                     # noqa: E402
from src.sequential import (output_error, relative_output_error,  # noqa: E402
                            sequential_quantize)
from src.synth import make_layer                     # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = range(5)
BITS = (8, 4, 3, 2)


def run() -> list[dict]:
    rows: list[dict] = []
    for bits in BITS:
        for method, kwargs in (
            ("RTN per-tensor", {"per_channel": False}),
            ("RTN per-channel", {"per_channel": True}),
            ("Sequential, natural order", {"ordering": "natural"}),
            ("Sequential, salience order", {"ordering": "salience"}),
            ("Sequential, salience + fp16 outliers", {"ordering": "salience",
                                                      "n_outliers": 4}),
        ):
            errs, rels, secs = [], [], []
            for seed in SEEDS:
                layer = make_layer(n_out=128, n_in=256, n_samples=512, cond=1e4,
                                   n_outliers=4, outlier_scale=20.0, seed=seed)
                t0 = time.perf_counter()
                if method.startswith("RTN"):
                    grid = Grid(bits, per_channel=kwargs["per_channel"])
                    w_hat = rtn_quantize(layer.w, grid)
                    n_out_ch = 0
                else:
                    grid = Grid(bits)
                    res = sequential_quantize(layer.w, layer.x, grid, damping=1e-2,
                                              **kwargs)
                    w_hat = res.w_hat
                    n_out_ch = len(res.outlier_cols)
                secs.append(time.perf_counter() - t0)
                errs.append(output_error(layer.w, w_hat, layer.x))
                rels.append(relative_output_error(layer.w, w_hat, layer.x))
            rows.append({
                "method": method,
                "nominal_bits": bits,
                "effective_bits": round(effective_bits(128, 256, bits, None,
                                                       fp16_cols=n_out_ch), 4),
                "output_error_mean": round(float(np.mean(errs)), 6),
                "output_error_std": round(float(np.std(errs)), 6),
                "relative_error_mean": round(float(np.mean(rels)), 6),
                "seconds_mean": round(float(np.mean(secs)), 4),
                "n_seeds": len(SEEDS),
            })
            print(f"{bits}b  {method:38s}  rel={rows[-1]['relative_error_mean']:.5f}")
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = run()
    out = RESULTS / "bits_vs_error.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
