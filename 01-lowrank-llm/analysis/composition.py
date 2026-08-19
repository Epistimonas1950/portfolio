#!/usr/bin/env python3
"""Where the low-rank / quantization crossover actually sits.

The brief for this project volunteers, up front, that 4-bit quantization usually beats
low-rank factorization at equal compression on modern LLMs. This script measures that
claim rather than repeating it, and answers the three questions it raises:

  1. At a matched compression budget, is it better to spend it on rank or on bits?
  2. Where is the crossover, and how stable is it?
  3. Does the order of composition matter?

Matching the axis honestly
--------------------------
Every configuration is placed on the same axis: bits per weight of the ORIGINAL matrix,
counting the fp16 scales. Counting a quantizer's per-row scales for one method and not
the other is the standard way these tables mislead.

There is a second, subtler trap. Quantization offers only a DISCRETE ladder of
compressions -- one per integer bit-width, so 1.98x, 2.64x, 3.94x, 5.22x, 7.76x on this
layer and nothing in between -- while rank is nearly continuous. Picking round target
compressions like "6x" therefore forces quantization to the next rung UP (2 bits, 7.76x)
and then compares a 7.76x-compressed quantized layer against a 6x-compressed factored
one. That is not a crossover measurement, it is a rigged one.

So the sweep below is anchored on the QUANTIZER's achievable rungs: at each bit-width,
the low-rank family is given exactly the compression the quantizer achieved, and both
arms are asked for their best configuration at that budget. The discreteness itself is a
real practical advantage of low-rank and is reported separately -- it is not allowed to
masquerade as an error-rate win.

Writes results/composition.csv (the crossover) and results/composition_order.csv
(ordering and ablations).
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.compose import (dense_bits, lowrank_only,             # noqa: E402
                         lowrank_then_quantize, quantize_only,
                         quantize_then_lowrank, quantized_bits, ranks_for_budget)
from src.synth import make_layer                               # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SEEDS = range(3)
M = N = 256
N_SAMPLES = 512
# 2 bits is the practical floor for a uniform symmetric grid: at 1 bit there is a single
# non-zero level and the "quantizer" is a sign map.
QUANT_BITS = (8, 6, 4, 3, 2)
FACTOR_BITS = (8, 6, 4, 3, 2)

# Four activation spectra, from the isotropic control to strongly spiked. The crossover
# is only a result if it survives this; a single spectrum would just be one data point.
SPECTRA = (
    ("isotropic control", 1.0, 0),
    ("anisotropic", 1e5, 0),
    ("anisotropic + 4 spiked", 1e5, 4),
    ("strongly spiked", 1e7, 8),
)


def best_lowrank_family(layer, target: float):
    """Best (lowest error) low-rank or composed configuration at a given compression."""
    best = None
    for bits in FACTOR_BITS:
        r = ranks_for_budget(M, N, bits, target)
        if r > 0:
            c = lowrank_then_quantize(layer.w, layer.x, r, bits)
            if best is None or c.rel_error < best.rel_error:
                best = c
    r = min(min(M, N), int((dense_bits(M, N) / target) // ((M + N) * 16)))
    if r > 0:
        c = lowrank_only(layer.w, layer.x, r)
        if best is None or c.rel_error < best.rel_error:
            best = c
    return best


def run_crossover() -> list[dict]:
    out: list[dict] = []
    for label, cond, spiked in SPECTRA:
        print(f"  spectrum: {label}")
        for bq in QUANT_BITS:
            target = dense_bits(M, N) / quantized_bits(M, N, bq)
            q_err, l_err, winners = [], [], []
            for seed in SEEDS:
                layer = make_layer(n_out=M, n_in=N, n_samples=N_SAMPLES, cond=cond,
                                   n_spiked=spiked, seed=seed)
                q = quantize_only(layer.w, layer.x, bq)
                l = best_lowrank_family(layer, target)
                q_err.append(q.rel_error)
                l_err.append(l.rel_error)
                winners.append(f"r={l.rank},b={l.bits if l.bits else 'fp16'}")
            qm, lm = float(np.mean(q_err)), float(np.mean(l_err))
            out.append({
                "spectrum": label, "activation_cond": cond, "spiked_channels": spiked,
                "quant_bits": bq,
                "achieved_compression": round(target, 4),
                "effective_bits": round(quantized_bits(M, N, bq) / (M * N), 4),
                "quantize_only_error": round(qm, 6),
                "quantize_only_std": round(float(np.std(q_err)), 6),
                "best_lowrank_family_error": round(lm, 6),
                "best_lowrank_family_std": round(float(np.std(l_err)), 6),
                "best_lowrank_config": winners[0],
                "winner": "quantize" if qm < lm else "lowrank",
                "ratio_lowrank_over_quant": round(lm / qm, 4) if qm else "",
                "n_seeds": len(SEEDS),
            })
            print(f"    {bq}b -> {target:5.2f}x   quant={qm:.4f}  lowrank={lm:.4f}"
                  f"  winner={out[-1]['winner']}")
    return out


def run_order() -> list[dict]:
    """Ordering and ablations, at three ranks spanning rounding- to truncation-limited.

    The ablations only bite where ROUNDING is the binding error, so the three ranks below
    are chosen to span the regimes rather than to look evenly spaced. On this spectrum
    the rounding share of the total error -- measured as 1 - err(fp16 factors)/err(4-bit
    factors) -- is 3.8% at r=8, 32.9% at r=32 and 99.9% at r=252. A first version of this
    study used r = 61/125/252, which reads like a spread but is 70%/94%/99.9% rounding:
    three samples of the same regime, and the ablations correctly showed no variation.
    """
    configs = [(252, 4, "rounding-limited (99.9% of error from rounding)"),
               (32, 4, "balanced (33% rounding)"),
               (8, 4, "truncation-limited (3.8% rounding)")]
    out: list[dict] = []
    for seed in SEEDS:
        layer = make_layer(n_out=M, n_in=N, n_samples=N_SAMPLES, cond=1e5, n_spiked=4,
                           seed=seed)
        for r, bits, regime in configs:
            for c in (
                lowrank_then_quantize(layer.w, layer.x, r, bits, aware=True, refit=True),
                lowrank_then_quantize(layer.w, layer.x, r, bits, aware=True, refit=False),
                lowrank_then_quantize(layer.w, layer.x, r, bits, aware=False, refit=True),
                lowrank_then_quantize(layer.w, layer.x, r, bits, aware=False, refit=False),
                quantize_then_lowrank(layer.w, layer.x, r, bits, requantize=False),
                quantize_then_lowrank(layer.w, layer.x, r, bits, requantize=True),
            ):
                out.append({
                    "seed": seed, "regime": regime, "rank": r, "bits": bits,
                    "method": c.method, "detail": c.detail,
                    "effective_bits": round(c.effective_bits(M, N), 4),
                    "compression": round(c.compression(M, N), 4),
                    "relative_error": round(c.rel_error, 6),
                })
        print(f"  order: seed {seed} done")
    return out


def write(rows: list[dict], name: str) -> None:
    path = RESULTS / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("crossover sweep (anchored on the quantizer's achievable compressions):")
    rows = run_crossover()
    write(rows, "composition.csv")

    print("\ncrossover location, per spectrum:")
    for label, _, _ in SPECTRA:
        sub = sorted((r for r in rows if r["spectrum"] == label),
                     key=lambda r: r["achieved_compression"])
        flip = [(a, b) for a, b in zip(sub, sub[1:])
                if a["winner"] == "quantize" and b["winner"] == "lowrank"]
        if flip:
            a, b = flip[0]
            print(f"  {label:24s} between {a['achieved_compression']:.2f}x "
                  f"({a['quant_bits']}-bit) and {b['achieved_compression']:.2f}x "
                  f"({b['quant_bits']}-bit)")
        else:
            print(f"  {label:24s} no crossover in the measured range")

    print("\nordering and ablations:")
    write(run_order(), "composition_order.csv")


if __name__ == "__main__":
    main()
