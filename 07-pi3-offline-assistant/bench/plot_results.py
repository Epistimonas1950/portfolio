#!/usr/bin/env python3
"""Draw the figures from the CSVs in results/.

Separated from the scripts that compute the numbers, on purpose. matplotlib is not
installed on the machine this repo was written on, so if plotting lived inside the
measurement scripts none of the numbers would exist either. Every figure this file
draws is redundant with a CSV that is already on disk:

    results/vad_snr_sweep.csv        -> vad_snr_sweep.png       (measured here)
    results/thermal_host_*.csv       -> thermal_host.png        (measured here)
    results/latency_simulated.csv    -> latency_simulated.png   (SIMULATED)

The brief also asks for `results/thermal_throttling.png` from a Pi 3. That figure is
not drawn by this script from anything, because the data for it does not exist. See
STATUS.md.
"""

from __future__ import annotations

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed, so no figure can be drawn.\n"
            "  The system interpreter here is PEP-668 externally managed, so\n"
            "  `pip install matplotlib` into it will fail. Use a venv:\n"
            "      python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
            "      .venv/bin/python bench/plot_results.py\n"
            "  Every number these figures would show is already in results/*.csv."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `make results` first.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_vad(plt) -> pathlib.Path:
    rows = _read("vad_snr_sweep.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for variant, style in (("full", "-o"), ("energy_only", "--s")):
        sel = [r for r in rows if r["variant"] == variant]
        snr = [float(r["snr_db"]) for r in sel]
        ax1.plot(snr, [float(r["f1"]) for r in sel], style, label=variant)
        ax2.plot(snr, [float(r["start_mae_ms"]) for r in sel], style, label=variant)
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("per-frame F1"); ax1.legend()
    ax1.set_title("VAD accuracy vs SNR (measured, synthetic ground truth)")
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("onset error (ms)"); ax2.legend()
    ax2.set_title("Boundary error")
    fig.tight_layout()
    out = RESULTS / "vad_snr_sweep.png"
    fig.savefig(out, dpi=140)
    return out


def plot_thermal(plt) -> pathlib.Path | None:
    candidates = sorted(RESULTS.glob("thermal_host_*.csv"))
    if not candidates:
        return None
    with candidates[0].open() as fh:
        rows = list(csv.DictReader(fh))
    temp_col = next((k for k in rows[0] if k.startswith("temp_C:")), None)
    fig, ax = plt.subplots(figsize=(7, 4))
    t = [float(r["elapsed_s"]) for r in rows]
    ax.plot(t, [float(r["gflops"]) for r in rows], "-o", label="throughput (GFLOP/s)")
    ax.set_xlabel("elapsed (s)"); ax.set_ylabel("GFLOP/s")
    if temp_col:
        ax2 = ax.twinx()
        ax2.plot(t, [float(r[temp_col] or "nan") for r in rows], "-r", label=temp_col)
        ax2.set_ylabel("temperature (C)")
    ax.set_title(f"Host thermal log -- NOT a Raspberry Pi 3 ({candidates[0].name})")
    fig.tight_layout()
    out = RESULTS / "thermal_host.png"
    fig.savefig(out, dpi=140)
    return out


def plot_latency(plt) -> pathlib.Path:
    rows = [r for r in _read("latency_simulated.csv") if r["stage"] != "end_to_end"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = [r["stage"] for r in rows]
    idx = range(len(names))
    ax.barh(list(idx), [float(r["median_ms"]) for r in rows], label="median")
    ax.barh(list(idx), [float(r["p95_ms"]) for r in rows], height=0.35, label="p95")
    ax.set_yticks(list(idx)); ax.set_yticklabels(names)
    ax.set_xlabel("ms")
    ax.set_title("SIMULATED stage latencies -- not a measurement of any hardware")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "latency_simulated.png"
    fig.savefig(out, dpi=140)
    return out


def main() -> int:
    plt = _require_matplotlib()
    for path in (plot_vad(plt), plot_thermal(plt), plot_latency(plt)):
        if path:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
