#!/usr/bin/env python3
"""Temperature against throughput under sustained load.

Why the brief asks for this
---------------------------

A Pi 3 under continuous multi-threaded inference heats up, the SoC firmware caps the
ARM clock at 80 C and again at 85 C, and sustained tokens-per-second is therefore a
*different and smaller* number than the tokens-per-second of the first thirty seconds.
Benchmarks that report the first thirty seconds are common and are not wrong about
anything except what a user experiences. Logging temperature alongside throughput is
how you tell the two apart.

What this script does
---------------------

Runs a fixed CPU workload in one-second windows, and at each window boundary records

  * every readable zone under `/sys/class/thermal/thermal_zone*/temp` (millidegrees C),
  * the current CPU frequency from `cpufreq/scaling_cur_freq` if the kernel exposes it,
  * the throughput achieved in that window,

then reports the Pearson correlation between the hottest zone and throughput.

Where it runs, and what that is worth
-------------------------------------

**On this x86 development host it runs and produces a real log** -- these thermal zone
files exist on almost any modern Linux box -- and that log is written to
`results/thermal_host_<arch>.csv`. It is a measurement of *this* machine and says
nothing whatsoever about a Raspberry Pi 3. Its purpose is to prove the instrument
works and to be pointed at a Pi unchanged.

Unlike every other CSV in `results/`, that file is **not reproducible**: it has no
seed, and what it records depends on how warm the machine already was. Two consecutive
runs here differed by 15 C and by the sign of the correlation. That is a property of
the measurement, not a defect in the script, and it is exactly why the brief asks for
a *sustained* run on the board rather than a spot check.

**On a machine with no thermal zones** (a container, most cloud VMs) it does not
crash: it logs the throughput series, writes the CSV with empty temperature columns,
and says plainly that no zones were readable.

The workload is a repeated `float64` matrix multiply, 2*N^3 flops per call. It is a
stand-in for llama.cpp's `ggml` matmul in the only respect that matters for a thermal
test -- it keeps every core busy indefinitely. It is not a proxy for inference
throughput and no tokens-per-second figure is derived from it.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
THERMAL_ROOT = pathlib.Path("/sys/class/thermal")
CPUFREQ = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")


def discover_zones(root: pathlib.Path = THERMAL_ROOT) -> list[tuple[str, pathlib.Path]]:
    """[(type, temp_path)] for every readable zone. Empty list is a valid answer."""
    if not root.is_dir():
        return []
    zones = []
    for zone in sorted(root.glob("thermal_zone*")):
        temp = zone / "temp"
        try:
            int(temp.read_text().strip())
        except (OSError, ValueError):
            continue  # zones can exist but be unreadable without root; skip quietly
        try:
            kind = (zone / "type").read_text().strip()
        except OSError:
            kind = zone.name
        zones.append((f"{kind}[{zone.name.replace('thermal_zone', '')}]", temp))
    return zones


def read_c(path: pathlib.Path) -> float | None:
    """Millidegrees C -> degrees C. None if the read fails mid-run (hotplug, perms)."""
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_cpu_mhz() -> float | None:
    try:
        return int(CPUFREQ.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson r, or nan if either series is constant (which it often is here)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run(seconds: float = 15.0, window_s: float = 1.0, n: int = 512,
        seed: int = 0) -> tuple[list[dict], dict]:
    """Sustained load with per-window temperature and throughput."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    flops_per_call = 2.0 * n ** 3

    zones = discover_zones()
    a @ b  # warm up BLAS so the first window is not measuring thread-pool creation

    rows: list[dict] = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        w0 = time.perf_counter()
        calls = 0
        while time.perf_counter() - w0 < window_s:
            a @ b
            calls += 1
        w1 = time.perf_counter()
        row = {
            "elapsed_s": round(w1 - t_start, 3),
            "window_s": round(w1 - w0, 4),
            "calls": calls,
            "gflops": round(calls * flops_per_call / (w1 - w0) / 1e9, 4),
            "cpu_mhz": read_cpu_mhz() or "",
        }
        for kind, path in zones:
            row[f"temp_C:{kind}"] = read_c(path)
        rows.append(row)

    temps = {kind: [r.get(f"temp_C:{kind}") for r in rows] for kind, _ in zones}
    gflops = [r["gflops"] for r in rows]
    hottest, hottest_max = "", float("-inf")
    for kind, series in temps.items():
        clean = [t for t in series if t is not None]
        if clean and max(clean) > hottest_max:
            hottest, hottest_max = kind, max(clean)

    meta = {
        "host": f"{platform.machine()} {platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "n_windows": len(rows),
        "zones_found": len(zones),
        "hottest_zone": hottest,
        "temp_min_C": min([t for t in temps.get(hottest, []) if t is not None],
                          default=""),
        "temp_max_C": hottest_max if zones else "",
        "gflops_first": gflops[0] if gflops else "",
        "gflops_last": gflops[-1] if gflops else "",
        "gflops_median": round(float(np.median(gflops)), 4) if gflops else "",
        "throughput_retained": round(gflops[-1] / gflops[0], 4)
        if gflops and gflops[0] else "",
        "pearson_temp_vs_gflops": round(
            pearson([t for t in temps.get(hottest, []) if t is not None], gflops), 4)
        if zones else "",
    }
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=512, help="matmul dimension")
    ap.add_argument("--out", default=None,
                    help="default: results/thermal_host_<arch>.csv")
    args = ap.parse_args(argv)

    zones = discover_zones()
    if not zones:
        print("no readable thermal zones under /sys/class/thermal -- this host does "
              "not expose CPU temperature (common in containers and cloud VMs).")
        print("Continuing: the throughput series is still logged, the temperature "
              "columns will be empty, and no correlation is reported.")
    else:
        print(f"thermal zones: {', '.join(k for k, _ in zones)}")

    rows, meta = run(seconds=args.seconds, window_s=args.window, n=args.size)

    out = ROOT / (args.out or f"results/thermal_host_{platform.machine()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["host", "note"]
    note = (f"measured on {meta['host']} -- NOT a Raspberry Pi 3. This validates the "
            "instrument; the Pi throttling curve is unmeasured. See STATUS.md.")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "host": meta["host"], "note": note})

    print()
    for k, v in meta.items():
        print(f"  {k:<26} {v}")
    print()
    if zones:
        r = meta["pearson_temp_vs_gflops"]
        if isinstance(r, float) and r == r:
            print(f"Pearson r(temperature, throughput) = {r:+.3f} over "
                  f"{meta['n_windows']} one-second windows.")
        else:
            print("Correlation undefined: temperature or throughput was constant.")
        print()
        print("Read that number with care, and do not read a sign into it.")
        print("  * This run depends entirely on what the machine was doing "
              "beforehand. Two consecutive runs on this development host gave "
              "68->71 C with r = -0.26 (cold start, no throttling) and 78->86 C "
              "with r = +0.57 (already warm from the test suite, ~13% throughput "
              "lost). Neither is wrong; the run is simply not reproducible, unlike "
              "every seeded CSV in results/.")
        print("  * A positive r is not a paradox. Once a chip is oscillating around "
              "its cap, the clock and the temperature rise and fall together, so "
              "temperature and throughput become POSITIVELY correlated -- the "
              "negative correlation people expect belongs to the initial heating "
              "transient, and a 15-second window cannot separate the two.")
        print("  * Separating them is what the Pi 3 study needs: a sustained "
              "inference load and tens of minutes, not a 15-second matmul.")
    print(f"\nwrote {out}")
    print("This file describes THIS host. results/latency_budget.md is where the Pi 3 "
          "numbers go, and it is deliberately empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
