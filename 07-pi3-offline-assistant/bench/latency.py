#!/usr/bin/env python3
"""Latency statistics: medians and p95, per stage and end to end.

The brief asks for medians and p95, explicitly not means, and that instruction is
worth stating the reason for rather than just obeying. Stage latency on a small board
is a positive, right-skewed, heavy-tailed quantity: a page fault, a `systemd-journald`
flush or a thermal-governor step adds a multiple of the typical cost, not a fraction.
The mean of such a sample is a weighted average of the typical case and the worst
case, and it tracks neither. The median says what usually happens; p95 says what the
user notices. Both are reported here; the mean is carried in the CSV only so that the
gap between it and the median is visible.

Percentile definition
---------------------

Linear interpolation between order statistics -- numpy's default, R's type 7. For a
sorted sample x_(1) <= ... <= x_(n) and probability p,

    h  = (n - 1) p
    Q(p) = x_(floor(h)+1) + (h - floor(h)) * ( x_(floor(h)+2) - x_(floor(h)+1) )

with the obvious clamp at the top. It is implemented here rather than delegated,
because "the harness computes percentiles correctly" is one of the two assertions this
project is anchored on and delegating it would make the test a test of numpy.

The check has an exact form. For a sample of size n from a distribution with density
f, the sample p-quantile is asymptotically normal about the true quantile with

    SE  =  sqrt( p (1 - p) / n ) / f(Q(p))

so `tests/test_latency.py` compares the harness's output against closed-form quantiles
of the uniform, exponential and lognormal distributions to within a few of these
standard errors -- not against a hardcoded number and not against another
implementation.

Percentiles do not add
----------------------

A latency budget is usually written as a table of per-stage p95s with a total at the
bottom. That total is wrong. p95 of a sum is not the sum of the p95s: for independent
stages, all of them being simultaneously unlucky is much rarer than any one of them
being unlucky, so the naive sum overstates the end-to-end p95. This script measures
the overstatement for the model it is running and writes it to the CSV as
`p95_overstatement`.

Stated carefully, because the general claim is false: quantiles are *not* subadditive
in general -- that non-subadditivity is the standard argument against value-at-risk as
a risk measure -- and comonotone or heavy-tailed stages can make the sum an
*under*statement. The honest claim is the measured one: under the independent
lognormal model used here, summing the per-stage p95s overstates the end-to-end p95 by
the factor in the CSV. The lesson that survives is only that the naive sum needs
checking, not that it is always conservative.

EVERY LATENCY NUMBER THIS SCRIPT WRITES IS DRAWN, NOT MEASURED, except the
`harness_overhead_*` columns, which are real wall-clock measurements of this host.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.orchestrate import (build_real_assistant, build_simulated_assistant,  # noqa: E402
                             SyntheticCaptureStage)
from src.stages import PLACEHOLDER_MODELS, Z_P50, Z_P95, Z_P99  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

STAGE_ORDER = ["capture", "vad", "asr", "llm_prefill", "llm_decode", "tts"]


def percentile(samples, q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation on order statistics."""
    a = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = a.size
    if n == 0:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must lie in [0, 100], got {q}")
    if n == 1:
        return float(a[0])
    h = (n - 1) * (q / 100.0)
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    return float(a[lo] + (h - lo) * (a[hi] - a[lo]))


def quantile_standard_error(p: float, n: int, density_at_quantile: float) -> float:
    """Asymptotic SE of a sample quantile: sqrt(p(1-p)/n) / f(Q(p)).

    Used by the tests to size the tolerance from the statistics rather than from
    whatever number happened to make the assertion pass.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if density_at_quantile <= 0:
        raise ValueError("density at the quantile must be positive")
    return float(np.sqrt(p * (1.0 - p) / n) / density_at_quantile)


def summarise(samples, label: str = "") -> dict:
    """median / p95 / p99 plus the supporting cast. Mean included, deliberately last."""
    a = np.asarray(samples, dtype=np.float64).ravel()
    return {
        "stage": label,
        "n": int(a.size),
        "median_ms": round(percentile(a, 50), 4),
        "p95_ms": round(percentile(a, 95), 4),
        "p99_ms": round(percentile(a, 99), 4),
        "min_ms": round(float(a.min()), 4),
        "max_ms": round(float(a.max()), 4),
        "mean_ms": round(float(a.mean()), 4),
    }


class NoCompletedTurns(RuntimeError):
    """Every turn failed, so there is nothing to take a percentile of."""


def run(n_turns: int = 400, seed: int = 0, snr_db: float = 20.0,
        real: bool = False, model: str = "models/model-q4_k_m.gguf",
        whisper_model: str = "models/ggml-tiny.en-q5_1.bin",
        voice: str = "voices/en_US-lessac-low.onnx") -> tuple[list[dict], dict]:
    """Run `n_turns` turns and collect per-stage samples.

    With `real=False` (the default here, and the only thing that works on a machine
    without the inference binaries): capture and VAD are genuinely executed and
    genuinely timed, the four inference stages are drawn. The two are kept in separate
    columns of the output and are never summed into a single unlabelled figure.

    With `real=True`: the same harness, pointed at `SubprocessStage`s that shell out
    to whisper.cpp, llama.cpp and Piper. THIS is what produces
    `results/latency_budget.md` on the board. It is a constructor swap and nothing
    else -- the timing, the statistics and the state machine are identical, which is
    the whole reason the `Stage` seam exists. On any host without the binaries every
    turn fails at the first inference stage, naming what is missing.
    """
    if real:
        assistant = build_real_assistant(model=model, whisper_model=whisper_model,
                                         voice=voice)
    else:
        assistant = build_simulated_assistant(
            capture=SyntheticCaptureStage(seed=seed, snr_db=snr_db), seed=seed)
    per_stage: dict[str, list[float]] = {}
    overhead: dict[str, list[float]] = {}
    end_to_end: list[float] = []
    simulated_flags: dict[str, bool] = {}
    failures = 0

    first_failure = None
    for i in range(n_turns):
        if not real:
            # A fresh capture seed per turn, so the VAD timing varies over real
            # signals rather than re-timing one identical array 400 times.
            assistant.capture = SyntheticCaptureStage(seed=seed + i, snr_db=snr_db)
        turn = assistant.run_turn()
        if not turn.ok or turn.final_state.value != "done":
            failures += 1
            if first_failure is None:
                first_failure = f"stage '{turn.failed_stage}': {turn.error}"
            continue
        for s in turn.stages:
            per_stage.setdefault(s.stage, []).append(s.latency_ms)
            overhead.setdefault(s.stage, []).append(s.harness_overhead_ms)
            simulated_flags[s.stage] = s.simulated
        end_to_end.append(turn.end_to_end_ms)

    if not end_to_end:
        raise NoCompletedTurns(
            f"all {n_turns} turns failed, so there are no latencies to summarise.\n"
            f"  first failure: {first_failure}\n"
            f"  This is the expected result of --real on a machine without the "
            f"inference binaries. Build them on the board with setup/install.sh, or "
            f"drop --real to exercise the harness against SIMULATED draws.")

    rows = []
    for name in STAGE_ORDER:
        if name not in per_stage:
            continue
        row = summarise(per_stage[name], name)
        sim = simulated_flags[name]
        model = PLACEHOLDER_MODELS.get(name)
        rows.append({
            "source": "SIMULATED" if sim else "measured",
            **row,
            "analytic_median_ms": round(model.quantile(Z_P50), 4) if model else "",
            "analytic_p95_ms": round(model.quantile(Z_P95), 4) if model else "",
            "analytic_p99_ms": round(model.quantile(Z_P99), 4) if model else "",
            "harness_overhead_median_ms": round(percentile(overhead[name], 50), 6),
            "harness_overhead_p95_ms": round(percentile(overhead[name], 95), 6),
        })

    e2e = summarise(end_to_end, "end_to_end")
    sum_p95 = sum(percentile(per_stage[n], 95) for n in per_stage)
    sum_median = sum(percentile(per_stage[n], 50) for n in per_stage)
    rows.append({
        "source": "SIMULATED",
        **e2e,
        "analytic_median_ms": "",
        "analytic_p95_ms": "",
        "analytic_p99_ms": "",
        "harness_overhead_median_ms": "",
        "harness_overhead_p95_ms": "",
    })

    meta = {
        "n_turns": n_turns,
        "failures": failures,
        "sum_of_stage_p95_ms": round(sum_p95, 4),
        "sum_of_stage_median_ms": round(sum_median, 4),
        "e2e_p95_ms": e2e["p95_ms"],
        "e2e_median_ms": e2e["median_ms"],
        "p95_overstatement": round(sum_p95 / e2e["p95_ms"], 4) if e2e["p95_ms"] else "",
        "harness_overhead_median_ms": round(
            percentile(np.concatenate([overhead[n] for n in overhead if simulated_flags[n]]), 50),
            6),
    }
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--out", default=None,
                    help="default: results/latency_simulated.csv, or "
                         "results/latency_budget.csv with --real")
    ap.add_argument("--real", action="store_true",
                    help="drive the real inference binaries instead of drawing. "
                         "THIS is the command that fills results/latency_budget.md, "
                         "and it needs a board with whisper.cpp, llama.cpp and Piper "
                         "installed. Everywhere else it fails naming what is missing.")
    ap.add_argument("--model", default="models/model-q4_k_m.gguf")
    ap.add_argument("--whisper-model", default="models/ggml-tiny.en-q5_1.bin")
    ap.add_argument("--voice", default="voices/en_US-lessac-low.onnx")
    args = ap.parse_args(argv)

    try:
        rows, meta = run(n_turns=args.turns, seed=args.seed, snr_db=args.snr,
                         real=args.real, model=args.model,
                         whisper_model=args.whisper_model, voice=args.voice)
    except NoCompletedTurns as exc:
        print(exc, file=sys.stderr)
        return 1

    # Every row carries the marker; so does the filename. Belt and braces, because a
    # CSV gets copied out of its directory and into a slide.
    fields = ["source", "stage", "n", "median_ms", "p95_ms", "p99_ms", "min_ms",
              "max_ms", "mean_ms", "analytic_median_ms", "analytic_p95_ms",
              "analytic_p99_ms", "harness_overhead_median_ms", "harness_overhead_p95_ms",
              "sum_of_stage_p95_ms", "p95_overstatement", "note"]
    note = ("SIMULATED draws from src.stages.PLACEHOLDER_MODELS; NOT a Raspberry Pi 3 "
            "measurement. capture/vad rows are real host measurements.")
    if args.real:
        note = "measured: real inference binaries, real wall clock"
    for r in rows:
        r.setdefault("sum_of_stage_p95_ms", "")
        r.setdefault("p95_overstatement", "")
        r["note"] = note if r["source"] == "SIMULATED" else \
            "measured on this host (real computation, real wall clock)"
    for r in rows:
        if r["stage"] == "end_to_end":
            r["sum_of_stage_p95_ms"] = meta["sum_of_stage_p95_ms"]
            r["p95_overstatement"] = meta["p95_overstatement"]

    default_out = ("results/latency_budget.csv" if args.real
                   else "results/latency_simulated.csv")
    out = ROOT / (args.out or default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    width = max(len(r["stage"]) for r in rows)
    print(f"{'stage':<{width}}  {'src':<9} {'median_ms':>10} {'p95_ms':>10} "
          f"{'mean_ms':>10}")
    for r in rows:
        print(f"{r['stage']:<{width}}  {r['source']:<9} {r['median_ms']:>10.2f} "
              f"{r['p95_ms']:>10.2f} {r['mean_ms']:>10.2f}")
    print()
    print(f"sum of per-stage p95 : {meta['sum_of_stage_p95_ms']:.1f} ms")
    print(f"end-to-end p95       : {meta['e2e_p95_ms']:.1f} ms")
    print(f"naive sum overstates end-to-end p95 by "
          f"{100 * (meta['p95_overstatement'] - 1):.1f}%  "
          f"(measured for this model, not a general theorem)")
    print(f"harness overhead, median over simulated stages: "
          f"{1000 * meta['harness_overhead_median_ms']:.1f} us  <- real measurement, "
          f"the resolution floor of this instrument")
    print(f"\nwrote {out}")
    if args.real:
        print("Measured against real binaries. Transcribe these into "
              "results/latency_budget.md.")
    else:
        print("ALL stage latencies above except capture/vad are SIMULATED. "
              "See STATUS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
