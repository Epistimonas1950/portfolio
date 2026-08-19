#!/usr/bin/env python3
"""Detector accuracy against SNR, with the ZCR channel ablated.

This is the one table in this repo that is a measurement of something the repo built,
on this machine, today. Everything else in `results/` is either a template awaiting
hardware or an explicitly simulated draw.

Method
------

For each SNR and each seed, `src.synth.make_utterance` builds a signal whose speech
boundaries are known exactly by construction. The detector runs on it. Ground truth is
converted to per-frame labels with the *same* framing convention the detector uses
(`src.audio.labels_from_spans`, >=50% frame overlap), so no part of the score comes
from a units mismatch.

Reported per SNR:

  precision, recall, F1   per frame
  start_mae_ms, end_mae_ms  boundary error over ground-truth segments matched to the
                            predicted segment they overlap most
  match_rate              fraction of true segments that got any prediction at all --
                          without it a low boundary error can just mean the detector
                          found one easy word and ignored the rest
  n_pred_per_true         over-segmentation check

The `variant` column is the ablation: `full` runs the detector as configured,
`energy_only` disables the ZCR endpoint extension and changes nothing else. The
difference is what the second feature is worth, and it is reported rather than
asserted, including where it is negative.

Writes results/vad_snr_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.audio import labels_from_spans                      # noqa: E402
from src.synth import make_utterance                         # noqa: E402
from src.vad import VadConfig, boundary_errors_ms, detect, frame_metrics  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNRS = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]


def evaluate(snr_db: float, seeds: range, cfg: VadConfig) -> dict:
    """Average the per-utterance scores over seeds. Averaging the scores, not pooling
    the frames: pooling would let one long utterance dominate the mean."""
    acc: dict[str, list[float]] = {}
    unvoiced_recall: list[float] = []
    for seed in seeds:
        sig = make_utterance(seed=seed, snr_db=snr_db)
        res = detect(sig.x, sig.fs, cfg)
        n = res.frame_labels.size
        truth = labels_from_spans(sig.segments, n, res.win, res.hop, sig.fs)
        uv = labels_from_spans(sig.unvoiced_spans, n, res.win, res.hop, sig.fs)
        fm = frame_metrics(res.frame_labels, truth)
        be = boundary_errors_ms(res.segments, sig.segments)
        for k in ("precision", "recall", "f1"):
            acc.setdefault(k, []).append(fm[k])
        for k in ("match_rate", "start_mae_ms", "end_mae_ms"):
            v = be[k]
            if not np.isnan(v):
                acc.setdefault(k, []).append(float(v))
        acc.setdefault("n_pred_per_true", []).append(
            be["n_pred"] / max(1, be["n_true"]))
        unvoiced_recall.append(
            float((res.frame_labels & uv).sum()) / max(1, int(uv.sum())))
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    out["unvoiced_recall"] = float(np.mean(unvoiced_recall))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/vad_snr_sweep.csv")
    args = ap.parse_args(argv)

    seeds = range(args.seeds)
    variants = {"full": VadConfig(), "energy_only": VadConfig(zcr_extend=False)}

    rows = []
    for name, cfg in variants.items():
        for snr in SNRS:
            m = evaluate(snr, seeds, cfg)
            rows.append({
                "variant": name,
                "snr_db": snr,
                "n_seeds": args.seeds,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "start_mae_ms": round(m.get("start_mae_ms", float("nan")), 2),
                "end_mae_ms": round(m.get("end_mae_ms", float("nan")), 2),
                "match_rate": round(m.get("match_rate", 0.0), 4),
                "n_pred_per_true": round(m["n_pred_per_true"], 3),
                "unvoiced_recall": round(m["unvoiced_recall"], 4),
                "source": "measured on this host",
            })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'variant':<12}{'SNR':>6}{'prec':>8}{'recall':>8}{'F1':>8}"
          f"{'start':>8}{'end':>8}{'match':>8}{'uv_rec':>8}")
    for r in rows:
        print(f"{r['variant']:<12}{r['snr_db']:>6.0f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['start_mae_ms']:>8.1f}"
              f"{r['end_mae_ms']:>8.1f}{r['match_rate']:>8.2f}"
              f"{r['unvoiced_recall']:>8.3f}")
    print(f"\nwrote {out}  ({args.seeds} seeds per point, synthetic signals with "
          "exact ground truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
