#!/usr/bin/env python3
"""ROC and AUC in pure numpy, for every detector variant and every scenario.

sklearn is not installed and is not needed. The AUC of a score against a binary label
is the Mann-Whitney U statistic -- the probability that a randomly chosen anomaly
scores above a randomly chosen normal sample:

    AUC = ( sum of ranks of the positives - n_pos (n_pos + 1) / 2 ) / (n_pos n_neg)

with *average* ranks assigned inside groups of equal scores. The tie handling is not
cosmetic: a detector that returns the same score for everything must come out at
exactly 0.5, and one that assigns ties arbitrarily can be made to look like 1.0.

Writes:
  results/auc.csv   one row per (scenario, anomaly kind, method)
  results/roc.csv   true-positive rate on a fixed 101-point false-positive-rate grid,
                    which is what a plot needs and is 20x smaller than every threshold

Scenarios and kinds are kept separate on purpose. Pooling the out-of-subspace spikes
with the rotation segment into one "anomaly" class would average a case the method is
built for with a case it is not, and produce a number that describes neither.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.batch_svd import (DATA, RESULTS, oracle_scores,               # noqa: E402
                              read_labels, read_stream)
from oracle.chost import (TRACKER32, CDefaults, c_available,              # noqa: E402
                          run_c_tracker)
from oracle.incremental import CHECK_EVERY, REORTH_TOL, run_stream        # noqa: E402

# Effective forgetting window of 400 samples; see src/forget.c for lambda <-> window.
LAMBDA = float(np.sqrt(1.0 - 1.0 / 400.0))

SCENARIOS = (
    ("unimodal", "anomalous.csv", "labels.csv"),
    ("multimode", "multimode.csv", "multimode_labels.csv"),
    ("manymode", "manymode.csv", "manymode_labels.csv"),
)


def auc(scores: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve, tie-corrected, no sklearn.

    Returns nan when one of the two classes is empty -- AUC is undefined there, and
    returning 0.5 instead would quietly put a meaningless number in a results table.
    """
    scores = np.asarray(scores, dtype=float)
    label = np.asarray(label, dtype=int)
    n_pos = int(np.sum(label == 1))
    n_neg = int(np.sum(label == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:                      # average ranks within tie groups
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = float(np.sum(ranks[label == 1]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(scores: np.ndarray, label: np.ndarray,
              grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """TPR on a fixed FPR grid, by sweeping the threshold downward through the scores."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    lab = np.asarray(label, dtype=int)[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    n_pos, n_neg = max(int(np.sum(lab == 1)), 1), max(int(np.sum(lab == 0)), 1)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return grid, np.interp(grid, fpr, tpr)


def _restrict(scores: np.ndarray, kind: list[str], positive: str
              ) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels for `normal` versus one anomaly kind only."""
    keep = np.array([k in ("normal", positive) for k in kind])
    return scores[keep], (np.array(kind)[keep] == positive).astype(int)


def collect_methods(x: np.ndarray, label: np.ndarray, stream_path: pathlib.Path
                    ) -> dict[str, np.ndarray]:
    """Every method's per-sample score vector for one stream."""
    methods = dict(oracle_scores(x, label))
    methods["incremental, reorth"] = run_stream(
        x, lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL,
        check_every=CHECK_EVERY).scores
    methods["incremental, no reorth"] = run_stream(
        x, lam=LAMBDA, reorth=False, check_every=CHECK_EVERY).scores
    # The gap criterion picks a larger rank than the energy criterion on the multimode
    # stream (6 against 5). Including it answers the obvious objection to the failure
    # case -- "you just chose r badly" -- with a measurement rather than an argument.
    methods["incremental, rank from gap criterion"] = run_stream(
        x, lam=LAMBDA, reorth=True, rank_mode="gap", check_every=CHECK_EVERY).scores
    if c_available():
        methods["C host binary, reorth"] = run_c_tracker(
            stream_path, CDefaults(lam=LAMBDA, reorth=True)).scores
        methods["C host binary, no reorth"] = run_c_tracker(
            stream_path, CDefaults(lam=LAMBDA, reorth=False)).scores
        if c_available(TRACKER32):
            methods["C host binary, float32, reorth"] = run_c_tracker(
                stream_path, CDefaults(lam=LAMBDA, reorth=True,
                                       reorth_tol=100.0 * 1.1920929e-07),
                binary=TRACKER32).scores
    return methods


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    auc_rows, roc_rows = [], []
    if not c_available():
        print("note: build/tracker not found -- C rows omitted. Run `make host`.")

    for scenario, stream_name, label_name in SCENARIOS:
        path = DATA / stream_name
        x = read_stream(path)
        label, kind = read_labels(DATA / label_name)
        methods = collect_methods(x, label, path)
        kinds = [k for k in ("spike", "rotate") if k in set(kind)]
        for method, scores in methods.items():
            for positive in kinds:
                s, y = _restrict(scores, kind, positive)
                value = auc(s, y)
                auc_rows.append({
                    "scenario": scenario, "anomaly_kind": positive,
                    "method": method, "auc": round(value, 5),
                    "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
                })
                print(f"{scenario:10s} {positive:7s} {method:42s} AUC={value:.4f}")
                fpr, tpr = roc_curve(s, y)
                for f, t in zip(fpr, tpr):
                    roc_rows.append({
                        "scenario": scenario, "anomaly_kind": positive,
                        "method": method, "fpr": round(float(f), 4),
                        "tpr": round(float(t), 6),
                    })

    for name, rows in (("auc.csv", auc_rows), ("roc.csv", roc_rows)):
        out = RESULTS / name
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
