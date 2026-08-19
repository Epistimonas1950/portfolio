#!/usr/bin/env python3
"""The translation step, benchmarked on its own.  Writes results/translation.csv.

This is the number the brief says is worth more than the demo: the search is optimal by
construction, so the system's true error rate IS the translation error rate, and it has
to be published separately or the 100% next to A* is misleading.

WHAT IS MEASURED
  exact match      canonicalised field-by-field equality of the whole spec against the
                   hand-written gold (stops sorted by name; every field an int)
  field agreement  capacity / stop set / demands / windows / service / start, so a near
                   miss is not scored the same as nonsense
  self-flagged     did the parser warn about a construction it could not represent?  A
                   failure the system announced is operationally very different from one
                   it did not, and the residue -- wrong and silent -- is reported alone.
  silent loss      the *other* dangerous class: the spec matches the gold exactly, but
                   the request contained a constraint the schema cannot carry (an
                   ordering requirement, a conditional, a reference to yesterday's run).
                   Exact-match scoring calls these successes.  They are not.

WHAT THE NUMBER DOES NOT MEAN.  The corpus is self-authored, so it bounds nothing about
real dispatcher traffic; it measures a parser against a grammar its author also wrote.
The 20 in-grammar requests are close to a lower bound on difficulty and the 11
out-of-grammar ones are chosen adversarially, so the aggregate is an artefact of that
mix and only the per-subset numbers mean anything.  The corpus was written before the
parser; one general fix (merging repeated mentions of the same stop) was made after the
first scoring run, on an in-grammar request, and the hard subset was not touched.
"""

from __future__ import annotations

import csv
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.domain import instance_from_spec, load_gazetteer, plan_cost   # noqa: E402
from src.search import heuristics as H                                 # noqa: E402
from src.search.astar import astar                                     # noqa: E402
from src.translate import RuleBackend, field_report, load_corpus       # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def run() -> tuple[list[dict], list[dict]]:
    backend = RuleBackend()
    gaz = load_gazetteer()
    corpus = load_corpus()
    rows: list[dict] = []

    for req in corpus["requests"]:
        res = backend.translate(req["text"])
        fields = field_report(req["gold"], res.spec)
        # Does the (possibly wrong) parse at least produce a solvable instance?
        planned = ""
        if res.ok:
            try:
                inst = instance_from_spec(res.spec, gaz, name=req["id"])
                out = astar(inst, H.make("h2"))
                if out.cost is None:
                    planned = "infeasible"
                else:
                    assert plan_cost(inst, out.plan) == out.cost
                    planned = str(out.cost)
            except (KeyError, ValueError) as exc:
                planned = f"error: {exc}"
        rows.append({
            "id": req["id"], "difficulty": req["difficulty"],
            "parsed": int(res.ok), "exact": fields["exact"],
            "capacity_ok": fields["capacity"], "stop_set_ok": fields["stop_set"],
            "demands_ok": fields["demands"], "windows_ok": fields["windows"],
            "service_ok": fields["service"], "start_ok": fields["start"],
            "self_flagged": int(bool(res.warnings)),
            "warnings": "|".join(res.warnings),
            "labelled_modes": "|".join(req["failure_modes"]),
            "error": res.error or "",
            "optimal_plan_cost": planned,
        })

    modes: Counter = Counter()
    silent_loss: Counter = Counter()
    for req, row in zip(corpus["requests"], rows):
        if not row["exact"]:
            for mode in (req["failure_modes"] or ["unclassified"]):
                modes[mode] += 1
        elif req["failure_modes"]:
            for mode in req["failure_modes"]:
                silent_loss[mode] += 1
    mode_rows = [{"failure_mode": m, "wrong_parses": modes.get(m, 0),
                  "exact_but_information_lost": silent_loss.get(m, 0)}
                 for m in sorted(set(modes) | set(silent_loss))]
    return rows, mode_rows


def report(rows: list[dict], mode_rows: list[dict]) -> None:
    def rate(sub, key="exact"):
        return 100.0 * sum(r[key] for r in sub) / max(len(sub), 1)

    plain = [r for r in rows if r["difficulty"] == "plain"]
    hard = [r for r in rows if r["difficulty"] == "hard"]
    print(f"\n  in-grammar requests   {len(plain):3d}   exact match {rate(plain):6.1f}%")
    print(f"  out-of-grammar        {len(hard):3d}   exact match {rate(hard):6.1f}%")
    print(f"  whole corpus          {len(rows):3d}   exact match {rate(rows):6.1f}%")

    print("\n  field agreement over the whole corpus:")
    for key, label in (("capacity_ok", "capacity"), ("stop_set_ok", "stop set"),
                       ("demands_ok", "demands"), ("windows_ok", "time windows"),
                       ("service_ok", "service time"), ("start_ok", "start time")):
        print(f"    {label:14s} {rate(rows, key):6.1f}%")

    wrong = [r for r in rows if not r["exact"]]
    flagged = [r for r in wrong if r["self_flagged"] or r["error"]]
    print(f"\n  wrong parses: {len(wrong)}   of which self-flagged or refused: "
          f"{len(flagged)}   silent and wrong: {len(wrong) - len(flagged)}")

    print("\n  failure taxonomy:")
    print(f"    {'mode':30s} {'wrong parse':>12s} {'exact, info lost':>18s}")
    for m in mode_rows:
        print(f"    {m['failure_mode']:30s} {m['wrong_parses']:12d} "
              f"{m['exact_but_information_lost']:18d}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("Translation accuracy (RuleBackend, offline)")
    rows, mode_rows = run()
    out = RESULTS / "translation.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out2 = RESULTS / "translation_modes.csv"
    with out2.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(mode_rows[0]))
        writer.writeheader()
        writer.writerows(mode_rows)
    report(rows, mode_rows)
    print(f"\nwrote {out}\nwrote {out2}")


if __name__ == "__main__":
    main()
