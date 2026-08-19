#!/usr/bin/env python3
"""End-to-end: a natural-language request in, an optimal plan and a briefing out.

BRIEF.md asks for a Gradio front end.  Gradio is not installed and there is no network
on this machine, so this is the same pipeline behind a command line instead of a web
page -- and unlike the Gradio version it runs today:

    natural language -> RuleBackend -> instance -> A* (proved optimal) -> English

    python3 -m src.demo
    python3 -m src.demo "Two crates to the harbour before 11:00, one to Baker Street. \
The van holds three crates."
"""

from __future__ import annotations

import sys

from .domain import instance_from_spec, load_gazetteer
from .explain import explain_no_plan
from .explain import explain_plan
from .search import heuristics as H
from .search.astar import astar
from .translate import RuleBackend

DEFAULT = ("The van holds three crates. Two crates to the harbour before 11:00, "
           "one crate to Baker Street between 09:00 and 10:00, and two to Elm Avenue.")


def main(argv: list[str]) -> int:
    text = " ".join(argv[1:]) if len(argv) > 1 else DEFAULT
    backend = RuleBackend()
    print(f'request: "{text}"\n')

    res = backend.translate(text)
    if res.warnings:
        print("parser warnings (constructions it cannot represent): "
              + ", ".join(res.warnings) + "\n")
    if not res.ok:
        print(f"translation failed: {res.error}")
        return 1
    print("parsed instance:")
    for stop in res.spec["stops"]:
        print(f"  {stop['name']:20s} demand {stop['demand']}  "
              f"window [{stop['earliest']}, {stop['latest']}]")
    print(f"  capacity {res.spec['capacity']}, service {res.spec['service_minutes']} "
          f"min, reload {res.spec['reload_minutes']} min\n")

    inst = instance_from_spec(res.spec, load_gazetteer(), name="demo")
    out = astar(inst, H.make("h2"))
    if out.cost is None:
        print(explain_no_plan(inst))
        return 0
    print(explain_plan(inst, out.plan, out.cost, "h2-mst", out.stats.expansions))
    print(f"\n[{out.stats.expansions} expansions, {out.stats.generated} generated, "
          f"{out.stats.re_expansions} re-expansions, {out.stats.seconds * 1e3:.1f} ms]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
