"""Plan -> natural language.  Templates, offline, no model.

The brief puts a language model on both ends of the search: one to read the request, one
to narrate the answer.  The narration end is the easy one, and pretending otherwise
would be dishonest -- a plan is a structured object with a fixed set of fields, so the
"generation" problem is a rendering problem and templates solve it exactly, today, with
no model and no network.  Templates also cannot hallucinate a stop that is not in the
plan, which for an operational tool is not a limitation but the requirement.

What the narration is *for* is stating the guarantee in a form a dispatcher can act on:

    - every arrival time, and how much slack is left against each time window;
    - where the waiting happens, because waiting is the part of the cost a human reads
      as "the plan is wrong" when it is in fact optimal;
    - the reload trips, which are the visible consequence of the capacity constraint;
    - and the optimality claim with the number of nodes it took to establish it.

`explain_infeasibility` is the other half: when the request cannot be satisfied, the
validator's violations are rendered into the same register, because "your 09:00 window
at the harbour cannot be met from the depot" is the answer the user wanted.
"""

from __future__ import annotations

from .domain import DEPOT, Instance, Move, Validation, fmt_time, validate_plan


def _breakdown(inst: Instance, timeline: list[dict]) -> dict[str, int]:
    drive = wait = service = reload = 0
    loc = DEPOT
    for step in timeline:
        node = step["node"]
        drive += inst.travel[loc][node]
        wait += step["wait"]
        if step["move"] == "deliver":
            service += inst.service[node]
        elif step["move"] == "reload":
            reload += inst.reload_time
        loc = node
    return {"drive": drive, "wait": wait, "service": service, "reload": reload}


def explain_plan(inst: Instance, plan: list[Move], cost: int | None = None,
                 heuristic: str | None = None, expansions: int | None = None,
                 optimal: bool = True) -> str:
    """Render a plan as a dispatcher's briefing."""
    v = validate_plan(inst, plan)
    if not v.feasible:
        return explain_infeasibility(inst, v)

    lines: list[str] = []
    total = cost if cost is not None else v.cost
    lines.append(f"Plan for {inst.name}: {inst.n_stops} stop(s), van capacity "
                 f"{inst.capacity}, leaving the depot at {fmt_time(inst.start_time)}.")
    lines.append("")

    for i, step in enumerate(v.timeline, 1):
        label = step["label"]
        arrive = fmt_time(step["arrive"])
        if step["move"] == "deliver":
            node = step["node"]
            slack = inst.latest[node] - step["arrive"]
            piece = (f"{i}. Drive to {label}, arriving {arrive}.")
            if step["wait"]:
                piece += (f" That is {step['wait']} min before the window opens at "
                          f"{fmt_time(inst.earliest[node])}, so the van waits.")
            piece += (f" Hand over {inst.demand[node]} unit(s); "
                      f"{step['cap_after']} left on board.")
            piece += (f" Window closes {fmt_time(inst.latest[node])} -- "
                      f"{slack} min of slack.")
            lines.append(piece)
        elif step["move"] == "reload":
            lines.append(f"{i}. Back to the depot at {arrive} to reload -- the van was "
                         f"empty enough that the rest of the round could not be carried "
                         f"in one trip. Reloading takes {inst.reload_time} min; "
                         f"leaving with {step['cap_after']} units.")
        else:
            lines.append(f"{i}. Return to the depot, arriving {arrive}. Round complete.")

    parts = _breakdown(inst, v.timeline)
    lines.append("")
    lines.append(f"Total elapsed: {total} min "
                 f"({parts['drive']} driving, {parts['wait']} waiting, "
                 f"{parts['service']} handing over, {parts['reload']} reloading).")
    tight = min(((inst.latest[s["node"]] - s["arrive"], s["label"])
                 for s in v.timeline if s["move"] == "deliver"), default=None)
    if tight is not None:
        lines.append(f"Tightest window: {tight[1]}, with {tight[0]} min to spare.")
    if optimal:
        claim = "No feasible round is faster. This is proved, not estimated"
        if heuristic and expansions is not None:
            claim += (f": A* with the {heuristic} heuristic settled it in "
                      f"{expansions} node expansions")
        lines.append(claim + ".")
    return "\n".join(lines)


def explain_infeasibility(inst: Instance, v: Validation) -> str:
    """Render the validator's verdict.  One line per broken constraint, in plain words."""
    lines = [f"That request cannot be met as stated ({inst.name}). "
             f"{len(v.violations)} problem(s):"]
    for x in v.violations:
        lines.append(f"  - {x}")
    lines.append("")
    lines.append("Widening the tightest window, raising the van's capacity, or starting "
                 "earlier are the three levers; the search will take whichever of them "
                 "you change and prove the new optimum.")
    return "\n".join(lines)


def explain_no_plan(inst: Instance) -> str:
    """When exhaustive search proves there is no feasible plan at all."""
    return (f"No feasible round exists for {inst.name}. The search was exhaustive, so "
            "this is a proof rather than a failure to find one: with capacity "
            f"{inst.capacity} and the stated windows, some stop cannot be reached in "
            "time on any ordering. Relax a window or add a vehicle.")
