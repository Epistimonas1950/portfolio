"""A neurosymbolic planner: the language model proposes, exact search guarantees.

The formal object everywhere in this package is a single-vehicle delivery problem with
hard time windows and a capacity that forces depot reloads. A state is

    s = (loc, U, t, cap)

-- current location, bitmask of undelivered parcels, clock, remaining load -- and the
cost of a plan is the clock reading when the vehicle is back at the depot with U empty:

    minimize  t_final     subject to   arrive(p) <= latest(p)  for every parcel p,
                                       load never negative,
                                       every parcel delivered exactly once.

Because g(s) = t and every edge cost is non-negative, the planning problem is a
shortest-path problem on a finite graph, and A* with an admissible heuristic returns
the exact optimum. That guarantee is the product. The natural-language front end in
`translate` is the only component that can be wrong, and it is measured separately.
"""
