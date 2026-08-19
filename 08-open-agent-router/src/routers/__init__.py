"""Routing policies. Every one satisfies the same two-method contract.

    select(x, expected_costs) -> arm index
    update(x, arm, reward, cost) -> None

`x` is the serving-time context (src/features.py); `expected_costs` is the (K,) vector
of costs the fleet advertises for this query, which a router legitimately knows before
dispatching -- you do not need to call a 32B model to know it is slower than a 3B one.
Feedback is bandit feedback: `reward` and `cost` arrive only for the arm that was
actually pulled.

Nothing in this package imports the simulator. That is the point: the same policy code
runs against src/fleet/client.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Router(Protocol):
    name: str

    def select(self, x: np.ndarray, expected_costs: np.ndarray) -> int:
        ...

    def update(self, x: np.ndarray, arm: int, reward: float, cost: float) -> None:
        ...
