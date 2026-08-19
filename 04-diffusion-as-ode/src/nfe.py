"""NFE accounting: the only cost model that matters for a diffusion sampler.

One network forward pass = one NFE. The batch dimension is free, because a sampler
evaluates the score on the whole batch at once, so `ScoreCounter` increments by one
per *call*, not per sample. Everything in this repo is compared at equal NFE and
never at equal step count: Heun takes half as many steps as Euler for the same NFE,
and quoting steps instead of evaluations is how second-order samplers get made to
look free.

Rejected adaptive steps count too. A step that was computed and thrown away still
cost its network evaluations, and adaptive-solver numbers that report only accepted
work are not comparable to fixed-step ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray, float], np.ndarray]


class ScoreCounter:
    """Wraps a score function and counts evaluations."""

    def __init__(self, fn: ScoreFn) -> None:
        self.fn = fn
        self.nfe = 0

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        self.nfe += 1
        return self.fn(x, t)


@dataclass
class SamplerResult:
    """What every sampler returns: the samples, and what they cost."""

    x: np.ndarray
    nfe: int
    steps: int
    accepted: int = 0
    rejected: int = 0
    times: np.ndarray | None = field(default=None, repr=False)

    @property
    def nfe_per_step(self) -> float:
        return self.nfe / self.steps if self.steps else float("nan")
