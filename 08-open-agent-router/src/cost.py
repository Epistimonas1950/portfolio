"""The cost model, behind one interface, because the policy must not know what a bill is.

Every model in this fleet is open-weight and runs on hardware you already own, so there
are no dollars and "cost-optimal" needs a definition before it means anything. The
quantity a company actually pays for is *occupancy of the accelerator*: how long a call
holds the device, times how much of the device the model's resident weights hold. So the
unit of cost here is

    price(call)  =  seconds * ( 1 + memory_weight * peak_memory_gb )

with `memory_weight = 0` by default, which reduces it to wall-clock seconds -- the
honest default when the fleet is served one model at a time and the card is reserved
regardless. Set `memory_weight > 0` for the shared-accelerator regime, where a 19 GB
arm and a 2 GB arm holding the card for the same second are not the same expense; the
resulting unit is GB-seconds, which is what a scheduler actually allocates.

The point of this module is the *interface*, not the formula. Every policy in
`src/routers/` sees a fleet only through

    CostModel.price(CallCost) -> float

so swapping to a hosted fleet priced per token means constructing a
`TokenPriceCostModel` with a $/1k-token table and changing nothing else. The bandit
code below never learns what a second is; it learns a scalar called cost. That
separation is the whole reason the budgeted router's dual variable `p*` has units of
"quality per unit cost" and stays meaningful when the unit changes.

All numbers produced through `ComputeCostModel` in this repo come from the simulated
fleet (`src/fleet/simulator.py`). No real GPU-second was measured anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class CallCost:
    """What one model call consumed. The real client fills the same fields."""

    seconds: float
    peak_memory_gb: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostModel(Protocol):
    """The only thing a routing policy is allowed to know about money."""

    unit: str

    def price(self, cost: CallCost) -> float:
        ...


@dataclass(frozen=True)
class ComputeCostModel:
    """Wall-clock seconds, optionally weighted by resident memory.

    memory_weight: 0 gives pure seconds. A value of 1/24 says "one second on a 24 GB
    card counts double if the model fills it", i.e. normalise GB by a reference device
    so the number stays order-one and comparable to a bare second. It is a modelling
    choice and it is exposed rather than buried, because it changes which arm is
    cheapest and therefore what the oracle policy is.
    """

    memory_weight: float = 0.0
    unit: str = "GPU-seconds (simulated)"

    def price(self, cost: CallCost) -> float:
        return cost.seconds * (1.0 + self.memory_weight * cost.peak_memory_gb)


@dataclass(frozen=True)
class TokenPriceCostModel:
    """The hosted-fleet swap-in: a $/1k-token table, same interface, no policy changes.

    `prompt_usd_per_1k` and `completion_usd_per_1k` are keyed by arm name. This class
    exists to demonstrate that the cost abstraction holds -- it is *not* used to produce
    any result in this repo, because that would require a price list and token counts
    from a real provider, and this machine has neither.
    """

    prompt_usd_per_1k: Mapping[str, float]
    completion_usd_per_1k: Mapping[str, float]
    arm_name: str
    unit: str = "USD"

    def price(self, cost: CallCost) -> float:
        return (self.prompt_usd_per_1k[self.arm_name] * cost.prompt_tokens
                + self.completion_usd_per_1k[self.arm_name] * cost.completion_tokens) / 1000.0


DEFAULT_COST_MODEL = ComputeCostModel()
