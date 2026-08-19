"""The real-fleet interface: the same shape as the simulator, over a real server.

Nothing in this file runs on this machine, and it says so loudly rather than degrading
into a mock. There is no Ollama, no llama.cpp, no vLLM, no weights and no network here.
The module exists to make one claim checkable: the routing policy code in
`src/routers/` never touches a simulator-specific attribute, so pointing it at a real
fleet is a change of constructor and nothing else.

    generate(prompt, arm) -> (text, CallCost)

is the whole contract. `CallCost` (src/cost.py) is what the router is billed, and it is
filled here with *measured* wall-clock and *reported* peak memory -- which is the
honest instrumentation boundary: seconds you can time from the client, resident memory
you have to ask the server for, and if the server will not tell you, the field stays
NaN rather than being guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..cost import CallCost


class FleetClient(Protocol):
    """What a router needs from a fleet. The simulator satisfies this too."""

    arm_names: tuple[str, ...]

    def generate(self, prompt: str, arm: int) -> tuple[str, CallCost]:
        ...


@dataclass
class InstrumentedCall:
    """One recorded call, so a real run can be replayed offline into the same evals."""

    arm: str
    prompt_chars: int
    text: str
    cost: CallCost


class OllamaClient:
    """Arms served by a local Ollama daemon. Not runnable in this environment.

    Kept as a class rather than a comment so that the interface is type-checkable and
    so the failure names precisely what is missing. The brief allows llama.cpp, Ollama
    or vLLM; the swap is the `generate` body and the fleet spec in serve/fleet.yaml.
    """

    def __init__(self, arm_names: tuple[str, ...], host: str = "http://localhost:11434",
                 timeout_s: float = 120.0):
        self.arm_names = arm_names
        self.host = host
        self.timeout_s = timeout_s
        self.calls: list[InstrumentedCall] = []

    def generate(self, prompt: str, arm: int) -> tuple[str, CallCost]:
        raise NotImplementedError(
            "OllamaClient needs a running Ollama daemon and pulled model weights, and "
            f"this machine has neither.\n"
            f"  wanted: POST {self.host}/api/generate with model={self.arm_names[arm]!r}\n"
            "  needs : `ollama serve`, `ollama pull <model>` for each arm in "
            "serve/fleet.yaml, and network access to fetch the weights (tens of GB).\n"
            "  also  : peak-memory instrumentation, which on Ollama means reading\n"
            "          /api/ps per call, or nvidia-smi if you are on a GPU.\n"
            "Use src.fleet.simulator.make_workload() instead; every result in this "
            "repo is a simulated-fleet result and is labelled as such.")

    def _time_call(self, fn, peak_memory_gb: float) -> tuple[str, CallCost]:
        """The instrumentation half, which is real code and would be reused verbatim.

        perf_counter rather than time(): we are measuring an interval on one machine,
        and a wall-clock adjustment mid-call would corrupt the cost that the budgeted
        router is billed.
        """
        t0 = time.perf_counter()
        text, prompt_tokens, completion_tokens = fn()
        elapsed = time.perf_counter() - t0
        return text, CallCost(seconds=elapsed, peak_memory_gb=peak_memory_gb,
                              prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens)


def load_fleet_spec(path: str) -> dict:
    """Read serve/fleet.yaml. PyYAML is installed; the models it names are not."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)
