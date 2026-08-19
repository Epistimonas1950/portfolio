"""A minimal multi-step agent loop, so the compounding numbers come from episodes.

Why this exists. BRIEF.md's compounding argument is arithmetic: n independent steps at
per-step success p give p^n end to end, so 0.95 over ten steps is 0.60 and 0.90 is 0.35.
That is the quantitative case for routing in an agent rather than a chatbot. But quoting
p^n and then reporting p^n as the "measured" end-to-end rate measures nothing. The gap
between the two is the interesting quantity, and to have a gap you need real episodes.

So: an episode is a sequence of steps; each step is a tool call the agent must get
right; the router picks which arm handles that step; the simulated fleet decides whether
the arm emitted a correct call; and then the tool *actually runs* (src/agent/tools.py)
and its output is compared to the ground-truth answer. The episode succeeds only if
every step's real output is right. No formula anywhere in the success accounting.

Where the correlation comes from, and which way it points. Every step of an episode
inherits a shared episode-level difficulty, plus per-step jitter. Positive dependence
across steps means the end-to-end rate is E_d[ p(d)^n ], and by Jensen's inequality
applied to the convex map q -> q^n,

    E_d[ p(d)^n ]  >=  ( E_d[ p(d) ] )^n  =  p_bar^n.                                (1)

So correlated steps make the measured end-to-end rate come out **above** the p^n
prediction, not below: an agent's failures bunch into a minority of hard episodes rather
than being spread evenly, and the independence formula is pessimistic. That is the
opposite of the intuition most people have, so `independent_steps=True` runs the same
loop with each step's difficulty drawn afresh -- the control in which the measurement
must land on p^n, and which is what makes the correlated number believable.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from ..fleet.simulator import Arm, success_probability
from .tools import ToolCall, ToolError, dispatch

DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass(frozen=True)
class Step:
    """One tool call the agent must emit correctly, and the answer it must produce."""

    tool: str
    args: dict
    truth: object
    difficulty: float


@dataclass
class Episode:
    steps: tuple[Step, ...]
    seed: int


@dataclass
class EpisodeResult:
    arms: list[int] = field(default_factory=list)
    step_ok: list[bool] = field(default_factory=list)
    seconds: float = 0.0
    succeeded: bool = False
    failed_at: int = -1


def ensure_corpus(root: pathlib.Path = DATA_ROOT, seed: int = 7,
                  n_files: int = 6, n_lines: int = 40) -> pathlib.Path:
    """Write a small deterministic corpus for the file tools to work on.

    Generated rather than committed so the repo carries no binaries and no text nobody
    wrote; deterministic from `seed` so the tool answers are reproducible and the tests
    never depend on anything outside the repo.
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    words = ["router", "bandit", "conformal", "budget", "oracle", "regret",
             "cascade", "escalate", "tokens", "latency"]
    for i in range(n_files):
        path = root / f"notes_{i:02d}.txt"
        if path.exists():
            continue
        lines = [" ".join(rng.choice(words, size=int(rng.integers(3, 9))))
                 for _ in range(n_lines)]
        path.write_text("\n".join(lines) + "\n")
    return root


def make_episode(rng: np.random.Generator, n_steps: int, root: pathlib.Path,
                 episode_difficulty: float, independent_steps: bool,
                 step_spread: float = 0.10) -> Episode:
    """Build one episode of real tool tasks with a shared or independent difficulty."""
    files = sorted(p.name for p in root.glob("notes_*.txt"))
    words = ["router", "bandit", "conformal", "budget", "oracle", "regret",
             "cascade", "escalate", "tokens", "latency"]
    steps = []
    for _ in range(n_steps):
        if independent_steps:
            d = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        else:
            d = float(np.clip(episode_difficulty + rng.normal(0.0, step_spread), 0.0, 1.0))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            a, b, c = (int(rng.integers(2, 40)) for _ in range(3))
            expr = f"({a} + {b}) * {c}"
            steps.append(Step("calculate", {"expression": expr}, float((a + b) * c), d))
        elif kind == 1:
            name = str(rng.choice(files))
            needle = str(rng.choice(words))
            truth = sum(1 for line in (root / name).read_text().splitlines()
                        if needle in line)
            steps.append(Step("search_text", {"needle": needle, "path": name}, truth, d))
        else:
            name = str(rng.choice(files))
            truth = len((root / name).read_text().splitlines())
            steps.append(Step("read_file", {"path": name}, truth, d))
    return Episode(steps=tuple(steps), seed=int(rng.integers(1 << 30)))


def _corrupt(call: ToolCall, rng: np.random.Generator) -> ToolCall:
    """What a failed model call looks like: a plausible but wrong tool call.

    Three failure shapes, because they have different consequences downstream and an
    agent loop that only ever sees one of them is not measuring error handling:
    a malformed expression (tool raises), a wrong argument (tool succeeds, answer is
    wrong), and the wrong tool entirely (tool raises).
    """
    mode = int(rng.integers(0, 3))
    if mode == 0:
        # Malformed argument: the tool raises. The loop sees an exception, not a value.
        return ToolCall(call.tool, {**call.args, "expression": "1 +"} if
                        call.tool == "calculate" else {**call.args, "path": "nope.txt"})
    if mode == 1:
        # Plausible but wrong argument: the tool *succeeds* and returns a wrong answer,
        # which is the failure mode an agent loop cannot detect from the tool's response
        # and the one that actually propagates.
        args = dict(call.args)
        if "expression" in args:
            args["expression"] = args["expression"].replace("+", "-", 1)
            return ToolCall(call.tool, args)
        if "needle" in args:
            args["needle"] = args["needle"] + "_x"
            return ToolCall(call.tool, args)
        return ToolCall("search_text", {**args, "needle": "zzz"})
    # Wrong tool entirely: dispatch raises.
    return ToolCall("no_such_tool", call.args)


def run_episode(episode: Episode, router, featurize_step, arms: tuple[Arm, ...],
                rng: np.random.Generator, root: pathlib.Path,
                lam: float = 0.05, learn: bool = True) -> EpisodeResult:
    """Execute one episode. Real tools, simulated decision about call correctness."""
    res = EpisodeResult()
    ok_so_far = True
    for i, step in enumerate(episode.steps):
        x, expected_costs = featurize_step(step, i)
        arm = router.select(x, expected_costs)
        p = float(success_probability(np.array([step.difficulty]), arms)[0, arm])
        emitted_ok = bool(rng.random() < p)

        call = ToolCall(step.tool, dict(step.args))
        if not emitted_ok:
            call = _corrupt(call, rng)

        try:
            out = dispatch(call, root)
            if step.tool == "read_file" and call.tool == "read_file":
                out = len(str(out).splitlines())
            step_ok = _matches(out, step.truth)
        except ToolError:
            step_ok = False

        seconds = float(arms[arm].base_seconds + arms[arm].seconds_per_token * 90)
        res.seconds += seconds
        res.arms.append(arm)
        res.step_ok.append(step_ok)
        if learn:
            router.update(x, arm, float(step_ok) - lam * seconds, seconds)
        if not step_ok and ok_so_far:
            ok_so_far = False
            res.failed_at = i
    res.succeeded = ok_so_far
    return res


def _matches(out, truth) -> bool:
    if isinstance(truth, float):
        return isinstance(out, (int, float)) and abs(float(out) - truth) < 1e-9
    return out == truth
