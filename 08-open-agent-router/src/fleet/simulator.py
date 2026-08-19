"""A simulated fleet of open-weight models whose ground truth is known.

Why simulate. No model of any kind can run on this machine -- no Ollama, no llama.cpp,
no weights, no network. That constraint turns out to be the right instrument rather
than a workaround, for three reasons that matter to the mathematics:

  1. The oracle policy is *computable*. Regret is defined against the best achievable
     policy; on a real fleet you can only estimate that, so "regret" becomes a number
     you fit rather than a number you measure. Here the per-query success probability
     of every arm is known in closed form, so the oracle is exact and regret is a
     measurement.
  2. Coverage can be checked against a known conditional distribution. Split conformal
     promises marginal coverage under exchangeability; to show it *breaks* under
     distribution shift you have to be able to shift the distribution on purpose.
  3. The capability/cost ladder can be dialled. With all arms equally good there is
     nothing to route, and with one arm dominating there is nothing to learn; the
     experiment needs the ladder as a controlled variable, not as whatever three
     checkpoints happened to be on disk.

Everything this module emits is therefore a *simulated-fleet* number, and is labelled
as such everywhere it is reported. `src/fleet/client.py` is the same interface over a
real server; the routers cannot tell the difference.

The generative model
--------------------
Each query t carries a latent difficulty d_t in [0,1], a task type, a tool-call depth,
a prompt length, and a true answer label y_t in {0..L-1}. Arm k has a skill s_k on the
same difficulty scale. Its success probability is the logistic ladder

    p_k(d)  =  sigma( beta * (s_k - d) + b0 )                                     (1)

which is strictly increasing in s_k for every d -- a genuine capability ladder, not a
coin flip -- and strictly decreasing in d.

Success indicators are *correlated across arms*, because a query that confuses a small
model usually confuses a big one too. There are two separate channels for that, and
keeping them apart matters:

  shared difficulty     every arm's p_k is a function of the same d_t, so hard queries
                        are hard for everyone. This channel is always on and cannot be
                        switched off without dismantling the routing problem -- it is
                        why routing is possible at all.
  the rho coupling      on top of that, with probability rho all arms are scored against
                        one shared uniform u_t; otherwise each arm draws its own. This
                        adds dependence that survives conditioning on d_t: arms failing
                        together on the *same* query, at fixed difficulty.

The mixture is exact, so marginally every arm still succeeds with probability precisely
p_k(d_t) -- the coupling changes the dependence, not the margins -- and rho = 1 makes the
ladder monotone, so "the cheapest arm that succeeds" is well defined. The consequence
worth flagging: even at rho = 0 an *unconditional* independence check on this simulator
fails (measured joint-failure ratio 4.2), because the shared difficulty is still there.
Conditional on a narrow difficulty band, rho = 0 really is independent (ratio 1.03).
tests/test_fleet.py asserts both.

Cost rises with capability:

    seconds_k  =  ( base_k + per_token_k * n_tokens ) * lognormal noise              (2)

with peak resident memory fixed per arm. Both are wall-clock quantities in the sense of
src/cost.py; neither was measured on hardware.

Confidence, for the conformal machinery
---------------------------------------
Each call also emits a probability vector over the L candidate answers. The emitted
argmax is the true label exactly when the call succeeded, so accuracy and confidence
come from one coherent draw. The peakedness of that vector is

    gap_k  ~  Exponential( mean = g0 + g1 * max(0, s_k - d) ) * (fail_ratio_k if wrong)

so a model is on average less confident when it is wrong -- but only *on average*, and
the small arm's `fail_ratio` is close to 1, which is what overconfidence means. That is
the whole reason a calibrated escalation rule is worth having: the raw softmax margin
is not comparable across tiers, and split conformal fixes exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cost import CallCost, CostModel, ComputeCostModel

N_LABELS = 8            # candidate answers per query; sets the conformal label space
TASK_TYPES = ("qa", "tool_call", "code")


@dataclass(frozen=True)
class Arm:
    """One model in the fleet. All numbers are simulated stand-ins, not measurements."""

    name: str
    tier: str                  # small | mid | large
    params_b: float            # nominal parameter count in billions -- a label only
    skill: float               # capability, on the same [0,1] scale as difficulty
    base_seconds: float        # fixed per-call overhead (prefill, scheduling)
    seconds_per_token: float   # decode rate
    peak_memory_gb: float      # resident weights + KV cache
    conf_scale: float          # g1 in the confidence gap above
    fail_conf_ratio: float     # how much confidence survives being wrong (1 = fully
                               # overconfident). Decreases with capability.


# A three-tier ladder in the shape the brief asks for (1-4B / 7-8B / ~30B quantized).
# The parameter counts are labels; the timings are simulated and deliberately span an
# order of magnitude, which is the regime in which routing is worth doing at all.
DEFAULT_FLEET: tuple[Arm, ...] = (
    Arm("small", "small", 3.0, skill=0.38, base_seconds=0.22,
        seconds_per_token=0.0035, peak_memory_gb=2.2, conf_scale=1.5,
        fail_conf_ratio=0.85),
    Arm("mid", "mid", 8.0, skill=0.60, base_seconds=0.40,
        seconds_per_token=0.0075, peak_memory_gb=5.4, conf_scale=2.3,
        fail_conf_ratio=0.68),
    Arm("large", "large", 32.0, skill=0.95, base_seconds=0.85,
        seconds_per_token=0.0210, peak_memory_gb=18.6, conf_scale=3.4,
        fail_conf_ratio=0.52),
)

# The three constants that set the shape of the problem. They were chosen by sweeping
# for a *balanced oracle*: with these values the exact expected-reward oracle sends
# roughly 23% / 56% / 21% of queries to the three arms (verified in tests). That
# balance is the point. A fleet on which the oracle always picks one arm has nothing
# to route and would make every policy below look identical; a fleet on which the arms
# are interchangeable has nothing to learn. The routing problem has to be non-trivial
# before any statement about routing means anything, and that is a property of the
# instrument, so it is set deliberately and checked rather than hoped for.
BETA = 6.0        # logistic slope in (1): how sharply success falls off with difficulty
B0 = 2.2          # logistic offset; sets the absolute accuracy level of the ladder
LAMBDA = 0.12     # price of one second of compute, in units of "probability of success"


@dataclass
class Workload:
    """A batch of queries plus every ground truth the simulator knows about them.

    Serving-time quantities (what a router may legally look at): prompt_tokens,
    task_type, tool_depth, difficulty_score, and the fleet's advertised expected cost.
    Everything else -- difficulty, label, p_success, success, probs -- is ground truth
    the simulator exposes so that the oracle and the regret are exact.
    """

    difficulty: np.ndarray        # (T,)   latent, NOT available at serving time
    difficulty_score: np.ndarray  # (T,)   noisy classifier output, IS available
    task_type: np.ndarray         # (T,)   int index into TASK_TYPES
    tool_depth: np.ndarray        # (T,)
    prompt_tokens: np.ndarray     # (T,)
    completion_tokens: np.ndarray # (T,)
    label: np.ndarray             # (T,)   true answer index, ground truth
    p_success: np.ndarray         # (T,K)  exact success probability of every arm
    success: np.ndarray           # (T,K)  realized draw for every arm (bool)
    seconds: np.ndarray           # (T,K)  realized wall-clock, every arm
    probs: np.ndarray             # (T,K,L) emitted answer distribution, every arm
    arms: tuple[Arm, ...]

    def __len__(self) -> int:
        return int(self.difficulty.shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def call_cost(self, t: int, arm: int) -> CallCost:
        a = self.arms[arm]
        return CallCost(seconds=float(self.seconds[t, arm]),
                        peak_memory_gb=a.peak_memory_gb,
                        prompt_tokens=int(self.prompt_tokens[t]),
                        completion_tokens=int(self.completion_tokens[t]))

    def cost_matrix(self, model: CostModel | None = None) -> np.ndarray:
        """(T, K) realized price of every arm on every query, in the model's unit."""
        model = model or ComputeCostModel()
        mem = np.array([a.peak_memory_gb for a in self.arms])
        # Inlined form of ComputeCostModel.price for the whole matrix at once; the
        # scalar path above is what the agent loop and the client actually call.
        if isinstance(model, ComputeCostModel):
            return self.seconds * (1.0 + model.memory_weight * mem[None, :])
        return np.array([[model.price(self.call_cost(t, k)) for k in range(self.n_arms)]
                         for t in range(len(self))])

    def expected_cost_matrix(self, model: CostModel | None = None) -> np.ndarray:
        """(T, K) cost a router can *anticipate* before calling -- the noise-free part.

        A router legitimately knows that the 32B arm is slower than the 3B arm before
        it dispatches; it does not know this call's lognormal jitter. The budgeted
        router plans against this matrix and is billed against `cost_matrix`.
        """
        model = model or ComputeCostModel()
        mem = np.array([a.peak_memory_gb for a in self.arms])
        base = np.array([a.base_seconds for a in self.arms])
        rate = np.array([a.seconds_per_token for a in self.arms])
        secs = base[None, :] + rate[None, :] * self.completion_tokens[:, None]
        if isinstance(model, ComputeCostModel):
            return secs * (1.0 + model.memory_weight * mem[None, :])
        raise NotImplementedError(
            "expected_cost_matrix is only defined for ComputeCostModel; a token-price "
            "model needs an expected token count, which this simulator does not model.")


def success_probability(difficulty: np.ndarray, arms: tuple[Arm, ...]) -> np.ndarray:
    """Equation (1): the logistic capability ladder, (T,) -> (T, K)."""
    skills = np.array([a.skill for a in arms])
    z = BETA * (skills[None, :] - np.asarray(difficulty)[:, None]) + B0
    return 1.0 / (1.0 + np.exp(-z))


def make_workload(n: int = 20_000, seed: int = 0, arms: tuple[Arm, ...] = DEFAULT_FLEET,
                  rho: float = 0.65, score_noise: float = 0.12,
                  difficulty_shift: float = 0.0) -> Workload:
    """Draw a workload of `n` queries against `arms`.

    rho: probability that a query is "common mode", i.e. all arms are scored against
         the same uniform. This is the arm-to-arm failure correlation knob. It leaves
         every marginal success probability exactly p_k(d) -- verified in the tests --
         so it changes only the dependence structure, which is what the cascade's
         union-bound slack is sensitive to.
    score_noise: standard deviation of the serving-time difficulty classifier's error.
         0 would make the routing problem trivially solvable and the bandit pointless;
         this is the single knob that creates the model-misspecification floor reported
         in results/regret.csv.
    difficulty_shift: added to the Beta(2,2) difficulty draw before clipping. Used only
         to break exchangeability on purpose in eval/coverage.py.
    """
    rng = np.random.default_rng(seed)
    k = len(arms)

    difficulty = np.clip(rng.beta(2.0, 2.0, size=n) + difficulty_shift, 0.0, 1.0)
    task_type = rng.integers(0, len(TASK_TYPES), size=n)
    # Tool-calling and code queries are harder at the same nominal difficulty; this is
    # what gives the task-type features something to carry.
    difficulty = np.clip(difficulty + np.array([0.0, 0.06, 0.10])[task_type], 0.0, 1.0)

    tool_depth = 1 + rng.poisson(1.2, size=n) * (task_type != 0)
    prompt_tokens = np.maximum(24, rng.lognormal(5.2, 0.55, size=n).astype(int))
    completion_tokens = np.maximum(8, rng.lognormal(4.4, 0.5, size=n).astype(int))

    # The serving-time difficulty classifier: the latent difficulty plus noise, clipped
    # back into [0,1]. This is the only view of `difficulty` a router ever gets.
    difficulty_score = np.clip(difficulty + rng.normal(0.0, score_noise, size=n), 0.0, 1.0)

    label = rng.integers(0, N_LABELS, size=n)
    p_success = success_probability(difficulty, arms)

    # Correlated success draws: exact uniform mixture (see module docstring).
    common = rng.random(n) < rho
    shared_u = rng.random(n)
    own_u = rng.random((n, k))
    u = np.where(common[:, None], shared_u[:, None], own_u)
    success = u < p_success

    # Cost, equation (2). Lognormal jitter with sigma 0.15 -- serving latency is
    # right-skewed (a slow batch, a cache miss), never negative, so lognormal rather
    # than Gaussian.
    base = np.array([a.base_seconds for a in arms])
    rate = np.array([a.seconds_per_token for a in arms])
    jitter = rng.lognormal(0.0, 0.15, size=(n, k))
    seconds = (base[None, :] + rate[None, :] * completion_tokens[:, None]) * jitter

    probs = _emit_answer_distributions(rng, difficulty, label, success, arms)

    return Workload(difficulty=difficulty, difficulty_score=difficulty_score,
                    task_type=task_type, tool_depth=tool_depth,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    label=label, p_success=p_success, success=success, seconds=seconds,
                    probs=probs, arms=arms)


def _emit_answer_distributions(rng: np.random.Generator, difficulty: np.ndarray,
                               label: np.ndarray, success: np.ndarray,
                               arms: tuple[Arm, ...]) -> np.ndarray:
    """(T, K, L) softmax over candidate answers, argmax consistent with `success`.

    Construction: draw a shared per-query confusion vector (so the arms agree about
    which wrong answers are tempting -- correlated errors are what makes a cascade's
    union bound loose), add arm-specific noise, then raise the *emitted* answer above
    every other logit by an exponential gap. Raising rather than swapping keeps the
    argmax exact without distorting the rest of the vector.
    """
    n, k = success.shape
    lab_l = N_LABELS

    shared = rng.normal(0.0, 0.8, size=(n, 1, lab_l))
    own = rng.normal(0.0, 0.6, size=(n, k, lab_l))
    logits = shared + own

    # Which answer each arm emits: the true label on success, a uniformly drawn wrong
    # label on failure.
    wrong = (label[:, None] + 1 + rng.integers(0, lab_l - 1, size=(n, k))) % lab_l
    emitted = np.where(success, label[:, None], wrong)

    skills = np.array([a.skill for a in arms])
    scale = np.array([a.conf_scale for a in arms])
    ratio = np.array([a.fail_conf_ratio for a in arms])
    mean_gap = 0.35 + scale[None, :] * np.maximum(0.0, skills[None, :] - difficulty[:, None])
    mean_gap = mean_gap * np.where(success, 1.0, ratio[None, :])
    gap = rng.exponential(np.maximum(mean_gap, 1e-6))

    rows, cols = np.indices((n, k))
    logits[rows, cols, emitted] = logits.max(axis=2) + gap

    logits = logits - logits.max(axis=2, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=2, keepdims=True)


# --------------------------------------------------------------------------------
# Oracles. The simulator knows the ground truth, so these are exact, not estimated.
# --------------------------------------------------------------------------------

def expected_reward_matrix(w: Workload, lam: float = LAMBDA,
                           model: CostModel | None = None) -> np.ndarray:
    """(T, K) E[ r | x, a ] = p_a(d) - lambda * E[cost].

    This is the quantity every regret number in the repo is defined against. It uses
    the *expected* cost, not the realized one, because the realized lognormal jitter is
    noise no policy can anticipate and charging it to the policy would put a random
    walk into the regret curve.
    """
    return w.p_success - lam * w.expected_cost_matrix(model)


def oracle_expected(w: Workload, lam: float = LAMBDA,
                    model: CostModel | None = None) -> np.ndarray:
    """The regret benchmark: argmax_a E[r | x_t, a], knowing the latent difficulty.

    This is the best any policy could do with full knowledge of the query's difficulty
    and the fleet's response curves. No learned router can beat it in expectation, and
    the test suite asserts exactly that.
    """
    return np.argmax(expected_reward_matrix(w, lam, model), axis=1)


def oracle_hindsight(w: Workload, lam: float = LAMBDA,
                     model: CostModel | None = None) -> np.ndarray:
    """The Pareto-frontier row: argmax_a ( success_{t,a} - lambda * cost_{t,a} ).

    With realized outcomes in hand this is literally "the cheapest arm that succeeds"
    whenever the cost spread across the fleet is worth less than one success, i.e.
    lambda * (c_max - c_min) < 1, which holds for the default fleet. It is not
    achievable by anything -- it reads the answer key -- and it exists to bound the
    table from above.
    """
    realized = w.success.astype(float) - lam * w.cost_matrix(model)
    return np.argmax(realized, axis=1)


def cheapest_sufficient(w: Workload, model: CostModel | None = None) -> np.ndarray:
    """The brief's phrasing of the oracle: lowest-cost arm that actually succeeded.

    Falls back to the cheapest arm when no arm succeeds, since nothing else is
    available and paying more for a certain failure is strictly worse.
    """
    costs = w.cost_matrix(model)
    big = costs.max() + 1.0
    masked = np.where(w.success, costs, big)
    choice = np.argmin(masked, axis=1)
    none_ok = ~w.success.any(axis=1)
    choice[none_ok] = np.argmin(costs[none_ok], axis=1)
    return choice
