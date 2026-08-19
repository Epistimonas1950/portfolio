"""The routing context vector x_t, and a hard line about what a router may look at.

A router runs *before* the call it is routing. So the only admissible features are
things measurable from the incoming request. That rules out the answer, the latency,
the token count of the completion, and above all the latent difficulty -- which is the
one feature that would make routing trivial, and which is exactly why the simulator
keeps it hidden and offers only a noisy classifier score in its place.

    x  =  [ 1,
            log_prompt_tokens,                     length of the request
            task_qa, task_tool, task_code,         one-hot task type (3, sums to 1)
            tool_depth,                            how deep the tool chain already is
            s,                                     difficulty-classifier score in [0,1]
            s^2,                                   curvature: p_a(d) is a sigmoid in d
            s * log_prompt_tokens,                 long *and* hard is worse than either
            s * tool_depth ]                       deep *and* hard is worse than either

Ten dimensions. The three interaction terms are there because the reward surface is
genuinely not additive: a hard query costs a small model little when it is short, and a
lot when it is long and nested. The `s^2` term is the cheapest correction for the fact
that the true response curve p_a(d) = sigma(beta (s_a - d) + b0) is a sigmoid and the
router's model is linear -- a quadratic in s is the best a linear-in-features model can
do against it, and the residual is the model-misspecification floor that shows up as a
non-vanishing regret rate in results/regret.csv. That floor is a finding, not a bug:
the Õ(d sqrt(T)) bound is a statement about a well-specified linear model, and the fleet
is not one.

`SERVING_AVAILABLE` records the audit. Every entry is True; the module exists partly to
make that claim checkable rather than assumed.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_prompt_tokens",
    "task_qa",
    "task_tool",
    "task_code",
    "tool_depth",
    "difficulty_score",
    "difficulty_score_sq",
    "score_x_length",
    "score_x_depth",
)

#: Whether each feature can be computed before the model call it is routing. The
#: latent difficulty and the true label are deliberately absent from FEATURE_NAMES.
SERVING_AVAILABLE: dict[str, bool] = {name: True for name in FEATURE_NAMES}

N_FEATURES = len(FEATURE_NAMES)


def featurize(workload) -> np.ndarray:
    """(T, 10) context matrix for a Workload. Scaled so every column is order one.

    Scaling matters for a ridge-regularised bandit: LinUCB's confidence width is
    sqrt(x^T A^{-1} x) with A = lambda I + sum x x^T, so a column measured in thousands
    would be regularised a thousand times less than a column measured in ones, and the
    single ridge constant would mean different things in different directions.
    """
    n = len(workload)
    log_len = np.log(workload.prompt_tokens) / np.log(1000.0)     # ~0.5-1.0
    depth = workload.tool_depth / 4.0                             # ~0-1
    s = workload.difficulty_score

    onehot = np.zeros((n, 3))
    onehot[np.arange(n), workload.task_type] = 1.0

    x = np.empty((n, N_FEATURES))
    x[:, 0] = 1.0
    x[:, 1] = log_len
    x[:, 2:5] = onehot
    x[:, 5] = depth
    x[:, 6] = s
    x[:, 7] = s * s
    x[:, 8] = s * log_len
    x[:, 9] = s * depth
    return x


def describe() -> str:
    """One line per feature, for the README and for anyone auditing the audit."""
    lines = [f"{i:2d}  {name:22s} serving-available={SERVING_AVAILABLE[name]}"
             for i, name in enumerate(FEATURE_NAMES)]
    return "\n".join(lines)
