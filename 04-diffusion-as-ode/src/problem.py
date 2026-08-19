r"""The test problem, and why its parameters are what they are.

Everything in this repo is measured on a Gaussian-mixture prior under the VP SDE.
That choice is what makes the score exact (src/sde.py) and the solution exact
(src/reference.py). The remaining question is *which* mixture, and it is not
cosmetic: the conditioning of the probability-flow map is set entirely by how
separated the modes are, and a badly conditioned problem destroys a convergence-order
measurement while looking, superficially, like a more impressive test case.

Conditioning of the flow map
----------------------------
The exact map is Phi = F_0^{-1} o F_T. Differentiating F_0(Phi(x)) = F_T(x),

    Phi'(x) = f_T(x) / f_0(Phi(x))

-- the ratio of the source and target densities at corresponding points. Between two
well-separated modes the target density is astronomically small while the source
density (nearly N(0,1) at t = T) is O(1), so Phi' there is astronomically large. A
trajectory threading that gap is exponentially ill-conditioned: a perturbation of size
eps at t = T is amplified by Phi' by the time it reaches t_eps.

That is a property of the *problem*, not of any integrator, and it is the real reason
diffusion samplers need many steps on sharply multimodal data. But it also means the
error constant C in ||e|| <= C h^p is proportional to max |Phi'|, so the asymptotic
regime -- the range of h where the fitted log-log slope is the true order -- begins at
h ~ 1/C. Measure the order on a prior with max|Phi'| ~ 10^4 over a decade of h and you
will read off a slope near 0.75 for a method that is provably first order.

So there are two priors here and both are used:

  CANONICAL  three clearly separated modes (4 sigma apart) with max|Phi'| ~ 4.6. All
             the order fits are done on this one. It is genuinely multimodal -- its
             density has three local maxima and the score is strongly nonlinear -- and
             it is well enough conditioned that N = 32 is already asymptotic.

  SHARP      modes 8 sigma apart, max|Phi'| ~ 1.7e4. Used once, in
             analysis/convergence_order.py, to show what a badly conditioned problem
             does to the same measurement: the fitted slopes drop well below their
             true values over the same range of h, and the per-interval orders jump
             around as individual trajectories cross from one mode's basin to another.
             Reporting only that number would be a mistake; hiding it would be a
             bigger one.

Two degenerate priors exist for the exactness checks:

  GAUSSIAN   a single Gaussian: the probability-flow ODE is then linear in x and the
             solution is affine, so `analytic_gaussian_flow` applies in any dimension.
  POINT_MASS a single Gaussian of zero variance: additionally, the noise prediction
             eps is constant along every trajectory, which is exactly the condition
             under which the exponential integrator is exact in one step.
"""

from __future__ import annotations

import numpy as np

from .reference import probability_levels, quantile_states
from .sde import GaussianMixture, VPSDE

#: The VP schedule of Song et al. (2011.13456) with the standard beta range. t_min is
#: 1e-3 rather than 0 because sigma(0) = 0 makes the score and the log-SNR infinite;
#: every diffusion implementation truncates here for the same reason.
SDE = VPSDE(beta_min=0.1, beta_max=20.0, t_min=1e-3, t_max=1.0)

#: Three modes about 4 component-sigma apart: unambiguously multimodal, still
#: well-conditioned (max |Phi'| = 4.6, measured by flow_map_condition below).
CANONICAL = GaussianMixture(
    weights=np.array([0.35, 0.40, 0.25]),
    means=np.array([[-1.5], [0.2], [1.8]]),
    variances=np.array([0.45, 0.40, 0.35]) ** 2,
)

#: The same shape, sharpened: modes 8 sigma apart. max |Phi'| ~ 1.7e4.
SHARP = GaussianMixture(
    weights=np.array([0.30, 0.45, 0.25]),
    means=np.array([[-2.0], [0.4], [2.6]]),
    variances=np.array([0.15, 0.25, 0.10]) ** 2,
)

#: Single Gaussian -- the linear case.
GAUSSIAN = GaussianMixture(np.array([1.0]), np.array([[0.6]]), np.array([0.5**2]))

#: Point mass -- the case where the exponential integrator is exact in one step.
POINT_MASS = GaussianMixture(np.array([1.0]), np.array([[0.7]]), np.array([0.0]))

PRIORS = {"canonical": CANONICAL, "sharp": SHARP, "gaussian": GAUSSIAN,
          "point_mass": POINT_MASS}


def flow_map_condition(sde, prior: GaussianMixture, n: int = 2001,
                       t_end: float | None = None) -> float:
    """max |Phi'| = max f_T(x) / f_0(Phi(x)) over the probability levels actually used.

    The error constant of every fixed-step method here is proportional to this, so it
    is reported alongside the fitted slopes rather than left implicit.
    """
    t_end = sde.t_min if t_end is None else t_end
    probs = probability_levels(n)
    x_hi = quantile_states(sde, prior, probs, sde.t_max).ravel()
    x_lo = quantile_states(sde, prior, probs, t_end).ravel()
    f_hi = sde.marginal(prior, sde.t_max).pdf(x_hi)
    f_lo = sde.marginal(prior, t_end).pdf(x_lo)
    return float(np.max(f_hi / f_lo))
