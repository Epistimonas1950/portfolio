r"""Cross-layer error propagation: the bound, and the measurement.

Layer l's output error is layer l+1's *input* perturbation, so per-layer error
figures do not compose by addition and a quantizer that looks fine layer-by-layer can
still wreck a deep stack. The bound below is loose on purpose -- it is the one you can
actually prove -- and the project's headline figure is predicted versus measured.

Setup. Let the exact stack be  a_l = W_l a_{l-1}  and the quantized stack
a_hat_l = W_hat_l a_hat_{l-1}, with a_0 = a_hat_0 = X. Define the propagated error
E_l = a_l - a_hat_l. Then

    E_l = W_l a_{l-1} - W_hat_l a_hat_{l-1}
        = (W_l - W_hat_l) a_{l-1}  +  W_hat_l (a_{l-1} - a_hat_{l-1})

and taking Frobenius norms with the submultiplicative inequality,

    ||E_l||_F  <=  ||(W_l - W_hat_l) a_{l-1}||_F  +  ||W_hat_l||_2 ||E_{l-1}||_F
                   \_________ local, measurable _________/   \___ amplification ___/

The first term is exactly the per-layer objective the quantizer minimizes, measured
on the *clean* activations -- so it is available for free at quantization time. The
recursion unrolls to

    ||E_L||_F  <=  sum_{l=1..L}  ( prod_{k=l+1..L} ||W_hat_k||_2 ) * local_l

Where it is loose, and why that is the interesting part: the triangle inequality
assumes the local error and the incoming propagated error point the same way, and the
operator norm assumes the incoming error lands on the top singular direction of the
next layer. Neither holds in practice -- errors from independent rounding decisions
are close to orthogonal, and they land isotropically. So the bound should overshoot,
and the *size of the gap* measures how uncorrelated the layers' errors are. Plotting
predicted against measured is the point of the exercise; a bound that were tight would
mean the rounding errors were conspiring, which would itself be a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PropagationRow:
    layer: int
    local_error: float
    predicted: float
    measured: float

    @property
    def ratio(self) -> float:
        return self.predicted / self.measured if self.measured else float("inf")


def propagate(weights: list[np.ndarray], weights_hat: list[np.ndarray],
              x0: np.ndarray) -> list[PropagationRow]:
    """Run both stacks side by side; return per-layer bound and measurement."""
    if len(weights) != len(weights_hat):
        raise ValueError("exact and quantized stacks must have the same depth")

    a = np.asarray(x0, dtype=np.float64)
    a_hat = a.copy()
    predicted = 0.0
    rows: list[PropagationRow] = []

    for i, (w, w_hat) in enumerate(zip(weights, weights_hat), start=1):
        local = float(np.linalg.norm((w - w_hat) @ a))     # on the clean input
        amplification = float(np.linalg.norm(w_hat, 2))
        predicted = local + amplification * predicted      # the recursion, unrolled

        a = w @ a
        a_hat = w_hat @ a_hat
        measured = float(np.linalg.norm(a - a_hat))

        rows.append(PropagationRow(layer=i, local_error=local,
                                   predicted=predicted, measured=measured))
    return rows
