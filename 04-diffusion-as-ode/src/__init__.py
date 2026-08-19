"""Diffusion sampling treated as what it is: numerical integration of an ODE/SDE.

The reverse-time process of a score-based generative model is

    dx = [ f(x,t) - g(t)^2 grad_x log p_t(x) ] dt + g(t) dw_bar        (SDE)
    dx/dt = f(x,t) - 0.5 g(t)^2 grad_x log p_t(x)                      (probability flow ODE)

and a "sampler" is an integrator for it. Everywhere in this package the score
grad_x log p_t(x) is *exact* -- it is the analytic score of a Gaussian-mixture prior
carried through the forward SDE -- so every error reported is the integrator's, and
nothing else's.
"""
