"""Activation-aware low-rank compression, built from the mathematics up.

Every module in this package minimizes the *activation-weighted* reconstruction error
of a single linear layer,

    minimize   ||(W - W_hat) X||_F      over rank-r W_hat,

not the unweighted error ||W - W_hat||_F that plain truncated SVD minimizes. The two
differ by exactly one change of variable -- the Cholesky factor of the activation
second moment -- and that change of variable is the whole project.
"""
