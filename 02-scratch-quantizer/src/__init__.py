"""Post-training quantization built from the mathematics up.

The objective everywhere in this package is the *activation-weighted* reconstruction
error of a single linear layer,

    minimize   ||W X - W_hat X||_F^2     over W_hat on a low-bit grid,

not the weight-space error ||W - W_hat||_F that round-to-nearest minimizes.
"""
