#!/usr/bin/env python3
"""Brand's rank-one incremental SVD, with forgetting, reorthogonalization, detection.

The numpy reference for `src/incsvd.c`, `src/reorth.c`, `src/forget.c` and
`src/detect.c`. Same algorithm, same constants, same order of operations -- deliberately,
because `oracle/compare.py` checks the compiled C against this file and a discrepancy
should mean a bug in the C, not a difference in specification.

The update
----------
Hold a thin factorization `A ~ U Sigma V^T` with `U` (m x r) orthonormal and `Sigma`
diagonal. A new sample `a` arrives. Split it into the part the model already explains
and the part it does not:

    mvec = U^T a          coordinates inside the subspace
    p    = a - U mvec     residual, orthogonal to the subspace
    rho  = || p ||        how much of `a` is new

Then, with `q = p / rho`,

    [ A  a ]  =  [ U  q ] [ Sigma  mvec ] [ V  0 ]^T
                          [   0     rho ] [ 0   1 ]

exactly. The middle matrix `K` is (r+1) x (r+1) and *small*: its SVD `K = U' S' V'^T`
costs O(r^3), and the updated factors are `U_new = [U q] U'`, `Sigma_new = S'`. Truncate
back to `r` columns and the state is again (m x r) + r. That is the whole algorithm:
O(mr + r^2) memory, O(mr + r^3) work per sample, and the stream is never re-read.

`V` is deliberately not tracked. It has one row per sample seen, so carrying it would
make memory grow with the length of the stream -- precisely what "constant memory"
forbids. Everything the detector needs lives in `U` and `Sigma`.

Truncation is the reason `U'` must be sorted by singular value before slicing. One-sided
Jacobi (which is what the C uses) does not sort, and an unsorted truncation silently
discards the *dominant* direction now and then: the tracker keeps running, the scores
stay finite, and the subspace is wrong. There is no exception and no crash.

The rho guard
-------------
When `a` lies in the current subspace to machine precision, `rho` is pure rounding
noise and `q = p / rho` is a random unit vector. Appending it injects a garbage
direction. The guard is `rho <= sqrt(eps) * ||a||`, the usual re-orthogonalization
criterion, and it is applied by setting `q = 0` and `rho = 0` rather than by branching
into separate code: with `rho = 0` the last row of `K` vanishes, so the top-r left
singular vectors of `K` have zero last entry and the update degenerates *exactly* into
the rank-preserving form `U <- U U'[:r, :r]`. One code path, no special case to get
wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oracle.rangefinder import randomized_svd, rank_by_energy, rank_by_gap

# Number of samples used to build the initial subspace, and then to calibrate the
# detector threshold. 300 >> r = 4 so the warm-up SVD is well determined; both windows
# together stay well inside the known-normal prefix of every generated stream.
WARMUP = 300
CALIBRATION = 300

# Threshold for || U^T U - I ||_F above which U is re-orthonormalized, and the same
# default the C uses (host/main.c). Scaling it with the working precision rather than
# fixing it at 1e-14 is what makes the identical default sensible in the float32 build,
# where the floor is nine orders of magnitude higher. 100 * eps is ~20x the floor a
# freshly orthonormalized 24 x 4 basis sits at, so the monitor does not chase rounding
# noise. Measured trajectories: analysis/drift_study.py.
REORTH_TOL = 100.0 * float(np.finfo(np.float64).eps)

# How often the monitor runs. It costs O(m r^2), the same order as the update itself,
# so doing it every sample would double the per-sample cost for a quantity that moves
# on a timescale of thousands of samples.
CHECK_EVERY = 20


def _sqrt_eps(dtype: np.dtype) -> float:
    return float(np.sqrt(np.finfo(dtype).eps))


@dataclass
class TrackerState:
    u: np.ndarray        # (m, r) orthonormal basis of the tracked subspace
    sigma: np.ndarray    # (r,) singular values, descending
    drift: float         # last measured || U^T U - I ||_F, before any repair
    n_reorth: int        # how many times the monitor fired
    max_drift: float = 0.0


def orthogonality_drift(u: np.ndarray) -> float:
    """|| U^T U - I ||_F. Zero iff the columns are exactly orthonormal."""
    r = u.shape[1]
    return float(np.linalg.norm(u.T @ u - np.eye(r, dtype=u.dtype), "fro"))


def reorthonormalize(u: np.ndarray, sigma: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Restore orthonormality of U *without* changing the subspace's factorization.

    Simply running Gram-Schmidt on U is wrong: `U Sigma V^T` is a factorization, and
    replacing `U` by `Q` from `U = Q R` changes what it factors unless `Sigma` moves
    too. The correct step keeps the product invariant:

        U Sigma = Q R Sigma = Q (R Sigma) = Q (U~ S~ V~^T)
        =>  U <- Q U~ ,  Sigma <- S~        (and V absorbs V~, which we do not track)

    `R` is within `drift` of the identity, so `S~` is within `drift` of `Sigma` -- the
    correction is tiny, which is exactly why skipping it looks harmless right up until
    the point where it is not.

    Cost is O(m r^2 + r^3), amortized over CHECK_EVERY samples.
    """
    q, r = np.linalg.qr(u)
    u_small, s_new, _ = np.linalg.svd(r * sigma[None, :], full_matrices=False)
    return q @ u_small, s_new


def initialize(block: np.ndarray, r_max: int = 8, oversampling: int = 6,
               power_iters: int = 1, energy: float = 0.95,
               rank_mode: str = "energy",
               seed: int = 0) -> tuple[TrackerState, np.ndarray, int, int]:
    """Build the initial subspace from a known-normal warm-up block.

    Returns (state, full_spectrum, r_energy, r_gap). Both rank criteria are always
    computed and returned even though only one is used, because the README has to
    show that they were both looked at.
    """
    rng = np.random.default_rng(seed)
    u_full, s_full = randomized_svd(block, r_max, oversampling, power_iters, rng)
    r_energy = rank_by_energy(s_full, energy)
    r_gap = rank_by_gap(s_full, r_max=r_max)
    if rank_mode == "energy":
        r = r_energy
    elif rank_mode == "gap":
        r = r_gap
    else:
        raise ValueError(f"rank_mode must be 'energy' or 'gap', got {rank_mode!r}")
    r = int(np.clip(r, 1, min(r_max, u_full.shape[1])))
    state = TrackerState(u=np.ascontiguousarray(u_full[:, :r]),
                         sigma=s_full[:r].copy(), drift=0.0, n_reorth=0)
    state.drift = orthogonality_drift(state.u)
    state.max_drift = state.drift
    return state, s_full, r_energy, r_gap


def score_only(state: TrackerState, a: np.ndarray) -> float:
    """Residual-energy score of `a` against the current subspace, without updating it."""
    resid = a - state.u @ (state.u.T @ a)
    a_norm = float(np.linalg.norm(a))
    return float(np.dot(resid, resid) / (a_norm * a_norm)) if a_norm > 0.0 else 0.0


def update(state: TrackerState, a: np.ndarray, lam: float = 1.0,
           reorth: bool = True, reorth_tol: float = REORTH_TOL,
           check_every: int = CHECK_EVERY, step: int = 0) -> float:
    """One rank-one update. Returns the residual-energy score of `a`.

    The score is computed against the subspace *before* `a` is folded in. Scoring
    afterwards would let every sample partially explain itself, which flatters isolated
    anomalies most -- exactly the samples the detector exists to catch.
    """
    u, sigma = state.u, state.sigma
    mvec = u.T @ a
    resid = a - u @ mvec
    rho = float(np.linalg.norm(resid))
    a_norm = float(np.linalg.norm(a))
    score = (rho * rho) / (a_norm * a_norm) if a_norm > 0.0 else 0.0

    # Exponential forgetting: scale Sigma, so the covariance contribution of a sample
    # k steps old is weighted lambda^{2k}. See src/forget.c for the window derivation.
    r = sigma.shape[0]
    if rho <= _sqrt_eps(a.dtype) * a_norm:
        rho, q = 0.0, np.zeros_like(a)
    else:
        q = resid / rho

    k = np.zeros((r + 1, r + 1), dtype=a.dtype)
    k[np.arange(r), np.arange(r)] = lam * sigma
    k[:r, r] = mvec
    k[r, r] = rho
    u_small, s_small, _ = np.linalg.svd(k)                 # numpy sorts descending
    u_big = np.concatenate([u, q[:, None]], axis=1)
    state.u = u_big @ u_small[:, :r]
    state.sigma = s_small[:r]

    # `drift` records the value AS MEASURED, before any repair; storing the post-repair
    # value would make a run with reorthogonalization report ~1e-16 forever and hide the
    # fact that the monitor fired at all. The saw-tooth that results -- rise to the
    # threshold, drop, rise again -- is the honest picture.
    if step % check_every == 0:
        state.drift = orthogonality_drift(state.u)
        state.max_drift = max(state.max_drift, state.drift)
        if reorth and state.drift > reorth_tol:
            state.u, state.sigma = reorthonormalize(state.u, state.sigma)
            state.n_reorth += 1
    return score


@dataclass
class StreamResult:
    scores: np.ndarray
    drift: np.ndarray
    threshold: float
    state: TrackerState
    spectrum: np.ndarray
    r: int
    r_energy: int
    r_gap: int


def run_stream(x: np.ndarray, lam: float = 1.0, reorth: bool = True,
               reorth_tol: float = REORTH_TOL, check_every: int = CHECK_EVERY,
               r_max: int = 8, oversampling: int = 6, power_iters: int = 1,
               energy: float = 0.95, rank_mode: str = "energy",
               quantile: float = 0.99, repeat: int = 1,
               seed: int = 0) -> StreamResult:
    """Run the tracker over every column of `x`, `repeat` times.

    Exactly one pass at `repeat = 1`: the first WARMUP samples are scored against the
    basis they built without updating it again, and the remaining n - WARMUP samples are
    each folded in once. `repeat` > 1 replays the file from the start, which is a device
    for the orthogonality-drift study and not how the tracker would be deployed --
    drift is a function of the NUMBER OF RANK-ONE UPDATES, not of how much distinct data
    there was, so 20 000 updates need a 2 000-sample file and `repeat = 10` rather than
    a 10x larger file. The drift trace covers every update and so has length
    (n - WARMUP) + (repeat - 1) * n; `scores` always has length n.

    The threshold is the `quantile` empirical quantile of the scores over the
    CALIBRATION samples that follow the warm-up. Those samples are known-normal by
    construction of the generator; on real data this is the "record the machine while
    it is healthy" step, and it is the only place a number is put in by hand -- and
    even there it is a false-positive rate, not a score.
    """
    m, n = x.shape
    if n <= WARMUP + CALIBRATION:
        raise ValueError(f"stream of {n} samples is shorter than the warm-up plus "
                         f"calibration windows ({WARMUP} + {CALIBRATION})")
    state, spectrum, r_energy, r_gap = initialize(
        x[:, :WARMUP], r_max=r_max, oversampling=oversampling,
        power_iters=power_iters, energy=energy, rank_mode=rank_mode, seed=seed)

    scores = np.zeros(n)
    drift_trace: list[float] = []

    # The warm-up samples built the basis; scoring them again is free, updating with
    # them again would count them twice. See incsvd_score in src/incsvd.h.
    for t in range(WARMUP):
        scores[t] = score_only(state, x[:, t])

    step = 0
    for t in range(WARMUP, n):
        scores[t] = update(state, x[:, t], lam=lam, reorth=reorth,
                           reorth_tol=reorth_tol, check_every=check_every, step=step)
        drift_trace.append(state.drift)
        step += 1
    for _ in range(1, repeat):
        for t in range(n):
            update(state, x[:, t], lam=lam, reorth=reorth, reorth_tol=reorth_tol,
                   check_every=check_every, step=step)
            drift_trace.append(state.drift)
            step += 1
    drift = np.asarray(drift_trace)

    warm = scores[WARMUP:WARMUP + CALIBRATION]
    threshold = float(np.quantile(warm, quantile))
    return StreamResult(scores=scores, drift=drift, threshold=threshold, state=state,
                        spectrum=spectrum, r=state.sigma.shape[0],
                        r_energy=r_energy, r_gap=r_gap)


# --- subspace comparison ----------------------------------------------------------
# U is only defined up to an r x r rotation and a sign per column, so comparing two
# bases elementwise tests nothing. These two quantities are invariant under that
# freedom, which is why every test in tests/ that compares subspaces uses one of them.


def principal_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between range(U) and range(V).

    cos(theta_i) are the singular values of U^T V for orthonormal U, V. All zero iff
    the subspaces coincide.

    Accurate only down to about sqrt(eps) radians. theta = arccos(1 - delta) ~
    sqrt(2 delta), so a cosine computed to 1e-16 yields an angle no better than 1e-8 rad
    -- reporting "the subspaces agree to 1e-15 degrees" would be reporting noise.
    `projection_distance` has no such square root and is the primary metric wherever a
    tight tolerance is asserted; the angles are reported because they are the
    interpretable quantity.
    """
    s = np.linalg.svd(u.T @ v, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def projection_distance(u: np.ndarray, v: np.ndarray) -> float:
    """|| U U^T - V V^T ||_F, the basis-independent distance between two subspaces.

    Equal to sqrt(2) * || sin(theta) ||_2 for equal-dimension subspaces, and bounded
    by sqrt(2 r). Zero iff the subspaces are identical.
    """
    return float(np.linalg.norm(u @ u.T - v @ v.T, "fro"))
