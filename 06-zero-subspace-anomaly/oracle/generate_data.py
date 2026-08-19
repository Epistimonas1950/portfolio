#!/usr/bin/env python3
"""Synthetic multi-channel sensor streams with labelled anomalies.

There is no Raspberry Pi Zero and no I2S microphone / ADXL345 / INA219 attached to
the machine this repo was built on, so the acquisition path in BRIEF.md could not be
run. Everything below is generated, seeded and reproducible, and the README says so
in plain words rather than dressing it up as a capture.

What the generator has to get right is the *structure* the detector exploits, and
nothing else:

    a_t = U_mode c_t + sigma * e_t ,      c_t ~ N(0, diag(s)^2),  e_t ~ N(0, I)

with `U_mode` an orthonormal m x r basis. Normal operation therefore lives within
`sigma * sqrt(m - r)` of an r-dimensional subspace, and the residual energy
`|| a - U U^T a ||^2` of a normal sample is a scaled chi-square with `m - r` degrees
of freedom. That is the null distribution the detector's threshold is calibrated
against, and knowing it exactly is the reason for using synthetic data here: on a
real motor I would not know the true rank, so I could not tell a detector failure
from a data surprise.

Four streams are written, each isolating one claim:

  normal.csv      single mode, no anomalies -- warm-up and threshold calibration
  anomalous.csv   single mode + labelled out-of-subspace spikes + a rotation segment
  multimode.csv   normal alternates between TWO subspaces, same spikes at the same
                  indices and the same amplitudes. The only variable changed is the
                  structure of the *normal* class, which is what makes the AUC drop
                  attributable to the limitation in BRIEF.md rather than to the
                  anomalies being harder.
  manymode.csv    FOUR subspaces, switching faster. The two-mode case turns out to be
                  fixable by picking the rank properly; this one is not, and shipping
                  only the fixable version would have misrepresented the limitation.
  rotating.csv    subspace rotates steadily, no anomalies -- the forgetting study

Rotations are generated with the Cayley transform, R = (I - W/2)^{-1} (I + W/2) for
skew-symmetric W, which is orthogonal to machine precision for any step size. Building
the rotation by re-orthonormalizing `I + dt*W` instead would inject exactly the kind of
drift this project is trying to measure elsewhere.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

# --- stream geometry -------------------------------------------------------------
# 24 channels is the size of a small filterbank or a modest sensor array. It is large
# enough that a rank-4 subspace is a real restriction (a random direction keeps
# 1 - 4/24 = 83% of its energy outside it) and small enough that the whole tracker
# state, 24 x 8 doubles, is 1.5 kB -- the "constant memory" claim is only interesting
# if the constant is small.
N_CHANNELS = 24
RANK_TRUE = 4

# Component amplitudes. Chosen so that a 95% energy threshold selects exactly r = 4:
# the cumulative energy fractions are 0.48, 0.75, 0.92, 1.00. A steeper spectrum would
# let the energy criterion succeed for the wrong reason.
COMPONENT_SCALES = np.array([6.0, 4.5, 3.5, 2.5])

# Per-channel noise. sigma = 0.05 puts the noise floor ~2 orders below the weakest
# signal component, so the rank is unambiguous and the singular-value gap criterion
# has something to find.
NOISE_SIGMA = 0.05

# Anomaly amplitude, calibrated rather than guessed. 1.1 is ~4.5x the ambient noise
# vector norm (0.05 * sqrt(24) = 0.24) but only 1.3% of the signal energy, and it was
# chosen by sweeping the amplitude and reading off the AUC: it puts the single-mode
# detector at 0.985, high but not saturated. A larger spike scores AUC = 1.000 in every
# scenario including the multimode one, and the limitation this repo volunteers would
# then be invisible -- which is the usual reason such a limitation goes unreported.
SPIKE_AMPLITUDE = 1.1


@dataclass
class Stream:
    x: np.ndarray                                  # (m, n), columns are samples
    label: np.ndarray                              # (n,) 0 = normal, 1 = anomaly
    kind: list[str] = field(default_factory=list)  # (n,) descriptive tag
    basis_final: np.ndarray | None = None          # true subspace at the last sample


def _orthonormal_basis(m: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random r-dimensional subspace of R^m, as an m x r orthonormal basis."""
    q, _ = np.linalg.qr(rng.normal(size=(m, m)))
    return np.ascontiguousarray(q[:, :r])


def _cayley(skew: np.ndarray) -> np.ndarray:
    """Exactly orthogonal rotation from a skew-symmetric generator.

    (I - W/2)^{-1}(I + W/2) is orthogonal for every skew-symmetric W, exactly, not to
    first order -- so repeated application never needs re-orthonormalizing. That
    matters here: the ground-truth basis must not drift, or the drift measured in
    src/reorth.c would be confounded with drift baked into the data.
    """
    n = skew.shape[0]
    eye = np.eye(n)
    return np.linalg.solve(eye - 0.5 * skew, eye + 0.5 * skew)


def _skew(m: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(m, m))
    w = a - a.T
    return w / np.linalg.norm(w, 2)


def _coefficients(r: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(r, n)) * COMPONENT_SCALES[:r, None]


def make_normal(n: int = 1500, seed: int = 0) -> Stream:
    """Single stationary mode, no anomalies."""
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(N_CHANNELS, RANK_TRUE, rng)
    x = basis @ _coefficients(RANK_TRUE, n, rng)
    x += NOISE_SIGMA * np.random.default_rng(seed + 500).normal(size=x.shape)
    return Stream(x=x, label=np.zeros(n, dtype=int), kind=["normal"] * n,
                  basis_final=basis)


def _spike_rng(seed: int) -> np.random.Generator:
    """A generator used for nothing but the spikes.

    The single-mode and multimode scenarios must inject *identical* anomalies -- same
    indices, same directions, same amplitude -- or the AUC gap between them confounds
    "the normal class got harder" with "the anomalies got harder". Drawing the spikes
    from a stream-independent generator is what guarantees that; sharing one generator
    would not, because the two scenarios consume different numbers of draws while
    building their normal parts.
    """
    return np.random.default_rng(seed + 10_000)


def _spikes(n_spikes: int, lo: int, hi: int, seed: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly indices and unit directions, none of them inside the warm-up region.

    The detector calibrates its threshold on the first part of the stream and is told
    that region is normal. Placing an anomaly there would be calibrating on
    contaminated data -- a real failure mode, but a different experiment.
    """
    rng = _spike_rng(seed)
    idx = np.sort(rng.choice(np.arange(lo, hi), size=n_spikes, replace=False))
    dirs = rng.normal(size=(n_spikes, N_CHANNELS))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return idx, dirs


def make_anomalous(n: int = 2000, seed: int = 1, n_spikes: int = 120,
                   rotate_from: float = 0.75) -> Stream:
    """Single mode + isolated out-of-subspace spikes + a subspace-rotation segment.

    Two anomaly kinds, deliberately different in character:

    `spike`   one sample gains SPIKE_AMPLITUDE of energy along a fresh random
              direction. In expectation 1 - r/m = 83% of that lands outside the
              subspace. This is the anomaly subspace methods are built for.
    `rotate`  from `rotate_from` onward the subspace itself turns. No single sample is
              odd; the *model* is stale. A tracker with forgetting will follow it and
              stop calling it an anomaly, which is the correct behaviour for a slow
              regime change and the wrong behaviour for a fault -- the distinction is
              a modelling choice, not something the mathematics decides. Labelled
              separately so the AUC for the two kinds can be reported separately.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 500)
    basis = _orthonormal_basis(N_CHANNELS, RANK_TRUE, rng)
    coef = _coefficients(RANK_TRUE, n, rng)

    start = int(rotate_from * n)
    # 50 degrees of total rotation over the tail. Slow enough that no single sample
    # looks wrong, fast enough that a lambda = 1 tracker is left behind.
    total_angle = np.deg2rad(50.0)
    step = _cayley((total_angle / (n - start)) * _skew(N_CHANNELS, rng))

    x = np.empty((N_CHANNELS, n))
    kind = ["normal"] * n
    label = np.zeros(n, dtype=int)
    current = basis.copy()
    for t in range(n):
        if t >= start:
            current = step @ current
            kind[t] = "rotate"
            label[t] = 1
        x[:, t] = current @ coef[:, t]
    x += NOISE_SIGMA * noise_rng.normal(size=x.shape)

    # Spikes live only in the non-rotating region, so the two anomaly kinds never
    # overlap and their AUCs can be reported separately.
    idx, dirs = _spikes(n_spikes, int(0.35 * n), start, seed)
    for t, direction in zip(idx, dirs):
        x[:, t] += SPIKE_AMPLITUDE * direction
        kind[t] = "spike"
        label[t] = 1

    return Stream(x=x, label=label, kind=kind, basis_final=current)


def make_multimode(n: int = 2000, seed: int = 1, n_spikes: int = 120,
                   dwell: int = 150) -> Stream:
    """Normal operation alternates between two subspaces. The failure case.

    Seed, length, spike count, spike positions and spike amplitudes are identical to
    `make_anomalous` by construction (same generator, same call order), so the AUC
    difference between the two scenarios is attributable to one change: the normal
    class is now a union of two subspaces instead of one.

    The two modes share two of their four directions. Completely independent modes
    would be an easier problem in disguise -- the union would have rank 8 and a rank-8
    tracker would cover both. Overlapping modes are the realistic case: a single
    subspace of any rank is a compromise between them.

    `dwell` = 150 samples per mode is deliberately shorter than the forgetting window
    (N_eff ~ 300, see src/forget.c). The tracker is therefore always chasing the mode
    it has just left, and every switch produces a burst of residual energy that is
    labelled normal -- because it is.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 500)
    basis = _orthonormal_basis(N_CHANNELS, RANK_TRUE, rng)
    coef = _coefficients(RANK_TRUE, n, rng)

    other = _orthonormal_basis(N_CHANNELS, RANK_TRUE, rng)
    mode_b = np.linalg.qr(np.hstack([basis[:, :2], other[:, :2]]))[0][:, :RANK_TRUE]

    x = np.empty((N_CHANNELS, n))
    kind = ["normal"] * n
    label = np.zeros(n, dtype=int)
    for t in range(n):
        current = basis if (t // dwell) % 2 == 0 else mode_b
        x[:, t] = current @ coef[:, t]
    x += NOISE_SIGMA * noise_rng.normal(size=x.shape)

    idx, dirs = _spikes(n_spikes, int(0.35 * n), int(0.75 * n), seed)
    for t, direction in zip(idx, dirs):
        x[:, t] += SPIKE_AMPLITUDE * direction
        kind[t] = "spike"
        label[t] = 1

    return Stream(x=x, label=label, kind=kind, basis_final=mode_b)


def make_manymode(n: int = 2000, seed: int = 1, n_spikes: int = 120,
                  n_modes: int = 4, dwell: int = 100) -> Stream:
    """Four operating modes instead of two. The case rank selection cannot rescue.

    `make_multimode` turns out to be fixable: its two modes span a rank-6 union, the gap
    criterion finds rank 6, and the AUC comes back to 0.986 (results/auc.csv). Shipping
    only that scenario would have let the limitation in BRIEF.md look like a rank-choice
    bug. So here is the harder one: four rank-4 modes sharing a single common direction,
    spanning a rank-13 union, switching every `dwell` = 100 samples -- shorter than the
    forgetting window the detector is tuned with.

    Every anomaly is again identical to the other scenarios' by construction.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 500)
    full = _orthonormal_basis(N_CHANNELS, N_CHANNELS, rng)
    shared = full[:, :1]
    modes = []
    for i in range(n_modes):
        block = full[:, 1 + 3 * i:1 + 3 * i + 3]
        modes.append(np.linalg.qr(np.hstack([shared, block]))[0][:, :RANK_TRUE])
    coef = _coefficients(RANK_TRUE, n, rng)

    x = np.empty((N_CHANNELS, n))
    kind = ["normal"] * n
    label = np.zeros(n, dtype=int)
    for t in range(n):
        x[:, t] = modes[(t // dwell) % n_modes] @ coef[:, t]
    x += NOISE_SIGMA * noise_rng.normal(size=x.shape)

    idx, dirs = _spikes(n_spikes, int(0.35 * n), int(0.75 * n), seed)
    for t, direction in zip(idx, dirs):
        x[:, t] += SPIKE_AMPLITUDE * direction
        kind[t] = "spike"
        label[t] = 1
    return Stream(x=x, label=label, kind=kind, basis_final=modes[-1])


def make_rotating(n: int = 2000, seed: int = 3,
                  total_degrees: float = 90.0) -> Stream:
    """A steadily rotating subspace, no anomalies. The forgetting benchmark.

    `basis_final` is the exact subspace at the last sample, so a tracker can be scored
    by the principal angles between what it converged to and what was actually there.
    Ninety degrees over the stream means the final subspace is orthogonal to the
    initial one: a lambda = 1 tracker, which weights the first sample as heavily as the
    last, cannot be right at both ends.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 500)
    basis = _orthonormal_basis(N_CHANNELS, RANK_TRUE, rng)
    coef = _coefficients(RANK_TRUE, n, rng)
    step = _cayley((np.deg2rad(total_degrees) / n) * _skew(N_CHANNELS, rng))

    x = np.empty((N_CHANNELS, n))
    current = basis.copy()
    for t in range(n):
        current = step @ current
        x[:, t] = current @ coef[:, t]
    x += NOISE_SIGMA * noise_rng.normal(size=x.shape)
    return Stream(x=x, label=np.zeros(n, dtype=int), kind=["normal"] * n,
                  basis_final=current)


def write_stream(path: pathlib.Path, x: np.ndarray) -> None:
    """CSV, one row per sample, one column per channel, with a header row.

    Row-major by sample is the layout a streaming reader wants; the algorithms below
    treat a sample as a column vector, so this file is the transpose of the data
    matrix A. Stated here because getting it backwards is the classic way to spend an
    afternoon debugging a subspace tracker.
    """
    header = ",".join(f"ch{i:02d}" for i in range(x.shape[0]))
    np.savetxt(path, x.T, delimiter=",", header=header, comments="", fmt="%.6g")


def write_labels(path: pathlib.Path, stream: Stream) -> None:
    with path.open("w") as fh:
        fh.write("index,label,kind\n")
        for i, (lab, knd) in enumerate(zip(stream.label, stream.kind)):
            fh.write(f"{i},{int(lab)},{knd}\n")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    jobs = [
        ("normal.csv", None, make_normal()),
        ("anomalous.csv", "labels.csv", make_anomalous()),
        ("multimode.csv", "multimode_labels.csv", make_multimode()),
        ("manymode.csv", "manymode_labels.csv", make_manymode()),
        ("rotating.csv", None, make_rotating()),
    ]
    for name, label_name, stream in jobs:
        write_stream(DATA / name, stream.x)
        if label_name:
            write_labels(DATA / label_name, stream)
        n_anom = int(stream.label.sum())
        print(f"{name:20s} {stream.x.shape[1]:5d} samples x {stream.x.shape[0]} channels"
              f"  anomalies={n_anom}")
    print(f"\nwrote {DATA}")


if __name__ == "__main__":
    main()
