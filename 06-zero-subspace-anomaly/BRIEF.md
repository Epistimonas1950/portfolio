# 06 · Zero: Streaming Subspace Tracking for Anomaly Detection

> SVD in bounded memory on a board too weak for a neural network — which is the point.

| | |
|---|---|
| **Effort** | 1–2 weeks |
| **Prerequisites** | None (shares SVD intuition with `01`) |
| **Hardware** | Raspberry Pi Zero / Zero W (armv6l) or Zero 2 W (Cortex-A53) |
| **Math** | Randomized range finding, incremental SVD, reorthogonalization, rank selection |
| **Status** | ☐ not started |

---

## Know the board before you plan the project

| | Zero / Zero W | Zero 2 W |
|---|---|---|
| Core | ARM11, **armv6l**, 1 GHz | 4× Cortex-A53, 1 GHz |
| RAM | 512 MB | 512 MB |
| Modern ML wheels | almost none prebuilt | yes (TFLite, ONNX Runtime) |
| LLM | no | impractical (seconds/token) |

**The original Zero is armv6.** Most modern Python ML wheels are not built for it, and
`llama.cpp` on a Pi Zero is a documented build fight. That constraint is normally a nuisance —
here it is the entire justification for the project. You are going to solve a real detection
problem with numerical linear algebra you compile yourself, on a board where the framework
approach simply does not run.

**Check which board you own before starting.** On a Zero 2 W you can additionally run a small
TFLite autoencoder as a comparison baseline, which strengthens the story. On an original Zero,
the absence of that option *is* the story.

## The problem

You have multi-channel sensor data — vibration from a motor, audio, power draw, or several
environmental channels. Normal operation lives near a low-dimensional subspace. Anomalies are
whatever has large energy *orthogonal* to it.

The batch solution is a full SVD of the whole data matrix, which needs all the data in memory and
`O(mn²)` work. On 512 MB with a stream that never ends, that is not available. So you track the
subspace **incrementally, in constant memory**.

## The mathematics

**1. Initial subspace via randomized range finding.** Draw a Gaussian sketch `Ω ∈ ℝ^{n×(r+p)}`
with oversampling `p ≈ 5–10`, form `Y = AΩ`, orthonormalize `Q = qr(Y)`, and project. With `q`
power iterations, `Y = (AAᵀ)^q AΩ`, the singular-value decay is sharpened and the approximation
error approaches the optimal rank-`r` error with high probability. **State the error bound and
what `p` and `q` buy you** — this is a probabilistic guarantee, and being able to state it
precisely separates you from someone who called a library.

**2. Incremental update (Brand's algorithm).** Given `A ≈ UΣVᵀ` and a new column `a`, split it
into the part inside the current subspace and the residual:

```
m = Uᵀa ,   p = a − Um ,   ρ = ‖p‖
```

Then the updated SVD follows from a small `(r+1)×(r+1)` decomposition of

```
[ Σ   m ]
[ 0   ρ ]
```

which costs `O(r³)` per sample with `r` small — constant work, constant memory, no re-reading
history. This is the algorithmic core.

**3. Reorthogonalization.** Repeated rank-one updates lose orthogonality of `U` in floating
point; the error accumulates and the subspace silently rotates. Monitor `‖UᵀU − I‖` and
re-orthonormalize when it exceeds a threshold. **Plot that quantity over time** with and without
periodic reorthogonalization — a clean, cheap figure that shows numerical awareness.

**4. Forgetting.** For a sliding window, apply an exponential forgetting factor `λ` to `Σ` so old
data decays. Choose `λ` from the desired effective window length and say how.

**5. Rank selection.** Do not hardcode `r`. Choose it from the singular-value spectrum — an energy
threshold (`Σ_{i≤r} σᵢ² / Σ σᵢ² ≥ 0.95`), or a gap criterion. Show the spectrum; justify the
choice.

**6. The detector.** Score each sample by residual energy `‖a − UUᵀa‖²`, normalized. Threshold
from the empirical distribution during a known-normal warm-up period, not a magic constant.

## What to build

- [ ] Data acquisition (I²S microphone, ADXL345 accelerometer, or an INA219 current sensor)
- [ ] Randomized range finder with configurable oversampling and power iterations
- [ ] Brand rank-one incremental SVD update
- [ ] Orthogonality monitor + periodic reorthogonalization
- [ ] Exponential forgetting
- [ ] Residual-energy detector with warm-up-calibrated threshold
- [ ] Batch full-SVD oracle **on your laptop** as the accuracy ceiling
- [ ] Labelled test set: record normal operation, then induce anomalies deliberately
- [ ] (Zero 2 W only) small TFLite autoencoder baseline for comparison

## How it's measured

| Method | AUC ↑ | Memory | Latency/sample | Runs on armv6? |
|---|---|---|---|---|
| Batch full SVD (laptop oracle) | | O(mn) | — | — |
| Yours: incremental, r = ? | | O(mr) | | yes |
| Yours: no reorthogonalization | | | | yes |
| TFLite autoencoder (Zero 2 W) | | | | no |

Plus: ROC curves on one axis, the orthogonality-drift plot, and the singular-value spectrum that
justifies your rank.

## Interview claim

> Not everything needs a neural network. I solved this with numerical linear algebra in bounded
> memory on a $15 board, and here is the ROC curve showing what it cost me versus the exact
> solution.

## The limitation you volunteer first

Subspace methods assume normal behaviour is *linearly* low-dimensional. If the normal regime has
several distinct operating modes, a single subspace blurs them and the detector degrades. Say so,
and show the failure case — then note the fix (a mixture of subspaces, or per-mode models) without
necessarily building it.

## Stack

Plain C (or NumPy on Zero 2 W) · no ML framework required · Python/NumPy on the laptop for the
oracle and the plots

## Suggested repo layout

```
zero-subspace-anomaly/
  README.md              <- ROC + orthogonality-drift plots
  src/
    rangefinder.c/h      randomized range finding
    incsvd.c/h           Brand rank-one update
    reorth.c             orthogonality monitor
    detect.c             residual energy + threshold
    sensor.c
  oracle/
    batch_svd.py         laptop ground truth
    roc.py
  data/
    normal.csv
    anomalous.csv
  results/
    roc.png
    orthogonality_drift.png
    spectrum.png
```

## References

- Halko, Martinsson & Tropp (2011). [*Finding Structure with Randomness: Probabilistic Algorithms
  for Constructing Approximate Matrix Decompositions*](https://arxiv.org/abs/0909.4061)
  (arXiv:0909.4061; SIAM Review 53(2):217–288, [doi:10.1137/090771806](https://doi.org/10.1137/090771806))
  — the randomized range-finder bound, with the oversampling and power-iteration analysis. **The
  paper to cite**, and free on arXiv.
- Brand, M. (2006). [*Fast low-rank modifications of the thin singular value
  decomposition*](https://doi.org/10.1016/j.laa.2005.07.021), Linear Algebra Appl. 415(1):20–30 —
  the rank-one update at the core of this project.
- Balzano, Chi & Lu. *Streaming PCA and Subspace Tracking: The Missing Data Case* — a survey of
  the field; search by title, several preprint copies circulate.
