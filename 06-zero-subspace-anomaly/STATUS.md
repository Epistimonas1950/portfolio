# Status

Honest accounting of what runs, what is stubbed, and what needs hardware that is not here.
The portfolio's whole thesis is that its measurements are real, so this file is part of
the deliverable rather than an appendix to it.

**Last verified on this machine:**
`make host` → gcc 13.3.0, clean under `-Wall -Wextra -std=c11`, two binaries.
`make test` → **52 tests, OK**, ~7 s.
`make results` → 13 CSVs regenerated in `results/`.
`make demo` → tracker on `data/anomalous.csv`: `r = 4`, threshold 0.0102, 482 of 2 000
samples flagged, 5.5 µs/sample on this x86-64 host.
`make asan` → AddressSanitizer + UBSan build, all three modes, **clean** (including the
leak check at exit).
Python 3.12.3, numpy 1.26.4, gcc 13.3.0. No other dependency, on either side.

## Runnable today

### The C, with gcc and libm and nothing else

| Component | File | State |
|---|---|---|
| Dense primitives: dot, norms, matvec, matvec-T, gemm, gemm-TN | `src/linalg.c` | complete, self-tested |
| Householder thin QR | `src/linalg.c` | complete, self-tested (`A = QR`, `‖QᵀQ − I‖`) |
| One-sided Jacobi SVD, sorted descending | `src/linalg.c` | complete, self-tested (reconstruction, orthogonality, ordering) |
| PCG32 + Box–Muller, reproducible across machines | `src/linalg.c` | complete |
| Randomized range finder, subspace-iteration form | `src/rangefinder.c` | complete, checked against numpy and against the HMT bound |
| Randomized SVD via the sketch | `src/rangefinder.c` | complete |
| Brand rank-one incremental SVD, with the ρ guard | `src/incsvd.c` | complete, matches batch SVD to 9e-05° |
| Orthogonality monitor + factorization-preserving repair | `src/reorth.c` | complete, drift measured at two precisions |
| Exponential forgetting, λ ↔ window | `src/forget.c` | complete, round trip tested |
| Residual-energy detector + quantile threshold | `src/detect.c` | complete, matches numpy's quantile convention |
| Rank selection: energy threshold and gap criterion | `src/rank.c` | complete, both computed and both reported |
| Streaming CSV driver, `track` / `rangefind` / `selftest` | `host/main.c` | complete |
| float32 build (`-DSUBSPACE_FLOAT32`) | `Makefile` | complete, same source, `build/tracker32` |

### The Python oracle and the analyses, with numpy alone

| Component | File | State |
|---|---|---|
| Synthetic streams: 1 mode, 2 modes, 4 modes, rotating, labelled anomalies | `oracle/generate_data.py` | complete, seeded |
| Randomized range finder + rank criteria (numpy reference) | `oracle/rangefinder.py` | complete |
| Brand update + reorth + forgetting + detector (numpy reference) | `oracle/incremental.py` | complete |
| Batch full-SVD oracle, and the spectrum/rank report | `oracle/batch_svd.py` | runs, CSV committed |
| ROC and AUC in pure numpy, tie-corrected | `oracle/roc.py` | runs, CSVs committed |
| C-versus-oracle comparison, subspaces and scores | `oracle/compare.py` | runs, CSVs committed |
| Subprocess wrapper around the compiled binary | `oracle/chost.py` | complete |
| Orthogonality-drift study, both precisions, both implementations, four streams | `analysis/drift_study.py` | runs, 3 CSVs committed |
| Rank × forgetting-window sweep over all three scenarios | `analysis/multimode_study.py` | runs, CSV committed |
| Oversampling / power-iteration study against the HMT bound | `analysis/rangefinder_study.py` | runs, CSV committed |
| Forgetting-window sweep on a rotating subspace | `analysis/forgetting_study.py` | runs, CSV committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed on this machine. It exits with a message naming the package instead of failing obscurely. Every number it would draw is already in `results/*.csv`, which is why no `.png` is committed. |
| The armv6 cross-compile (`make cross-note`) | no cross toolchain is installed and no board is attached. The recipe is printed, not executed. Nothing in `src/` uses an intrinsic, a compiler extension or a library beyond libm, so the only thing that changes is the triple — but "should cross-compile" is a claim, not a measurement, and it is labelled as one. |

## Not built, because the hardware is not here

There is **no Raspberry Pi Zero, no Zero 2 W, no I²S microphone, no ADXL345 and no INA219**
attached to the machine this repo was built on. The following parts of
[BRIEF.md](BRIEF.md) are therefore absent rather than mocked, and the corresponding cells
in the README's table are empty rather than estimated:

- **Data acquisition.** `src/sensor.c` in the brief's layout does not exist. All data is
  synthetic and `oracle/generate_data.py` says so in its first paragraph.
- **On-board memory and latency.** The memory figures in the README are the program's own
  accounting of its heap (`tracker_bytes`, printed by the binary), which is exact but is
  not a `/proc/self/status` reading on a Zero. Every timing figure is x86-64 host and is
  labelled as such everywhere it appears. There is no armv6 number in this repo.
- **The TFLite autoencoder baseline (Zero 2 W).** No board, no `tflite-runtime` wheel
  installable here, no measurement. The row in the results table is empty. On an *original*
  Zero this comparison cannot be made at all, for the reason the project exists.
- **A real labelled fault.** "Record normal operation, then induce anomalies deliberately"
  needs a machine to induce them on.

## Known limitations of what *is* here

These are real and are argued in the README rather than hidden:

- **Rank is fixed after warm-up.** Adaptive rank during streaming is not implemented. On
  the four-mode stream neither automatic criterion finds the rank-13 union, which is the
  repo's headline limitation and is measured in `results/multimode_sweep.csv`.
- **The threshold is calibrated once.** A rolling recalibration on samples the detector
  itself calls normal is the obvious fix and is not implemented, so a drift in the healthy
  noise level moves the false-positive rate and nothing notices.
- **Anomalous samples still update the subspace.** There is no gating on the detector's own
  output. With `N_eff = 400` a single spike moves the subspace by about 1/400, so isolated
  anomalies are harmless; a sustained fault would eventually be learned as normal.
- **A mixture of subspaces is not built.** It is the right fix for multiple operating modes
  and the README says so without pretending to have implemented it.
- **`--repeat` re-reads the file.** It exists solely so the drift study can reach 22 200
  updates from a 1 500-sample file, and it is the only code path that reads a sample twice.
  At `repeat = 1` the tracker is a genuine single pass.

## What the numbers here do and do not support

**Supported.** Brand's rank-one update reproduces a batch full SVD to 9e-05 degrees and its
singular values to 1e-09 relative. Orthogonality drift grows 26.7× (double) and 625×
(float32) over 22 200 updates without repair and is held at 1.3× the threshold with it, for
35 repairs; the C drifts 2.7–23× less than the numpy oracle across four streams, a
difference I have measured but not attributed to a specific step. The randomized range finder reaches 1.0000× the Eckart–Young optimum at
`p = 16, q = 2` and satisfies the HMT Frobenius bound in expectation. Exponential
forgetting tracks a 90° subspace rotation to 3.3° where `λ = 1` reaches 36.0°, with an
interior optimum in the detector's residual. The detector reaches AUC 0.985 on
out-of-subspace spikes, within 0.0002 of the batch ceiling, and collapses to 0.648 and
0.580 on two- and four-mode normal regimes — recovering to 0.986 and 0.978 at the union
rank. The C agrees with the numpy oracle to 1e-05 relative on scores and 5e-07 on
subspaces. The tracker's steady-state heap is 3 776 bytes in double and 1 948 in float32.

**Not supported.** Any claim about a Raspberry Pi Zero. Any latency, throughput, power or
memory figure on ARM. Any comparison against TFLite. Any claim about real sensor data, real
faults, or a real machine's operating modes. Those need the hardware listed above, and
until it is attached the cells stay empty.
