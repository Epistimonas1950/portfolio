# 07 · Pi 3: the Compressed Model, Actually Deployed

> The deployment capstone — an offline voice assistant instrumented so that every
> millisecond of its latency can be attributed to a stage. **The instrument is built,
> tested and measured. The deployment is not: there is no Raspberry Pi attached to the
> machine this was written on, so the latency budget this project exists to produce is
> an empty table with the right columns in it.**

**Status:** the voice-activity detector, the framing/WAV layer, the timing harness, the
percentile statistics and the orchestration state machine run and are tested here —
48 unit tests, numpy only. The four inference stages (whisper.cpp, llama.cpp prefill,
llama.cpp decode, Piper) fail by naming the binary they need. Every latency figure
derived from a `SimulatedStage` is labelled `SIMULATED` in the same breath as the
number. See [STATUS.md](STATUS.md).

**This README will not tell you how fast a Pi 3 is.** I have not measured one. What it
tells you is what I built to measure one with, and what that instrument does on
signals whose answers are known.

---

## The headline result

The one thing here that is genuinely measured, on this machine, today: **the VAD
recovers speech boundaries from noise, and its accuracy degrades monotonically as SNR
falls.** Synthetic signals with boundaries exact by construction, 20 seeds per point,
regenerate with `make results`; raw numbers in
[`results/vad_snr_sweep.csv`](results/vad_snr_sweep.csv).

| SNR | precision | recall | **F1** | onset err | offset err | words found |
|---:|---:|---:|---:|---:|---:|---:|
| 30 dB | 0.960 | 1.000 | **0.980** | 14 ms | 24 ms | 100% |
| 25 dB | 0.982 | 0.998 | **0.990** | 11 ms | 13 ms | 100% |
| 20 dB | 0.993 | 0.927 | **0.958** | 26 ms | 20 ms | 100% |
| 15 dB | 0.997 | 0.810 | **0.894** | 49 ms | 42 ms | 100% |
| 10 dB | 0.996 | 0.568 | **0.712** | 83 ms | 68 ms | 82% |
| 5 dB | 0.848 | 0.208 | **0.327** | 99 ms | 74 ms | 31% |
| 0 dB | 0.000 | 0.000 | **0.000** | — | — | 0% |

Two things in that table are load-bearing beyond the top-line F1.

**It fails in the right direction.** Precision stays above 0.99 down to 10 dB while
recall collapses to 0.57. An energy detector with a robust noise floor should fail by
missing quiet speech, not by hallucinating speech in noise, and this one does. A
detector that failed the other way would score a similar F1 and be useless in a room
with a fridge in it. The asserted version of that sentence is in
[`tests/test_vad.py`](tests/test_vad.py).

**The second feature earns its place.** Disabling the zero-crossing-rate endpoint
extension and changing nothing else:

| at 20 dB SNR | with ZCR | energy only |
|---|---:|---:|
| F1 | **0.958** | 0.881 |
| recall on unvoiced (fricative) frames | **0.809** | 0.455 |
| onset error | **26 ms** | 77 ms |

The unvoiced edges of a word sit about 5 dB over the noise floor, below the onset
threshold, so energy alone truncates every word by roughly 50 ms at its start. That is
what the second feature buys, and it is reported as a row of the CSV rather than
asserted in prose. Where it stops helping — at 30 dB, where energy alone already finds
everything and the extension costs a point of precision — the table says so.

And the deliverable the project is actually named after:

### [`results/latency_budget.md`](results/latency_budget.md) — empty, deliberately

| Stage | Median | p95 | Peak RSS | Notes |
|---|---|---|---|---|
| Audio capture + VAD | | | | |
| ASR (whisper.cpp tiny q5) | | | | |
| LLM prefill | | | | |
| LLM decode | | | | |
| TTS (Piper) | | | | |
| **End to end** | | | | |

Filling those cells needs a board. A filled-in version produced without one would be
fiction, and it is the only thing in this portfolio that would be genuinely
disqualifying.

---

## The problem

The brief is blunt about this project: *"Math: None new — that is the point."* It is
right, and the README should own that rather than dress it up. Wiring three inference
binaries together is not interesting. What is interesting is the claim underneath:

> I can attribute every millisecond of this pipeline to a stage, and show what the
> compression work in projects `01` and `02` bought.

That claim is an instrumentation claim, and instrumentation can be wrong in ways that
are invisible in the output. A budget table is only as good as the percentile function
that produced it and the detector that decided when the user stopped talking. So those
two are what this repo builds properly and tests hard, and the rest is scaffolded and
labelled.

## The mathematics that is here anyway

### 1. Two features, and why the obvious ZCR rule is backwards

Per frame `m` of length `N` at hop `R`:

```
E_m = 10 log10( (1/N) Σ_n x[mR+n]² + ε )              short-time energy, dB
Z_m = (1/(N-1)) Σ_n 1[ sgn x[mR+n] ≠ sgn x[mR+n-1] ]  zero-crossing rate
```

`E` in dB because the decision is about a *ratio* to a noise floor, and a ratio is a
difference in dB — which is what makes one threshold constant transferable between a
quiet room and a loud one.

`Z` is a cheap spectral-centroid proxy: for a zero-mean process the expected crossing
rate relates to the lag-one autocorrelation as `Z ≈ (1/π) arccos ρ₁`. The textbook
rule "high ZCR means unvoiced" is not a statement about speech, it is a statement about
the noise it is sitting in. Against a **white** floor — which has the maximum possible
crossing rate — a fricative is the *lower* of the two and the rule fires backwards. So
the implementation tests a **two-sided** deviation from the floor's own ZCR,
`|Z_m − Z_floor| > k·σ_Z`, and the synthetic floor in
[`src/synth.py`](src/synth.py) is band-limited pink rather than white so that the
distinction is actually exercised. → [`src/vad.py`](src/vad.py)

### 2. A noise floor that speech cannot poison

```
floor = median(E_0 … E_K)     σ = 1.4826 · median|E_k − floor|
```

Median, not mean: if the speaker starts early, up to half the estimation window can be
speech before the estimate moves. The 1.4826 makes the MAD a consistent estimator of
the standard deviation under a Gaussian, which is the only reason that constant exists.

The floor then keeps adapting, `floor ← (1−a)·floor + a·E_m`, **but only while the
state machine is confidently in silence** — so a fan spinning up is tracked while the
speaker's own voice can never raise the floor and deafen the detector.

### 3. Hysteresis, because one threshold chatters

Two thresholds, `floor + 8 dB` to open and `floor + 3 dB` to close, with three frames
of evidence to trigger and a hundred milliseconds of hangover to release. A single
threshold on a signal that happens to sit near it produces a burst of one-frame
segments. Measured, on a signal engineered to straddle a threshold with ±3 dB of
jitter: **one threshold gives 15 segments, two give 1.** The segment ends at the first
sub-threshold frame rather than at the end of the hangover, so robustness does not cost
100 ms on every reported duration. Both are assertions, not claims.

### 4. Percentiles, and why the budget's bottom row is not a sum

The brief asks for medians and p95, not means. The reason is worth stating: stage
latency on a small board is positive, right-skewed and heavy-tailed — a page fault or
a governor step adds a *multiple* of the typical cost, not a fraction. The mean is a
weighted average of the typical case and the worst case and tracks neither.

The harness implements the percentile itself (linear interpolation on order statistics,
R's type 7) and is checked against distributions with closed-form quantiles, with the
tolerance derived from the asymptotic standard error of a sample quantile

```
SE = sqrt( p(1−p)/n ) / f(Q(p))
```

rather than from a constant that happened to make the assertion pass. Three
distributions, for three different reasons — uniform (constant density, so an
interpolation error has nowhere to hide), exponential (thin density at p95, where an
off-by-one looks most like sampling noise) and the lognormal the simulator actually
draws from. → [`bench/latency.py`](bench/latency.py),
[`tests/test_latency.py`](tests/test_latency.py)

**And the total row of a latency budget is not the sum of the p95 column.** Under the
independent model the harness runs, summing the per-stage p95s **overstates the
end-to-end p95 by about 10%** (10.5% in the committed CSV), while the same sum of
medians lands within 1.3% of the measured end-to-end median. The p95 column is where
the error lives, and the p95 column is the one people add up. That is a measured
property of this model, not a theorem — quantiles are not subadditive in general, and
[`bench/latency.py`](bench/latency.py) spells out when the naive sum goes the other
way. The lesson that survives either way: a budget totalled by adding a p95 column
needs measuring end to end.

## Running it

No installation, no downloads, no network. numpy is the only requirement.

```bash
make test        # 48 tests, ~9 s
make demo        # one turn: real capture + real VAD + SIMULATED inference
make results     # regenerates the seeded CSVs in results/
make thermal     # host thermal log; unseeded, rewrites its CSV with a new answer
make plots       # figures; needs matplotlib in a venv, see requirements.txt
```

On a board, the one command that fills the empty table is
`python3 bench/latency.py --real`. It is the same harness with the same statistics
pointed at `SubprocessStage`s instead of draws — a constructor swap, not a rewrite.
Here it exits 1 and names the first thing it could not find.

The assertions the project rests on are marked in the source with
`=== THE TEST THAT MATTERS ===`, and there are two, because this project's anchor is
signal processing and statistics rather than the deployment claim — the deployment
claim needs hardware:

- the VAD recovers known boundaries across seven SNRs, with per-frame
  precision/recall, boundary error, and **monotone degradation** —
  [`tests/test_vad.py`](tests/test_vad.py)
- p50 and p95 from the harness match closed-form quantiles to within 5 standard
  errors on three distributions — [`tests/test_latency.py`](tests/test_latency.py)

alongside: hysteresis produces one segment where a single threshold produces fifteen;
a 60 ms blip is dropped by the minimum-duration rule and kept when the rule is
relaxed; the int16 WAV round trip is bit-exact over all 65 536 sample values; the
orchestrator stops at a failing stage and names it instead of continuing on a `None`
payload; and every `SubprocessStage` fails with a message naming its missing binary
and where to get it.

## Results

### What was measured here

| Quantity | Value | Where |
|---|---|---|
| VAD F1 at 20 dB SNR, 20 seeds | 0.958 | [`vad_snr_sweep.csv`](results/vad_snr_sweep.csv) |
| VAD onset error at 20 dB | 26 ms (hop is 10 ms) | same |
| ZCR extension, F1 gain at 20 dB | +0.077 | same |
| VAD wall clock, 4.3 s of 16 kHz audio | ~1.3 ms median — **on x86, not a Pi** | [`latency_simulated.csv`](results/latency_simulated.csv) |
| Harness dispatch overhead | ~5 µs median — the instrument's resolution floor | same |
| p95 overstatement of the naive sum | 10.5% (medians: 1.3%) | same |
| Host thermal log under sustained load | runs; one observed run lost 13% of throughput over 78 → 86 °C | [`thermal_host_x86_64.csv`](results/thermal_host_x86_64.csv) |

The two wall-clock rows drift between runs — they are real measurements of a shared
desktop, so they move with whatever else it is doing. The seeded rows above them do
not. That distinction is the whole reason the columns are separated.

The thermal file deserves a caveat louder than the number. It is a measurement of a
**desktop**, it has no seed, and what it records depends on how warm the machine
already was: four runs while writing this gave 68→71 °C with r = −0.26, 78→86 °C with
r = +0.57, 80→87 °C with r = +0.36, and 73→76 °C with no throttling at all. **The
committed file is whichever of those `make results` produced last, and re-running will
change it.** It is in the repo to show the
instrument works, and for one genuinely useful observation — **a positive correlation
between temperature and throughput is not a paradox.** Once a chip oscillates around
its cap, clock and temperature rise and fall together; the negative correlation people
expect belongs to the initial heating transient. A fifteen-second window cannot
separate the two, which is precisely why the brief asks for a *sustained* run.

### What was simulated

[`results/latency_simulated.csv`](results/latency_simulated.csv) contains draws from a
shifted lognormal, `T = floor + median·exp(σZ)`. Every row carries `source=SIMULATED`
and a note; the filename says it; this paragraph says it. **The medians in that file
are parameters I typed, not observations.** They exist so the percentile arithmetic,
the state machine and the budget composition can be exercised at all, and they will be
irrelevant the moment `bench/latency.py --real` runs against a board.

### What is empty

[`results/latency_budget.md`](results/latency_budget.md) — the real deliverable, its
columns and method fixed, its cells blank. The model-variant comparison (FP16 vs Q4 vs
low-rank vs both), the throttling curve and the demo video are absent for the reasons
in [STATUS.md](STATUS.md).

## The limitation I volunteer first

**This project's headline deliverable does not exist, and I am not going to pretend
otherwise.** The brief promises a measured latency budget on a Raspberry Pi 3. There
is no Pi. Everything downstream of the VAD in this repo is either a subprocess call to
a binary that is not installed or an explicitly labelled draw from a distribution I
chose. `results/latency_budget.md` is an empty table. If you are evaluating this repo
against the brief, that is the gap, and it is the first thing I would say in an
interview rather than the thing I would hope you did not check.

What I would say next is that I decided *how* to be blocked. The parts that do not
need hardware — the detector, the statistics, the state machine, the build
specification — are built to the standard the rest would be held to, and the seam
between them is a `Stage` interface with two implementations, so pointing it at real
binaries is a constructor change and not a rewrite.

**Second: the VAD is measured on synthetic audio, not speech.** The boundaries are
exact and the SNR is set rather than inferred, which is why the numbers are as sharp
as they are — but a generator I wrote cannot surprise me the way a real recording
would. Room reverberation, a television in the background, and the fact that real
unvoiced consonants are not band noise are all absent. I would expect the 20 dB row to
get meaningfully worse on real audio and I would want to know by how much before
trusting the detector in a product.

**Third: `setup/install.sh` and `setup/assistant.service` have never been run.** They
are reasoned specifications, syntax-checked and nothing more. I would expect the first
real attempt to find two or three wrong CMake option names, because both upstream
projects rename their build flags regularly.

**Fourth: I have not quoted a tokens-per-second figure for this board, from anyone.**
The brief is explicit that published numbers for this hardware class are scattered
single blog data points, and the interview claim I want to be able to make — *"I can
give you the latency budget stage by stage"* — is worth nothing if the numbers came
from someone else's blog. Better an empty cell than a borrowed one.

## References

- [How to Run an AI Server on a Raspberry Pi 3 with 1GB RAM](https://lepczynski.it/en/other/how-to-run-an-ai-server-on-a-raspberry-pi-3-with-1gb-ram/)
- [How Well Do LLMs Perform on a Raspberry Pi 5?](https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5) — for a sense of the newer-board ceiling
- [Modder crams LLM onto Raspberry Pi Zero-powered USB stick](https://www.tomshardware.com/raspberry-pi/raspberry-pi-zero/pi-zero-llm-usb-stick) — the floor

*Both reported-figure sources above are single blog data points. They are context, not
specifications, and nothing in this repo is calibrated against them.*

- `llama.cpp` and `whisper.cpp` upstream documentation — the authority for ARM build
  flags, and the reason `setup/build_flags.md` says to re-check the option names
  against the tree you actually clone.
- Piper (Rhasspy) — the TTS stage.
- Rabiner, L. R. & Sambur, M. R. *An Algorithm for Determining the Endpoints of
  Isolated Utterances*, Bell System Technical Journal, 1975 — the two-feature energy +
  zero-crossing endpoint rule that `src/vad.py` reimplements. I have read the method,
  not a copy of the paper, so it is cited by title and author without a link.

Full brief: [BRIEF.md](BRIEF.md).
