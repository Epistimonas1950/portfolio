# Status

Honest accounting of what runs, what is stubbed, and what is missing. This project is
the portfolio's deployment capstone and **its headline deliverable is not present**, so
this file matters more here than in any of the other seven.

**Last verified:** `python3 -m unittest discover -s tests -t . -v` → **48 tests, OK**,
~9 s. `make results` → 2 seeded CSVs regenerated; `make thermal` → 1 unseeded CSV. Python 3.12.3, numpy 1.26.4, no other
dependency.

---

## The one-line version

There is **no Raspberry Pi 3** attached to the machine this was built on, and no
`llama.cpp`, `whisper.cpp`, Piper, GGUF model, microphone or speaker. The latency
budget the brief asks for is therefore an empty table with correct columns
([`results/latency_budget.md`](results/latency_budget.md)). What is built and tested is
everything that does not need the board.

## Runnable today, with numpy alone

| Component | File | State |
|---|---|---|
| WAV read/write (stdlib `wave`), int16 ↔ float scaling | `src/audio.py` | complete, tested (bit-exact round trip) |
| Frame slicing, hop/window, frame↔time convention | `src/audio.py` | complete, tested |
| Short-time energy + zero-crossing rate | `src/vad.py` | complete, tested against closed-form ZCR of sinusoids |
| Robust noise-floor estimate (median + MAD), adaptive | `src/vad.py` | complete, tested |
| Two-threshold hysteresis state machine | `src/vad.py` | complete, tested (1 segment vs 15) |
| Minimum-duration + gap-merge smoothing | `src/vad.py` | complete, tested |
| ZCR endpoint extension (Rabiner–Sambur style) | `src/vad.py` | complete, **ablated** in the results CSV |
| VAD scoring: frame P/R/F1, boundary error, match rate | `src/vad.py` | complete, tested |
| Labelled synthetic speech generator, exact boundaries | `src/synth.py` | complete |
| `Stage` abstraction with per-stage timing | `src/stages.py` | complete, tested |
| `SimulatedStage` + lognormal latency model | `src/stages.py` | complete, tested against closed-form quantiles |
| Orchestration state machine + JSONL logging | `src/orchestrate.py` | complete, tested (failure propagation) |
| Percentile / p50 / p95 arithmetic | `bench/latency.py` | complete, tested to within 5 SE on 3 distributions |
| VAD accuracy sweep vs SNR, with ZCR ablation | `bench/vad_snr_sweep.py` | runs, CSV committed |
| Latency harness | `bench/latency.py` | runs, CSV committed (**SIMULATED**, labelled) |
| Thermal / throughput logger | `bench/thermal.py` | runs on this host, CSV committed (**host, not Pi**; own `make thermal` target, see below) |

## Present, correct, and guaranteed to fail here — by design

| Component | What happens | Why |
|---|---|---|
| `SubprocessStage` for `whisper-cli` | `MissingBinaryError` naming the binary, its upstream repo and `setup/install.sh` | whisper.cpp is not installed and cannot be built here (no network) |
| `SubprocessStage` for `llama-cli` (prefill + decode) | same | llama.cpp is not installed |
| `SubprocessStage` for `piper` | same | Piper is not installed |
| `arecord` capture stage | fails with the ALSA error verbatim: `audio open error: No such file or directory` | `alsa-utils` *is* installed on this host, so this one gets past the binary check and dies at the missing capture device instead. Still a clear, named failure; it is listed here because it is the one stage that does not fail the way the others do. |
| `bench/plot_results.py` | exits naming matplotlib and the venv command | matplotlib is not installed; every number it would draw is already in `results/*.csv` |

`MissingBinaryError` subclasses `NotImplementedError`, per the portfolio convention
that a stub says it is a stub. None of them falls back to anything.

## Written but never executed

| Component | Verified how far |
|---|---|
| `setup/install.sh` | `bash -n` clean; `set -euo pipefail`, all variables quoted, refuses to run on anything but `aarch64`, restores swap in an `EXIT` trap. Has never run on a board. |
| `setup/assistant.service` | Reasoned directive by directive, never loaded by systemd. |
| `setup/build_flags.md` | An argument, not a measurement. It says so at the top. |

I expect the first real run of `install.sh` to find two or three wrong CMake option
names; `llama.cpp` and `whisper.cpp` both rename build flags regularly.

## Blocked on hardware

None of this is mocked. It is absent.

| Deliverable | Needs |
|---|---|
| Every cell of the latency budget | Raspberry Pi 3 / 3B+, 64-bit Pi OS Lite, USB mic, speaker |
| ASR row | whisper.cpp built on the board + `ggml-tiny.en-q5_1.bin` |
| LLM prefill / decode rows, tok/s | llama.cpp built on the board + a GGUF |
| TTS row | Piper aarch64 binary + a low-quality voice model |
| Peak RSS column | the board (RSS on x86 says nothing about a 1 GB ARM board) |
| Thermal throttling curve, `thermal_throttling.png` | the board plus a sustained ~30 min inference load |
| `demo.mp4` | the board, working, with audio |

## Blocked on the other portfolio projects

| Deliverable | Needs |
|---|---|
| The model-variant comparison table (FP16 / Q4 / low-rank / both) | **projects `01` and `02` finished**, and `02`'s GGUF export in particular |
| "The uncompressed model does not run on this hardware at all; mine does" | the FP16 baseline to actually be attempted on the board |
| Perplexity column | project `02`'s real-model evaluation, which is itself not run yet |

`setup/install.sh` deliberately refuses to download a substitute model. A stand-in
would invalidate the only claim this project exists to support.

## What the numbers here do and do not support

**Supported.** The detector works and fails in the right direction: F1 0.958 at 20 dB
SNR with 26 ms onset error, degrading monotonically to 0 at 0 dB, with precision
holding above 0.99 down to 10 dB while recall collapses. The ZCR endpoint extension is
worth +0.077 F1 and 51 ms of onset accuracy at 20 dB, and nothing at 30 dB. The
percentile arithmetic matches closed-form quantiles of three distributions to within 5
standard errors. Summing per-stage p95s overstates end-to-end p95 by ~10% under the
harness's independent lognormal model, where the same sum of medians is within 1.3%. The harness's own dispatch overhead is 5.4 µs
median, measured — the resolution floor of the instrument.

**Not supported.** Any statement about how fast anything runs on a Raspberry Pi 3.
Any tokens-per-second figure. Any peak-RSS figure. Any claim about thermal throttling
on that board. Any claim that the compressed model from `01`/`02` fits in 1 GB, or that
the FP16 baseline does not. Any claim that `install.sh` works.

## A note on reproducibility

Every CSV produced by `make results` is deterministic from a seed. The one exception
is `results/thermal_host_x86_64.csv`, which has no seed and cannot have one: it records
what this machine's temperature and throughput did during one particular fifteen
seconds, and that depends on how warm it already was. Four runs gave 68→71 °C
(r = −0.26), 78→86 °C (r = +0.57), 80→87 °C (r = +0.36) and 73→76 °C with no
throttling. **It is therefore not part of `make results`** — it has its own
`make thermal` target, so regenerating the results does not produce a spurious diff.
The committed file is one captured run, kept to show the instrument works, not as a
result.

`results/turn_log.jsonl` is a demo artefact and is gitignored.
