# Latency budget — Raspberry Pi 3 / 3B+

> **UNMEASURED. Every cell below is empty because there is no Raspberry Pi attached to
> the machine this repo was built on.** This file is the deliverable's template: the
> columns, the method and the definitions are fixed here so that filling it in is a
> mechanical act once hardware exists. A filled-in version of this table produced
> without a board would be fiction, and this portfolio's whole thesis is that it
> doesn't do that.
>
> Do not confuse this file with `results/latency_simulated.csv`. That file contains
> draws from a lognormal distribution chosen to exercise the arithmetic in
> `bench/latency.py`. It is labelled `SIMULATED` in its filename, in a column on every
> row, and in a note on every row. It is not a prediction of anything below.

**Board:** *(not present)*
**OS:** 64-bit Raspberry Pi OS Lite, version *(unfilled)*
**Build:** `-mcpu=cortex-a53 -mtune=cortex-a53 -O3`, see [`setup/build_flags.md`](../setup/build_flags.md)
**Model:** GGUF export from projects `01` + `02` — *(not yet produced)*
**Method when run:** `bench/latency.py --real --turns 200`, medians and p95 over 200
turns, `perf_counter`, one stage at a time and then end to end. Not means — see the
module docstring in [`bench/latency.py`](../bench/latency.py) for why.

---

## The budget

| Stage | Median | p95 | Peak RSS | Notes |
|---|---|---|---|---|
| Audio capture + VAD | | | | the VAD half runs today at ~1.3 ms median for 4.3 s of 16 kHz audio — on an x86 host, not this board, and therefore not a number for this table |
| ASR (whisper.cpp tiny q5) | | | | ×real-time factor |
| LLM prefill | | | | ms/token, prompt-length dependent |
| LLM decode | | | | **tok/s — the headline number** |
| TTS (Piper) | | | | |
| **End to end** | | | | first-audio-out latency |

Note on the total row: it is **not** the sum of the per-stage p95 column. p95 of a sum
is not the sum of p95s, and under the independent model in `bench/latency.py` the
naive sum overstates the end-to-end p95 by ~10%. The end-to-end row must be measured
end to end.

## The comparison that justifies projects 01 and 02

| Model variant | Size on disk | Peak RSS | Decode tok/s | Perplexity |
|---|---|---|---|---|
| FP16 baseline | | *(expected to OOM on 1 GB — report it if it does)* | | |
| Q4 quantized only (`02`) | | | | |
| Low-rank only (`01`) | | | | |
| **Low-rank + Q4 (`01`+`02`)** | | | | |

If the FP16 baseline does not fit in 1 GB, that is a result and gets stated plainly
rather than left as a blank.

## Thermal

| | |
|---|---|
| Idle temperature | |
| Steady-state temperature under sustained decode | |
| Time to first throttle (80 °C soft cap) | |
| Sustained decode tok/s after 30 min | |
| Throughput retained vs. first minute | |

`bench/thermal.py` is the instrument and it works today — it produced
[`thermal_host_x86_64.csv`](thermal_host_x86_64.csv) on the development machine. That
file is a measurement of a desktop, it is unseeded, and `make results` overwrites it
with a different answer every time; it is in the repo only to show the harness runs.
The rows above need the board.

## What each cell requires

| Cell | Blocked on |
|---|---|
| every row of the budget | a Raspberry Pi 3 with a USB mic and speaker |
| ASR row | `whisper.cpp` built on the board + a tiny.en q5 model |
| LLM rows | `llama.cpp` built on the board + the GGUF from `01`/`02` |
| TTS row | Piper aarch64 binary + a low-quality voice |
| the model-variant table | projects `01` and `02` finished and exporting GGUF |
| the thermal rows | the board, plus a sustained 30-minute run |
