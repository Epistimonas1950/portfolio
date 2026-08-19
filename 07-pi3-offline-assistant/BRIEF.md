# 07 · Pi 3: the Compressed Model, Actually Deployed

> The capstone — project `01` and `02`'s output running offline on 1 GB of RAM.

| | |
|---|---|
| **Effort** | 1 week |
| **Prerequisites** | **`01` and `02` must be finished** — this deploys their output |
| **Hardware** | Raspberry Pi 3 / 3B+, 64-bit Pi OS, USB microphone, speaker |
| **Math** | None new — that is the point |
| **Status** | ☐ not started |

---

## Know the board before you plan the project

| | Pi 3 / 3B+ |
|---|---|
| Core | 4× Cortex-A53 @1.2–1.4 GHz |
| RAM | 1 GB (shared with the GPU) |
| LLM | **barely** — a 4-bit sub-1B model, slowly |
| Also viable | whisper.cpp tiny, Piper TTS, nano-class detectors |

This is the only board in the set where a language model is genuinely usable. It will still be
slow. **Plan the project around that fact instead of hiding it.**

Two non-negotiables:

1. **Use the 64-bit OS.** The 32-bit image costs you meaningful inference throughput on ARMv8.
2. **Never train on the board.** Train and compress on your laptop or a rented GPU; the Pi runs
   inference only.

## What it is

A fully offline voice assistant, nothing leaving the device:

```
USB mic ──► whisper.cpp (tiny, quantized)  ──► text
                                               │
                    your compressed model (01 + 02) via llama.cpp
                                               │
                                               ▼
            Piper TTS ──► speaker            response text
```

The interest is not the assistant. Anyone can wire three binaries together. **The interest is the
latency budget** — and the fact that you can attribute every millisecond of it, and show what your
compression work bought.

## The real deliverable: a latency budget

Measure each stage separately, then end to end. Report medians and p95, not means.

| Stage | Median | p95 | Peak RSS | Notes |
|---|---|---|---|---|
| Audio capture + VAD | | | | |
| ASR (whisper.cpp tiny q5) | | | | ×real-time factor |
| LLM prefill | | | | ms/token, prompt length dependent |
| LLM decode | | | | **tok/s — the headline number** |
| TTS (Piper) | | | | |
| **End to end** | | | | first-audio-out latency |

And the comparison that justifies projects `01` and `02` existing:

| Model variant | Size on disk | Peak RSS | Decode tok/s | Perplexity |
|---|---|---|---|---|
| FP16 baseline | | *(likely OOM on 1 GB — report that)* | | |
| Q4 quantized only (`02`) | | | | |
| Low-rank only (`01`) | | | | |
| **Low-rank + Q4 (`01`+`02`)** | | | | |

If the FP16 baseline does not fit in 1 GB, **that is a result** — state it plainly. "The
uncompressed model does not run on this hardware at all; mine does, at N tok/s, with X perplexity
degradation" is a stronger sentence than any speedup ratio.

## What to build

- [ ] 64-bit Pi OS Lite image, headless, documented setup script
- [ ] Cross-compile or on-device build of `llama.cpp` and `whisper.cpp` (record the flags — NEON
      matters)
- [ ] GGUF conversion of your compressed model from `01`/`02`
- [ ] Piper TTS with a small voice model
- [ ] Orchestration script with per-stage timing instrumentation
- [ ] Simple VAD so it isn't push-to-talk
- [ ] `systemd` unit so it survives reboot — this is the "it's a real deployment" detail
- [ ] Thermal check: sustained load on a Pi 3 throttles. Log temperature alongside throughput and
      show the throttling curve. Most people never look.
- [ ] A short video of it working, in the README

## The limitation you volunteer first

**On a Pi 3 this will be slow — likely a few tokens per second at best.** Frame it as a
latency-budget study rather than a product demo, and the slowness becomes the interesting part:
you know precisely where the time goes and what each optimization returned.

Do not quote someone else's benchmark for your board. Published figures for this hardware class
are scattered single data points from blog posts; **measure your own and publish the method**.
That is the difference between citing and knowing.

## Interview claim

> I took the compression work from theory to a 1 GB device, and I can give you the latency budget
> stage by stage — including the point where the board thermally throttles and what that does to
> sustained throughput.

## Stack

`llama.cpp` · `whisper.cpp` · Piper · `systemd` · 64-bit Raspberry Pi OS Lite

## Suggested repo layout

```
pi3-offline-assistant/
  README.md              <- latency table + video at the top
  setup/
    install.sh           reproducible from a fresh image
    assistant.service    systemd unit
    build_flags.md       NEON / arch flags, and why
  src/
    orchestrate.py       pipeline + per-stage timing
    vad.py
  bench/
    latency.py
    thermal.py           throughput vs. temperature over time
  results/
    latency_budget.md
    thermal_throttling.png
    demo.mp4
```

## References

- [How to Run an AI Server on a Raspberry Pi 3 with 1GB RAM](https://lepczynski.it/en/other/how-to-run-an-ai-server-on-a-raspberry-pi-3-with-1gb-ram/)
- [How Well Do LLMs Perform on a Raspberry Pi 5?](https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5) — for a sense of the newer-board ceiling
- [Modder crams LLM onto Raspberry Pi Zero-powered USB stick](https://www.tomshardware.com/raspberry-pi/raspberry-pi-zero/pi-zero-llm-usb-stick) — the floor
- `llama.cpp` and `whisper.cpp` upstream documentation for ARM build flags

*Both reported-figure sources above are single blog data points. Treat them as context, not
specifications, and measure your own.*
