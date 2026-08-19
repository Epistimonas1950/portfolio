# Build flags, and why each one is there

Nothing on this page has been run. There is no Raspberry Pi attached to the machine
this repo was written on, so this is a reasoned specification of the build rather than
a report of one. Where a claim is a measurement someone else made, it says so.

---

## The two non-negotiables from the brief

**1. Use the 64-bit OS.** Not the 32-bit image, even though the Pi 3 boots it happily
and most tutorials assume it.

The Pi 3's Cortex-A53 implements ARMv8-A, which has two instruction sets: AArch64 and
AArch32. The 32-bit image runs the second one, and the difference is not cosmetic:

| | AArch32 (armhf) | AArch64 (arm64) |
|---|---|---|
| General-purpose registers | 16 × 32-bit | 31 × 64-bit |
| NEON / SIMD registers | 16 × 128-bit | **32 × 128-bit** |
| Addressable per process | 4 GB (3 GB usable) | 64-bit |

`llama.cpp`'s quantized matrix-vector kernels are the inner loop of decode, and they
are register-pressure bound: each one holds a block of quantized weights, its scale,
an accumulator and a slice of the activation vector in vector registers at once. Half
the vector register file means spilling to the stack inside the loop that runs
billions of times. The 64-bit ABI's larger integer register file matters for the same
reason in the dequantization path.

I have not measured the gap on this board and will not quote a number for it. The
argument for arm64 does not depend on the size of the gap: it costs nothing, the OS is
a free download, and there is no compensating advantage to armhf on a 1 GB board.

**2. Never train on the board.** The Pi runs inference only. Training or quantizing on
1 GB of shared RAM, on an SD card, on four 1.2 GHz cores, is not slow-but-workable —
it is a way to corrupt an SD card. Projects `01` (low-rank factorization) and `02`
(quantization) run on a laptop or a rented GPU and produce a GGUF; that file is copied
across with `scp`. `setup/install.sh` refuses to download a substitute model for
exactly this reason: a stand-in model would silently invalidate the whole point of the
latency budget, which is to show what *this* compression bought.

---

## The compiler flags

```
-mcpu=cortex-a53 -mtune=cortex-a53 -O3
```

**`-mcpu=cortex-a53`** sets the architecture *and* the scheduling model together. It
implies `-march=armv8-a+crc` and it is what unlocks NEON code generation without any
further flag — on AArch64, Advanced SIMD is part of the base architecture and is
always available, so there is no `-mfpu=neon` to write. (That flag is AArch32-only,
and seeing it in a build script is a reliable sign the script was copied from a 32-bit
Pi 2 guide.)

The `-mtune` half is the part that is easy to skip and shouldn't be. The A53 is
**in-order, dual-issue**. It has no out-of-order window to hide a badly scheduled
load, so instruction ordering that costs a big out-of-order core nothing costs this
core real cycles. `-mtune=cortex-a53` gives GCC the correct latency and issue-width
tables to schedule against. It changes no instruction *selection*, only ordering, so
it is free of correctness risk.

**Why not `-march=native` / `-DGGML_NATIVE=ON`?** Two reasons. It is inert when you
are already naming the exact part, and it silently produces a binary that will not run
on any other board — which matters the moment you cross-compile, or build on a Pi 4
and deploy to a Pi 3. `GGML_NATIVE=OFF` plus an explicit `-mcpu` is the reproducible
form, and reproducibility is the deliverable here.

**Why not `-Ofast` or `-ffast-math`?** `-ffast-math` sets `-ffinite-math-only` and
flushes denormals, and it lets the compiler reassociate floating-point addition. In a
quantized inference path the accumulations are already the numerically delicate part,
and a softmax with `inf`/`NaN` checks compiled under `-ffinite-math-only` can go
quietly wrong. The gain would be small — the hot kernels are hand-written intrinsics
that the optimizer barely touches — and an unquantifiable accuracy change in exchange
for an unmeasured speedup is a bad trade in a project whose subject is knowing where
the time goes.

**`-O3` over `-O2`** for the vectorizer, on the non-intrinsic scalar glue.

## Threads

`-t 4`, matching the core count, and no higher. There is no SMT on an A53, so a fifth
thread cannot run anywhere; it only adds context switches and steals the core the ALSA
callback needs. The systemd unit pins `OMP_NUM_THREADS=4` for the same reason.

Whether 4 is actually better than 3 on this board is an open question I would want to
answer with `llama-bench` rather than by assertion — decode is memory-bandwidth bound
and the fourth thread may buy less than it costs. That measurement is in
`results/latency_budget.md`, unfilled.

## BLAS

`libopenblas-dev` is installed and linked. It helps **prefill**, which is a matrix-
matrix product over the whole prompt, and does essentially nothing for **decode**,
which is matrix-*vector* and memory-bandwidth bound. This is precisely why the two are
separate stages in `src/stages.py` and separate rows in the budget: they respond to
different optimizations, and a single "LLM" row would average away the only signal a
person tuning this would want.

## Memory, on a 1 GB board shared with the GPU

- **`gpu_mem=16`** in `/boot/firmware/config.txt`. Headless, so the GPU needs the
  minimum. The default 64 MB is 6% of total RAM given to a framebuffer nobody looks at.
- **Swap: raise to 1 GB for the build, drop back to 100 MB after.** `llama.cpp`'s
  larger translation units peak over 1 GB under `-O3` and will not link otherwise.
  Leaving a big SD-card swapfile enabled afterwards wears the card and makes inference
  latency spiky — which would pollute exactly the p95 this project exists to report.
  `setup/install.sh` restores it in an `EXIT` trap so a failed build still cleans up.
- **`mmap` the GGUF** (llama.cpp's default). The model pages in on demand and the page
  cache can evict it under pressure instead of the OOM killer choosing for you. The
  cost is that first-token latency includes SD-card reads, which is worth knowing and
  is why "first-audio-out latency" is its own row in the budget.
- **Q4_K_M over Q4_0** as the starting point, on the argument that its per-block
  scales cost a few percent of size for a meaningful quality gain. Whether that trade
  holds for *this* compressed model is a question for project `02`'s error analysis
  plus a perplexity run, not for this page.

## What is deliberately absent

No `-DGGML_BLAS=ON` for decode benchmarking without a measurement first; no
`-march=native`; no cross-compilation toolchain. Cross-compiling would be faster than
the ~40 minutes this takes on-device, but it adds a sysroot and a second set of
libraries to get wrong, and the build is a one-time cost.

## References

Upstream `llama.cpp` and `whisper.cpp` build documentation are the authority for the
current CMake option names; both projects rename options regularly, and the flags
above should be checked against the tree that is actually cloned.
