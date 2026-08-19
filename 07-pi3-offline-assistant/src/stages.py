"""The four pipeline stages, with per-stage timing built into the abstraction.

Each stage of the assistant -- ASR, LLM prefill, LLM decode, TTS -- has exactly two
implementations here, and the difference between them is the whole honesty story of
this repo:

`SubprocessStage`
    What a real deployment runs: shell out to `whisper-cli`, `llama-cli` or `piper`.
    On this machine none of those binaries exist and none can be built (no ARM board,
    no network), so the stage raises `MissingBinaryError` -- a subclass of
    `NotImplementedError` -- naming the binary, saying where it comes from, and
    pointing at `setup/install.sh`. It never falls back to anything.

`SimulatedStage`
    Draws a latency from an explicit distribution. It exists so that the statistics
    harness, the state machine and the budget arithmetic can be tested at all, and for
    no other reason. Every value it produces is stamped `simulated=True`, every CSV row
    derived from it carries `source=SIMULATED`, and the file it lands in is called
    `latency_simulated.csv`. **No number originating here is a measurement of a
    Raspberry Pi 3, or of anything else.**

Timing
------

`Stage.run` wraps `_execute` in `time.perf_counter()` and reports two numbers, which
are never conflated:

    latency_ms            what the stage cost. For a real stage this is the measured
                          wall clock. For a simulated stage it is the DRAWN value.
    harness_overhead_ms   measured wall clock of everything around the drawn value --
                          dispatch, dataclass construction, logging. For a real stage
                          this is 0 by definition (the measurement is the stage).

The overhead column is not decoration. It is a genuine measurement made on this host,
and it is the resolution floor of the instrument: any future per-stage figure from a
Pi is only meaningful well above it. `bench/latency.py` reports its median.

`perf_counter` rather than `time.time`: it is monotonic and unaffected by NTP steps,
which matters on a headless Pi that corrects its clock shortly after boot -- exactly
when a systemd-launched assistant is taking its first measurements.

The latency model
-----------------

`LatencyModel` is a shifted lognormal:

    T = floor_ms + median_ms * exp(sigma * Z),      Z ~ N(0, 1)

chosen because latency is positive, right-skewed and has a hard floor, which is what
a lognormal plus a shift is. It also has closed-form quantiles,

    Q(p) = floor_ms + median_ms * exp(sigma * z_p)

with `z_p` the standard normal quantile, so the harness's empirical p50 and p95 can be
checked against exact values rather than against another implementation of the same
approximation. That check is one half of this project's anchor test.
"""

from __future__ import annotations

import abc
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# Standard-normal quantiles, hardcoded to avoid a scipy dependency that is not
# installed. Values from the standard normal CDF; used only for closed-form checks.
Z_P50 = 0.0
Z_P95 = 1.6448536269514722
Z_P99 = 2.3263478740408408


class MissingBinaryError(NotImplementedError):
    """A real inference binary is not present. Deliberately a NotImplementedError:
    this is a stub in the sense the portfolio conventions mean, not a runtime fault."""


class StageFailure(RuntimeError):
    """A stage ran and failed. Distinct from a missing binary."""


@dataclass
class StageResult:
    """One stage's outcome. `simulated` is carried, not inferred, all the way to CSV."""

    stage: str
    ok: bool
    latency_ms: float
    harness_overhead_ms: float
    simulated: bool
    payload: Any = None
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        """Flat, JSON-safe, payload dropped. What the structured log and CSV get."""
        return {
            "stage": self.stage,
            "ok": self.ok,
            "source": "SIMULATED" if self.simulated else "measured",
            "latency_ms": round(self.latency_ms, 4),
            "harness_overhead_ms": round(self.harness_overhead_ms, 6),
            "error": self.error,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


@dataclass
class ExecOut:
    """What `_execute` hands back. `latency_ms=None` means 'time me'."""

    payload: Any
    latency_ms: float | None = None
    meta: dict = field(default_factory=dict)


class Stage(abc.ABC):
    """Base class. Subclasses implement `_execute`; timing is not their business."""

    simulated: bool = False

    def __init__(self, name: str):
        self.name = name

    @property
    def label(self) -> str:
        return f"SIMULATED {self.name}" if self.simulated else self.name

    @abc.abstractmethod
    def _execute(self, payload: Any) -> ExecOut:
        ...

    def run(self, payload: Any = None) -> StageResult:
        """Execute, time, and never raise. Failures come back as `ok=False`.

        Returning rather than raising is what lets the orchestrator record a partial
        turn, log which stage died, and report it -- instead of unwinding the stack
        and losing every timing collected so far.
        """
        t0 = time.perf_counter()
        try:
            out = self._execute(payload)
        except Exception as exc:  # noqa: BLE001 -- deliberate: reported, not swallowed
            elapsed = (time.perf_counter() - t0) * 1e3
            return StageResult(
                stage=self.name, ok=False, latency_ms=elapsed,
                harness_overhead_ms=0.0, simulated=self.simulated,
                error=f"{type(exc).__name__}: {exc}",
            )
        measured_ms = (time.perf_counter() - t0) * 1e3
        if out.latency_ms is None:
            latency, overhead = measured_ms, 0.0
        else:
            # The stage reported its own cost, so everything measured here is the
            # harness's own overhead around it.
            latency, overhead = float(out.latency_ms), measured_ms
        return StageResult(
            stage=self.name, ok=True, latency_ms=latency,
            harness_overhead_ms=overhead, simulated=self.simulated,
            payload=out.payload, meta=out.meta,
        )


# ------------------------------------------------------------------ the real thing


class SubprocessStage(Stage):
    """Shells out to an inference binary. Absent here; fails by naming what is absent."""

    simulated = False

    def __init__(
        self,
        name: str,
        binary: str,
        argv: Callable[[Any], Sequence[str]] | Sequence[str],
        source: str,
        parse: Callable[[subprocess.CompletedProcess], Any] | None = None,
        timeout_s: float = 300.0,
        alt_names: Sequence[str] = (),
    ):
        super().__init__(name)
        self.binary = binary
        self.alt_names = tuple(alt_names)
        self.argv = argv
        self.source = source
        self.parse = parse
        self.timeout_s = timeout_s

    def resolve(self) -> str | None:
        """First of `binary` or its historical names found on PATH."""
        for candidate in (self.binary, *self.alt_names):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def _execute(self, payload: Any) -> ExecOut:
        exe = self.resolve()
        if exe is None:
            names = " / ".join((self.binary, *self.alt_names))
            raise MissingBinaryError(
                f"stage '{self.name}' requires the '{self.binary}' binary and it is "
                f"not on PATH (searched: {names}). It is not vendored in this repo "
                f"and cannot be built on this host -- it is built on the target board "
                f"by setup/install.sh. Upstream: {self.source}. "
                f"To exercise the pipeline without it, use SimulatedStage, whose "
                f"output is labelled SIMULATED and is not a measurement."
            )
        cmd = [exe, *(self.argv(payload) if callable(self.argv) else self.argv)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            raise StageFailure(
                f"{self.binary} exited {proc.returncode}: "
                f"{proc.stderr.strip()[:400] or '(no stderr)'}"
            )
        result = self.parse(proc) if self.parse else proc.stdout
        # latency_ms=None -> the wrapper's measured wall clock is the latency.
        return ExecOut(payload=result, meta={"binary": exe, "argv_len": len(cmd)})


def whisper_stage(model: str = "models/ggml-tiny.en-q5_1.bin", threads: int = 4) -> SubprocessStage:
    """ASR. `whisper-cli` is the current name; `main` is what older builds installed."""
    return SubprocessStage(
        name="asr",
        binary="whisper-cli",
        alt_names=("whisper.cpp", "main"),
        argv=lambda wav: ["-m", model, "-t", str(threads), "-nt", "-f", str(wav)],
        source="https://github.com/ggerganov/whisper.cpp",
    )


def llama_prefill_stage(model: str = "models/model-q4_k_m.gguf", threads: int = 4) -> SubprocessStage:
    """Prompt ingestion. Separated from decode because the two scale differently:
    prefill is compute-bound in the prompt length, decode is memory-bandwidth-bound
    per token. On a Pi 3 they are the two ends of the budget and averaging them
    together hides which one to attack."""
    return SubprocessStage(
        name="llm_prefill",
        binary="llama-cli",
        alt_names=("main",),
        argv=lambda prompt: ["-m", model, "-t", str(threads), "-n", "0",
                             "-p", str(prompt)],
        source="https://github.com/ggerganov/llama.cpp",
    )


def llama_decode_stage(model: str = "models/model-q4_k_m.gguf", threads: int = 4,
                       n_predict: int = 64) -> SubprocessStage:
    """Token generation -- the tok/s the brief calls the headline number."""
    return SubprocessStage(
        name="llm_decode",
        binary="llama-cli",
        alt_names=("main",),
        argv=lambda prompt: ["-m", model, "-t", str(threads), "-n", str(n_predict),
                             "-p", str(prompt)],
        source="https://github.com/ggerganov/llama.cpp",
    )


def piper_stage(voice: str = "voices/en_US-lessac-low.onnx") -> SubprocessStage:
    """TTS."""
    return SubprocessStage(
        name="tts",
        binary="piper",
        argv=lambda text: ["-m", voice, "-f", "/tmp/assistant-out.wav"],
        source="https://github.com/rhasspy/piper",
    )


# -------------------------------------------------------------------- the simulator


@dataclass(frozen=True)
class LatencyModel:
    """Shifted lognormal with closed-form quantiles. See the module docstring."""

    median_ms: float
    sigma_log: float = 0.35
    floor_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.median_ms <= 0:
            raise ValueError(f"median_ms must be positive, got {self.median_ms}")
        if self.sigma_log < 0:
            raise ValueError(f"sigma_log must be non-negative, got {self.sigma_log}")

    def draw(self, rng: np.random.Generator, size: int | None = None):
        return self.floor_ms + self.median_ms * np.exp(self.sigma_log * rng.standard_normal(size))

    def quantile(self, z: float) -> float:
        """Exact quantile at standard-normal deviate `z`."""
        return self.floor_ms + self.median_ms * float(np.exp(self.sigma_log * z))

    @property
    def p50(self) -> float:
        return self.quantile(Z_P50)

    @property
    def p95(self) -> float:
        return self.quantile(Z_P95)


class SimulatedStage(Stage):
    """Draws a latency instead of measuring one. LABELLED SIMULATED EVERYWHERE."""

    simulated = True

    def __init__(
        self,
        name: str,
        model: LatencyModel,
        rng: np.random.Generator | None = None,
        seed: int = 0,
        sleep: bool = False,
        transform: Callable[[Any], Any] | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        # sleep=False by default: actually sleeping would make a 500-run benchmark
        # take half an hour to produce numbers that are drawn anyway. With sleep=True
        # the pipeline runs in something like real time, for a demo.
        self.sleep = sleep
        self.transform = transform

    def _execute(self, payload: Any) -> ExecOut:
        drawn = float(self.model.draw(self.rng))
        if self.sleep:
            time.sleep(drawn * 1e-3)
        out = self.transform(payload) if self.transform else payload
        return ExecOut(
            payload=out,
            latency_ms=drawn,
            meta={"SIMULATED": True, "model_median_ms": self.model.median_ms,
                  "model_sigma_log": self.model.sigma_log},
        )


# Placeholder distributions for the four stages. These are ARBITRARY -- they set the
# scale of the arithmetic so the harness can be exercised, and they are not predictions
# of Pi 3 performance. The brief is explicit that published figures for this board are
# scattered single blog data points, so nothing here is fitted to one. When hardware
# arrives, `bench/latency.py` runs against SubprocessStage and this dictionary becomes
# irrelevant.
PLACEHOLDER_MODELS: dict[str, LatencyModel] = {
    "asr": LatencyModel(median_ms=1800.0, sigma_log=0.30, floor_ms=120.0),
    "llm_prefill": LatencyModel(median_ms=900.0, sigma_log=0.25, floor_ms=60.0),
    "llm_decode": LatencyModel(median_ms=6000.0, sigma_log=0.35, floor_ms=200.0),
    "tts": LatencyModel(median_ms=700.0, sigma_log=0.20, floor_ms=40.0),
}


def simulated_pipeline(seed: int = 0, sleep: bool = False) -> dict[str, SimulatedStage]:
    """The four simulated stages, each with its own independent generator stream.

    Independent streams, not one shared generator: it makes each stage's sample
    sequence reproducible regardless of how many stages ran before it, so removing a
    stage from the pipeline does not silently change the others' numbers.
    """
    root = np.random.default_rng(seed)
    return {
        name: SimulatedStage(name, model, rng=np.random.default_rng(root.integers(2**32)),
                             sleep=sleep)
        for name, model in PLACEHOLDER_MODELS.items()
    }
