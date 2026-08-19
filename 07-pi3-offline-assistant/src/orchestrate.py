"""The pipeline: capture -> VAD -> ASR -> LLM prefill -> LLM decode -> TTS.

A state machine rather than a straight-line script, for one reason: when a stage fails
on a headless board at 3am, the useful artefact is a log that says which state the turn
died in and how long every earlier stage took. A script that raises loses all of it.

    IDLE -> CAPTURING -> DETECTING -+-> TRANSCRIBING -> PREFILL -> DECODING
                                    |                                 |
                                    +-> NO_SPEECH (a normal outcome)  v
                                                                   SPEAKING -> DONE
    any state -------------------------------------------------------> FAILED

`NO_SPEECH` is a terminal *success*: the VAD deciding that a 3-second recording of a
fridge is not a question is the detector working, and an assistant that treats it as an
error will restart itself all night.

Failure handling
----------------

`Stage.run` returns rather than raises, so `run_turn` inspects `ok` after every stage.
The first `ok=False` stops the pipeline: the state goes to FAILED, the failing stage
name and its error are recorded, and no later stage runs. Everything timed up to that
point is kept and logged. `tests/test_orchestrate.py` asserts exactly this -- that a
failure propagates and is reported, rather than the pipeline continuing on a `None`
payload and producing a plausible-looking turn out of nothing.

What end-to-end latency means here
----------------------------------

`end_to_end_ms` is the sum of the stages' own reported latencies. When any stage is a
`SimulatedStage` that sum contains drawn numbers, so `TurnResult.source` reports
`SIMULATED` for the whole turn and every log line and CSV row inherits it. A turn is
only `measured` if every one of its stages measured itself.

`wall_ms` is separately the real wall clock of `run_turn`. With simulated stages and
`sleep=False` it is microseconds while `end_to_end_ms` is seconds; the gap is the
point, and printing both makes it impossible to mistake one for the other.
"""

from __future__ import annotations

import argparse
import enum
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src import audio, vad
from src.stages import (ExecOut, Stage, StageResult, SubprocessStage,
                        simulated_pipeline)


class TurnState(enum.Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    DETECTING = "detecting"
    NO_SPEECH = "no_speech"
    TRANSCRIBING = "transcribing"
    PREFILL = "prefill"
    DECODING = "decoding"
    SPEAKING = "speaking"
    DONE = "done"
    FAILED = "failed"


TERMINAL = {TurnState.DONE, TurnState.FAILED, TurnState.NO_SPEECH}


# --------------------------------------------------------------------- real stages


class WavCaptureStage(Stage):
    """Reads a WAV off disk. Genuinely measured -- file I/O and decode, no simulation."""

    simulated = False

    def __init__(self, path: str | pathlib.Path, name: str = "capture"):
        super().__init__(name)
        self.path = pathlib.Path(path)

    def _execute(self, payload: Any) -> ExecOut:
        if not self.path.exists():
            raise FileNotFoundError(
                f"stage '{self.name}': no such WAV: {self.path}. Generate one with "
                "`python3 -m src.orchestrate --write-wav`, or point --wav at a "
                "16-bit PCM file."
            )
        x, fs = audio.read_wav(self.path)
        return ExecOut(payload=(x, fs), meta={"path": str(self.path), "fs": fs,
                                              "n_samples": int(x.size)})


class SyntheticCaptureStage(Stage):
    """Generates a labelled utterance in memory. Real computation, measured honestly;
    the *audio* is synthetic, which is a different claim from the *timing* being
    simulated, and the two are not conflated anywhere."""

    simulated = False

    def __init__(self, seed: int = 0, snr_db: float = 20.0, name: str = "capture"):
        super().__init__(name)
        self.seed = seed
        self.snr_db = snr_db

    def _execute(self, payload: Any) -> ExecOut:
        from src.synth import make_utterance
        sig = make_utterance(seed=self.seed, snr_db=self.snr_db)
        return ExecOut(payload=(sig.x, sig.fs),
                       meta={"synthetic_audio": True, "snr_db": self.snr_db,
                             "n_true_segments": len(sig.segments)})


def arecord_stage(device: str = "plughw:1,0", seconds: float = 5.0,
                  out: str = "/tmp/assistant-in.wav") -> SubprocessStage:
    """Capture from the USB microphone on the board. Same absent-binary contract as
    the inference stages: `arecord` is not on this host either."""
    return SubprocessStage(
        name="capture",
        binary="arecord",
        argv=["-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1",
              "-d", str(int(seconds)), out],
        source="alsa-utils, `apt install alsa-utils` on Raspberry Pi OS",
        parse=lambda proc: out,
    )


class VadStage(Stage):
    """Voice activity detection. Real signal processing, real measured latency.

    This is the only inference-adjacent stage in the pipeline that runs for real on
    every machine, which is why it is the stage this repo's test suite is built around.
    """

    simulated = False

    def __init__(self, cfg: vad.VadConfig | None = None, name: str = "vad"):
        super().__init__(name)
        self.cfg = cfg or vad.VadConfig()

    def _execute(self, payload: Any) -> ExecOut:
        x, fs = payload
        result = vad.detect(x, fs, self.cfg)
        speech = vad.extract_speech(x, fs, result)
        return ExecOut(
            payload=(speech, fs, result),
            meta={"n_segments": len(result.segments),
                  "speech_s": round(float(speech.size / fs), 4),
                  "noise_floor_db": round(result.noise_floor_db, 2),
                  "speech_fraction": round(result.speech_fraction, 4)},
        )


# ------------------------------------------------------------------------ the turn


@dataclass
class TurnResult:
    turn: int
    ok: bool
    final_state: TurnState
    stages: list[StageResult] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    error: str | None = None
    wall_ms: float = 0.0

    @property
    def end_to_end_ms(self) -> float:
        """Sum of the stages' own reported latencies."""
        return float(sum(s.latency_ms for s in self.stages))

    @property
    def contains_simulated(self) -> bool:
        return any(s.simulated for s in self.stages)

    @property
    def source(self) -> str:
        return "SIMULATED" if self.contains_simulated else "measured"

    def stage_latency(self, name: str) -> float | None:
        for s in self.stages:
            if s.stage == name:
                return s.latency_ms
        return None

    def to_records(self) -> list[dict]:
        base = {"turn": self.turn, "turn_source": self.source}
        return [{**base, **s.to_record()} for s in self.stages]

    def summary(self) -> dict:
        return {
            "turn": self.turn,
            "ok": self.ok,
            "final_state": self.final_state.value,
            "source": self.source,
            "end_to_end_ms": round(self.end_to_end_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "failed_stage": self.failed_stage,
            "error": self.error,
            "transitions": " -> ".join(self.transitions),
        }


class JsonlLogger:
    """One JSON object per line. Greppable on a board with no log aggregator."""

    def __init__(self, path: str | pathlib.Path | None = None, stream=None):
        self.path = pathlib.Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        self.stream = stream
        self.records: list[dict] = []

    def __call__(self, record: dict) -> None:
        record = {"ts": round(time.time(), 3), **record}
        self.records.append(record)
        line = json.dumps(record, default=str)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(line + "\n")
        if self.stream:
            print(line, file=self.stream)


class Assistant:
    """capture -> VAD -> ASR -> prefill -> decode -> TTS, with a state machine."""

    def __init__(
        self,
        capture: Stage,
        asr: Stage,
        prefill: Stage,
        decode: Stage,
        tts: Stage,
        vad_cfg: vad.VadConfig | None = None,
        logger: Callable[[dict], None] | None = None,
        prompt_template: str = "User: {text}\nAssistant:",
    ):
        self.capture = capture
        self.vad_stage = VadStage(vad_cfg)
        self.asr = asr
        self.prefill = prefill
        self.decode = decode
        self.tts = tts
        self.logger = logger
        self.prompt_template = prompt_template
        self.state = TurnState.IDLE
        self._turn = 0

    # -- state machine bookkeeping ------------------------------------------------

    def _goto(self, state: TurnState, transitions: list[str]) -> None:
        transitions.append(state.value)
        self.state = state

    def _log(self, result: StageResult, state: TurnState) -> None:
        if self.logger:
            self.logger({"turn": self._turn, "state": state.value, **result.to_record()})

    def _step(self, stage: Stage, payload: Any, state: TurnState,
              turn: TurnResult) -> StageResult:
        self._goto(state, turn.transitions)
        result = stage.run(payload)
        turn.stages.append(result)
        self._log(result, state)
        return result

    # -- the turn -----------------------------------------------------------------

    def run_turn(self) -> TurnResult:
        self._turn += 1
        turn = TurnResult(turn=self._turn, ok=False, final_state=TurnState.IDLE)
        t0 = time.perf_counter()
        self.state = TurnState.IDLE
        turn.transitions.append(self.state.value)

        def fail(res: StageResult) -> TurnResult:
            self._goto(TurnState.FAILED, turn.transitions)
            turn.ok = False
            turn.final_state = TurnState.FAILED
            turn.failed_stage = res.stage
            turn.error = res.error
            turn.wall_ms = (time.perf_counter() - t0) * 1e3
            if self.logger:
                self.logger({"turn": self._turn, "event": "turn_failed",
                             "stage": res.stage, "error": res.error})
            return turn

        cap = self._step(self.capture, None, TurnState.CAPTURING, turn)
        if not cap.ok:
            return fail(cap)

        det = self._step(self.vad_stage, cap.payload, TurnState.DETECTING, turn)
        if not det.ok:
            return fail(det)

        speech, fs, vad_result = det.payload
        if not vad_result.segments:
            # Not an error. The board heard the room, not a question.
            self._goto(TurnState.NO_SPEECH, turn.transitions)
            turn.ok = True
            turn.final_state = TurnState.NO_SPEECH
            turn.wall_ms = (time.perf_counter() - t0) * 1e3
            if self.logger:
                self.logger({"turn": self._turn, "event": "no_speech",
                             "noise_floor_db": round(vad_result.noise_floor_db, 2)})
            return turn

        asr = self._step(self.asr, speech, TurnState.TRANSCRIBING, turn)
        if not asr.ok:
            return fail(asr)

        prompt = self.prompt_template.format(text=asr.payload)
        pre = self._step(self.prefill, prompt, TurnState.PREFILL, turn)
        if not pre.ok:
            return fail(pre)

        dec = self._step(self.decode, pre.payload, TurnState.DECODING, turn)
        if not dec.ok:
            return fail(dec)

        tts = self._step(self.tts, dec.payload, TurnState.SPEAKING, turn)
        if not tts.ok:
            return fail(tts)

        self._goto(TurnState.DONE, turn.transitions)
        turn.ok = True
        turn.final_state = TurnState.DONE
        turn.wall_ms = (time.perf_counter() - t0) * 1e3
        if self.logger:
            self.logger({"turn": self._turn, "event": "turn_done", **turn.summary()})
        return turn


def build_simulated_assistant(
    capture: Stage | None = None,
    seed: int = 0,
    sleep: bool = False,
    vad_cfg: vad.VadConfig | None = None,
    logger: Callable[[dict], None] | None = None,
) -> Assistant:
    """Real capture + real VAD + SIMULATED ASR/LLM/TTS.

    The only configuration that runs end to end on this machine. Anything downstream
    of the VAD is drawn, not measured, and says so in every record it emits.
    """
    sim = simulated_pipeline(seed=seed, sleep=sleep)
    for name, stage in sim.items():
        stage.transform = _PAYLOAD_TRANSFORMS[name]
    return Assistant(
        capture=capture or SyntheticCaptureStage(seed=seed),
        asr=sim["asr"], prefill=sim["llm_prefill"], decode=sim["llm_decode"],
        tts=sim["tts"], vad_cfg=vad_cfg, logger=logger,
    )


# The simulated stages have to hand something down the pipe. These placeholders are
# obviously not model output and are not meant to be mistaken for it.
_PAYLOAD_TRANSFORMS = {
    "asr": lambda speech: "[SIMULATED TRANSCRIPT]",
    "llm_prefill": lambda prompt: {"SIMULATED": True, "prompt_chars": len(str(prompt))},
    "llm_decode": lambda kv: "[SIMULATED RESPONSE]",
    "tts": lambda text: "[SIMULATED AUDIO]",
}


def build_real_assistant(model: str, whisper_model: str, voice: str,
                         device: str = "plughw:1,0",
                         logger: Callable[[dict], None] | None = None) -> Assistant:
    """What runs on the board. Every stage raises MissingBinaryError on this host."""
    from src.stages import (llama_decode_stage, llama_prefill_stage, piper_stage,
                            whisper_stage)
    return Assistant(
        capture=arecord_stage(device=device),
        asr=whisper_stage(model=whisper_model),
        prefill=llama_prefill_stage(model=model),
        decode=llama_decode_stage(model=model),
        tts=piper_stage(voice=voice),
        logger=logger,
    )


# ------------------------------------------------------------------------- the demo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run one assistant turn. Capture and VAD are real; ASR/LLM/TTS "
                    "are SIMULATED on any machine without the inference binaries.")
    ap.add_argument("--wav", default=None, help="16-bit PCM WAV to run on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr", type=float, default=20.0, help="synthetic capture SNR, dB")
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--sleep", action="store_true",
                    help="make simulated stages actually sleep, for a real-time demo")
    ap.add_argument("--real", action="store_true",
                    help="use the subprocess stages; fails naming the missing binary")
    ap.add_argument("--log", default="results/turn_log.jsonl")
    ap.add_argument("--write-wav", default="data/demo_utterance.wav",
                    help="where to write the synthesised capture")
    args = ap.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[1]
    logger = JsonlLogger(root / args.log, stream=sys.stdout)

    if args.real:
        assistant = build_real_assistant(model="models/model-q4_k_m.gguf",
                                         whisper_model="models/ggml-tiny.en-q5_1.bin",
                                         voice="voices/en_US-lessac-low.onnx",
                                         logger=logger)
    else:
        if args.wav:
            capture: Stage = WavCaptureStage(args.wav)
        else:
            from src.synth import make_utterance
            sig = make_utterance(seed=args.seed, snr_db=args.snr)
            wav_path = root / args.write_wav
            audio.write_wav(wav_path, sig.x, sig.fs)
            print(f"# wrote synthetic capture: {wav_path} "
                  f"({sig.duration_s:.2f} s, {len(sig.segments)} true segments, "
                  f"SNR {sig.snr_db:g} dB)")
            capture = WavCaptureStage(wav_path)
        assistant = build_simulated_assistant(capture=capture, seed=args.seed,
                                              sleep=args.sleep, logger=logger)

    rc = 0
    for _ in range(args.turns):
        turn = assistant.run_turn()
        print("# " + json.dumps(turn.summary()))
        if not turn.ok:
            rc = 1
    if not args.real:
        print("# NOTE: ASR / LLM prefill / LLM decode / TTS latencies above are "
              "SIMULATED draws, not measurements. See STATUS.md.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
