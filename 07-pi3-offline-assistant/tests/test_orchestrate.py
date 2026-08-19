"""The state machine, and the two failure modes that matter on a headless board.

A pipeline that continues past a dead stage produces a turn that looks fine and is
made of nothing. A stub that fails without naming what is missing costs an hour of
someone's evening. Both are asserted here.
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np

from src import audio
from src.orchestrate import (Assistant, JsonlLogger, SyntheticCaptureStage, TurnState,
                             VadStage, WavCaptureStage, build_simulated_assistant)
from src.stages import (ExecOut, MissingBinaryError, Stage, StageFailure,
                        llama_decode_stage, piper_stage, whisper_stage)
from src.synth import make_utterance
from src.vad import VadConfig


class ExplodingStage(Stage):
    """A stage that always fails, with a message we can look for downstream."""

    simulated = False

    def __init__(self, name="llm_decode", message="the GGUF is 700 MB and RAM is 1 GB"):
        super().__init__(name)
        self.message = message

    def _execute(self, payload):
        raise StageFailure(self.message)


def _assistant_with(failing: Stage) -> Assistant:
    sim = build_simulated_assistant(capture=SyntheticCaptureStage(seed=0, snr_db=20.0))
    setattr(sim, {"asr": "asr", "llm_prefill": "prefill", "llm_decode": "decode",
                  "tts": "tts"}[failing.name], failing)
    return sim


class TestFailurePropagation(unittest.TestCase):

    def test_a_failing_stage_stops_the_turn_and_is_named(self):
        """The pipeline must NOT continue on a None payload and produce a plausible
        turn out of nothing. It must stop, say which stage died, and keep the timings
        it already collected."""
        assistant = _assistant_with(ExplodingStage("llm_decode"))
        turn = assistant.run_turn()

        self.assertFalse(turn.ok)
        self.assertEqual(turn.final_state, TurnState.FAILED)
        self.assertEqual(turn.failed_stage, "llm_decode")
        self.assertIn("700 MB", turn.error)
        self.assertIn("StageFailure", turn.error)

        ran = [s.stage for s in turn.stages]
        # Everything up to and including the failure ran...
        self.assertEqual(ran, ["capture", "vad", "asr", "llm_prefill", "llm_decode"])
        # ...and nothing after it did.
        self.assertNotIn("tts", ran)
        # The timings collected before the failure survive rather than being lost with
        # the stack -- that is the reason `Stage.run` returns instead of raising.
        self.assertGreater(turn.stage_latency("vad"), 0.0)
        self.assertEqual(turn.transitions[-1], "failed")

    def test_an_early_failure_stops_everything_after_it(self):
        assistant = _assistant_with(ExplodingStage("asr"))
        turn = assistant.run_turn()
        self.assertEqual([s.stage for s in turn.stages], ["capture", "vad", "asr"])
        self.assertEqual(turn.failed_stage, "asr")

    def test_a_missing_capture_file_is_reported_not_swallowed(self):
        assistant = build_simulated_assistant(
            capture=WavCaptureStage("/nonexistent/definitely-not-here.wav"))
        turn = assistant.run_turn()
        self.assertFalse(turn.ok)
        self.assertEqual(turn.failed_stage, "capture")
        self.assertIn("definitely-not-here.wav", turn.error)

    def test_silence_is_a_successful_turn_not_a_failure(self):
        """The VAD deciding a recording of a fridge is not a question is the detector
        working. An assistant that treats it as an error restarts itself all night."""
        rng = np.random.default_rng(0)
        quiet = 1e-4 * rng.standard_normal(16000 * 3)

        class QuietCapture(Stage):
            simulated = False

            def _execute(self, payload):
                return ExecOut(payload=(quiet, 16000))

        assistant = build_simulated_assistant(capture=QuietCapture("capture"))
        turn = assistant.run_turn()
        self.assertTrue(turn.ok)
        self.assertEqual(turn.final_state, TurnState.NO_SPEECH)
        self.assertEqual([s.stage for s in turn.stages], ["capture", "vad"])


class TestMissingBinaries(unittest.TestCase):

    def test_subprocess_stages_name_the_missing_binary_and_where_to_get_it(self):
        for stage, binary, upstream in (
            (whisper_stage(), "whisper-cli", "whisper.cpp"),
            (llama_decode_stage(), "llama-cli", "llama.cpp"),
            (piper_stage(), "piper", "piper"),
        ):
            with self.subTest(binary=binary):
                result = stage.run("payload")
                self.assertFalse(result.ok)
                self.assertIn("MissingBinaryError", result.error)
                self.assertIn(binary, result.error)
                self.assertIn(upstream, result.error)
                self.assertIn("setup/install.sh", result.error)

    def test_missing_binary_error_is_a_not_implemented_error(self):
        """The portfolio convention is that a stub raises NotImplementedError. This
        one does, while still being catchable as its own type."""
        self.assertTrue(issubclass(MissingBinaryError, NotImplementedError))
        with self.assertRaises(MissingBinaryError):
            whisper_stage()._execute("x")

    def test_a_present_binary_is_actually_invoked(self):
        """Not a mock: `true` exists on every POSIX host, so this exercises the real
        subprocess path and proves the failure above is about absence, not about the
        code never getting that far."""
        from src.stages import SubprocessStage
        stage = SubprocessStage(name="probe", binary="true", argv=[],
                                source="coreutils")
        result = stage.run(None)
        self.assertTrue(result.ok, result.error)
        self.assertFalse(result.simulated)
        self.assertGreater(result.latency_ms, 0.0)

    def test_a_nonzero_exit_is_reported_with_its_status(self):
        from src.stages import SubprocessStage
        stage = SubprocessStage(name="probe", binary="false", argv=[],
                                source="coreutils")
        result = stage.run(None)
        self.assertFalse(result.ok)
        self.assertIn("StageFailure", result.error)
        self.assertIn("exited 1", result.error)


class TestStructuredLogging(unittest.TestCase):

    def test_every_stage_emits_one_valid_json_line_carrying_its_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "turn.jsonl"
            logger = JsonlLogger(path)
            assistant = build_simulated_assistant(
                capture=SyntheticCaptureStage(seed=0), logger=logger)
            turn = assistant.run_turn()
            self.assertTrue(turn.ok)

            lines = [json.loads(line) for line in path.read_text().splitlines()]
            stage_lines = [r for r in lines if "stage" in r and "state" in r]
            self.assertEqual([r["stage"] for r in stage_lines],
                             ["capture", "vad", "asr", "llm_prefill", "llm_decode",
                              "tts"])
            for record in stage_lines:
                self.assertIn(record["source"], ("SIMULATED", "measured"))
                self.assertIn("latency_ms", record)
                self.assertIn("ts", record)
            sources = {r["stage"]: r["source"] for r in stage_lines}
            self.assertEqual(sources["capture"], "measured")
            self.assertEqual(sources["vad"], "measured")
            for name in ("asr", "llm_prefill", "llm_decode", "tts"):
                self.assertEqual(sources[name], "SIMULATED")

    def test_a_turn_containing_any_simulated_stage_is_a_simulated_turn(self):
        """End-to-end latency built partly from draws is not a measurement, and the
        label has to survive aggregation or it is worthless."""
        assistant = build_simulated_assistant(capture=SyntheticCaptureStage(seed=1))
        turn = assistant.run_turn()
        self.assertTrue(turn.contains_simulated)
        self.assertEqual(turn.source, "SIMULATED")
        for record in turn.to_records():
            self.assertEqual(record["turn_source"], "SIMULATED")

    def test_wall_clock_and_reported_end_to_end_are_kept_apart(self):
        """With sleep=False the drawn total is seconds and the real elapsed time is
        milliseconds. If these were ever the same field, the repo would be reporting
        a simulation as a measurement."""
        assistant = build_simulated_assistant(capture=SyntheticCaptureStage(seed=2))
        turn = assistant.run_turn()
        self.assertGreater(turn.end_to_end_ms, 1000.0)
        self.assertLess(turn.wall_ms, 500.0)


class TestPipelineOnRealAudio(unittest.TestCase):

    def test_the_vad_stage_finds_the_words_it_was_given_through_the_pipeline(self):
        """The one part of the pipeline that is genuinely doing its job end to end:
        capture a WAV off disk, run the detector, and get the right number of words."""
        sig = make_utterance(seed=5, snr_db=20.0, n_words=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "utterance.wav"
            audio.write_wav(path, sig.x, sig.fs)
            stage = VadStage(VadConfig())
            captured = WavCaptureStage(path).run(None)
            self.assertTrue(captured.ok, captured.error)
            result = stage.run(captured.payload)
        self.assertTrue(result.ok, result.error)
        speech, fs, vad_result = result.payload
        self.assertEqual(len(vad_result.segments), len(sig.segments))
        self.assertFalse(result.simulated)
        # The extracted audio must be a real subset of the input, not empty padding.
        self.assertGreater(speech.size, 0)
        self.assertLess(speech.size, sig.x.size)


if __name__ == "__main__":
    unittest.main()
