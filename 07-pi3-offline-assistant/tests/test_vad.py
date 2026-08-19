"""The signal-processing half of this project's anchor.

READ THIS BEFORE THE ASSERTIONS.

This repo is the portfolio's deployment capstone, and its advertised deliverable is a
latency budget measured on a Raspberry Pi 3. There is no Raspberry Pi. So the claim
this test suite defends is NOT the deployment claim -- that one is unmeasured and is
marked unmeasured everywhere in this repo. The anchor here is the part that is real:
the signal processing in `src/vad.py`, and the statistics in `bench/latency.py`
(asserted in `tests/test_latency.py`).

Those two are testable to a standard the deployment claim cannot currently be held to,
so they carry the assertions. Everything downstream of the VAD is either a subprocess
call to a binary that does not exist on this host, or an explicitly labelled draw.
"""

import unittest

import numpy as np

from src.audio import labels_from_spans
from src.synth import make_blip, make_plateau, make_utterance
from src.vad import (VadConfig, boundary_errors_ms, detect, frame_metrics,
                     hysteresis_segments, robust_floor, zero_crossing_rate)

SNRS = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]
SEEDS = range(8)


def score(snr_db: float, cfg: VadConfig, seeds=SEEDS) -> dict:
    """Average per-utterance detector scores over seeds at one SNR.

    Ground truth is converted to frame labels with the SAME framing helper the
    detector uses, so nothing in the score comes from a units mismatch between
    frame-start and frame-centre conventions.
    """
    acc: dict[str, list[float]] = {}
    for seed in seeds:
        sig = make_utterance(seed=seed, snr_db=snr_db)
        res = detect(sig.x, sig.fs, cfg)
        n = res.frame_labels.size
        truth = labels_from_spans(sig.segments, n, res.win, res.hop, sig.fs)
        uv = labels_from_spans(sig.unvoiced_spans, n, res.win, res.hop, sig.fs)
        fm = frame_metrics(res.frame_labels, truth)
        be = boundary_errors_ms(res.segments, sig.segments)
        for key, value in (("precision", fm["precision"]), ("recall", fm["recall"]),
                           ("f1", fm["f1"]), ("match_rate", be["match_rate"])):
            acc.setdefault(key, []).append(value)
        for key in ("start_mae_ms", "end_mae_ms"):
            if not np.isnan(be[key]):
                acc.setdefault(key, []).append(float(be[key]))
        acc.setdefault("unvoiced_recall", []).append(
            float((res.frame_labels & uv).sum()) / max(1, int(uv.sum())))
    return {k: float(np.mean(v)) for k, v in acc.items()}


class TestVadAccuracy(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # The detector must recover speech boundaries it was never told about, from
    # signals whose boundaries are exact by construction, at several SNRs -- and its
    # accuracy must fall off monotonically as the noise rises. A detector that always
    # said "speech" would pass a shape assertion and fail every line below: it would
    # score recall 1.0 with precision ~0.4, boundary errors of hundreds of
    # milliseconds, and an accuracy curve flat in SNR.
    #
    # The monotonicity requirement is the part that cannot be faked by tuning. Any
    # threshold set can be made to look good at one SNR. Only a detector that is
    # actually measuring signal-to-noise degrades in the right order.
    def test_recovers_known_segments_and_degrades_monotonically_with_snr(self):
        cfg = VadConfig()
        scores = {snr: score(snr, cfg) for snr in SNRS}

        # 1. Clean-ish speech: near-perfect frame labels and tight boundaries.
        clean = scores[20.0]
        self.assertGreaterEqual(clean["f1"], 0.90, f"F1 at 20 dB SNR: {clean}")
        self.assertGreaterEqual(clean["precision"], 0.95, f"precision at 20 dB: {clean}")
        self.assertGreaterEqual(clean["recall"], 0.85, f"recall at 20 dB: {clean}")
        # Every ground-truth word must get a prediction. Without this, a low boundary
        # error can just mean the detector found one easy word and ignored the rest.
        self.assertEqual(clean["match_rate"], 1.0,
                         f"missed whole words at 20 dB SNR: {clean}")
        # 40 ms is four hops. The detector cannot in principle do better than one hop
        # (10 ms), and a 25 ms window smears an onset over more than one frame.
        self.assertLess(clean["start_mae_ms"], 40.0, f"onset error at 20 dB: {clean}")
        self.assertLess(clean["end_mae_ms"], 40.0, f"offset error at 20 dB: {clean}")

        # 2. Very clean speech: everything found.
        self.assertGreaterEqual(scores[30.0]["recall"], 0.98)
        self.assertGreaterEqual(scores[30.0]["f1"], 0.95)

        # 3. Monotone degradation. Tolerance 0.02 rather than 0: above ~25 dB SNR F1
        # saturates and the residual error is boundary quantisation, not noise, so the
        # curve is flat there and wobbles by a fraction of a percent between adjacent
        # points. It is 0.0097 between 30 and 25 dB as measured. Below 25 dB the
        # decline is large and strict, and that is asserted separately.
        f1 = [scores[s]["f1"] for s in SNRS]
        for a, b, sa, sb in zip(f1, f1[1:], SNRS, SNRS[1:]):
            self.assertGreaterEqual(
                a + 0.02, b,
                f"F1 rose from {a:.4f} at {sa} dB to {b:.4f} at {sb} dB: accuracy is "
                "not tracking SNR")
        strict = [scores[s]["f1"] for s in SNRS if s <= 25.0]
        for a, b in zip(strict, strict[1:]):
            self.assertGreater(a, b, f"F1 did not strictly fall below 25 dB: {strict}")

        # 4. The degradation is real, not a rounding wobble: from near-perfect to
        # nothing across the range tested.
        self.assertGreater(f1[0] - f1[-1], 0.8, f"F1 across the SNR range: {f1}")

        # 5. HOW it degrades is also a claim. An energy detector with a robust floor
        # should fail by missing quiet speech, not by hallucinating speech in noise.
        # Precision therefore has to hold up while recall collapses.
        mid = scores[10.0]
        self.assertGreater(mid["precision"], 0.90,
                           f"false-alarm rate blew up at 10 dB SNR: {mid}")
        self.assertLess(mid["recall"], 0.80, f"recall at 10 dB looks too good: {mid}")
        self.assertGreater(mid["recall"], 0.30, f"recall at 10 dB collapsed: {mid}")

        # 6. And at 0 dB SNR, where the speech RMS equals the noise RMS, it should be
        # honestly beaten rather than reporting something.
        self.assertLess(scores[0.0]["f1"], 0.2, f"F1 at 0 dB SNR: {scores[0.0]}")

    def test_zcr_extension_recovers_unvoiced_edges(self):
        """The second feature has to earn its place in the code.

        Unvoiced edges sit ~5 dB over the floor, below the onset threshold, so the
        energy channel alone truncates every word. If disabling the ZCR extension did
        not measurably hurt, the extension would be deleted rather than kept for
        decoration. Measured at 20 dB SNR: unvoiced-frame recall 0.86 with it against
        0.48 without, and onset error 20 ms against 78 ms.
        """
        full = score(20.0, VadConfig(zcr_extend=True))
        energy_only = score(20.0, VadConfig(zcr_extend=False))
        self.assertGreater(
            full["unvoiced_recall"], energy_only["unvoiced_recall"] + 0.20,
            f"ZCR extension bought nothing: {full['unvoiced_recall']:.3f} vs "
            f"{energy_only['unvoiced_recall']:.3f}")
        self.assertLess(
            full["start_mae_ms"], 0.6 * energy_only["start_mae_ms"],
            f"onset error {full['start_mae_ms']:.1f} ms vs "
            f"{energy_only['start_mae_ms']:.1f} ms without ZCR")
        self.assertGreater(full["f1"], energy_only["f1"])


class TestHysteresisAndSmoothing(unittest.TestCase):

    def test_hysteresis_prevents_chatter(self):
        """A signal sitting on one threshold must produce one segment, not dozens.

        Both branches run on the SAME energy array, so the comparison is purely
        one threshold against two. Absolute thresholds are used rather than the
        adaptive floor, otherwise the threshold would be a function of the signal and
        the test would be circular.
        """
        x = make_plateau(seed=0, level_db=-30.0, jitter_db=3.0)
        cfg = VadConfig(abs_onset_db=-30.0, abs_offset_db=-36.0)
        res = detect(x, 16000, cfg)

        # Same energies, one threshold, no hysteresis, no hangover.
        chattering = hysteresis_segments(res.energy_db, -30.0, -30.0, 1, 1)

        self.assertGreater(len(chattering), 8,
                           "the plateau signal is not actually straddling the "
                           "threshold; the test instrument is broken, not the VAD")
        self.assertEqual(len(res.segments), 1,
                         f"hysteresis produced {len(res.segments)} segments where it "
                         f"should produce 1 (single threshold gives "
                         f"{len(chattering)})")

    def test_minimum_duration_removes_blips(self):
        """A 60 ms burst is shorter than any word and must be discarded -- and the
        detector must say it discarded it, not merely fail to report it."""
        x = make_blip(seed=0, blip_ms=60.0)
        strict = VadConfig(abs_onset_db=-25.0, abs_offset_db=-40.0,
                           min_speech_ms=120.0, onset_frames=2, hangover_frames=3,
                           zcr_extend=False)
        lenient = VadConfig(abs_onset_db=-25.0, abs_offset_db=-40.0,
                            min_speech_ms=20.0, onset_frames=2, hangover_frames=3,
                            zcr_extend=False)
        dropped = detect(x, 16000, strict)
        kept = detect(x, 16000, lenient)
        self.assertEqual(len(dropped.segments), 0)
        self.assertGreaterEqual(dropped.dropped_segments, 1,
                                "the blip was never detected at all, so the minimum-"
                                "duration rule is untested by this signal")
        self.assertEqual(len(kept.segments), 1,
                         "the same blip must survive a 20 ms minimum, or the two "
                         "branches differ by something other than the duration rule")

    def test_offset_above_onset_is_rejected(self):
        with self.assertRaises(ValueError):
            hysteresis_segments(np.zeros(10), onset=-30.0, offset=-20.0,
                                onset_frames=1, hangover_frames=1)

    def test_segment_end_is_not_inflated_by_the_hangover(self):
        """The hangover buys robustness; it must not add 100 ms to every duration."""
        e = np.array([-60.0] * 20 + [-10.0] * 30 + [-60.0] * 40)
        segs = hysteresis_segments(e, onset=-30.0, offset=-40.0,
                                   onset_frames=3, hangover_frames=10)
        self.assertEqual(segs, [(20, 50)])


class TestFeatures(unittest.TestCase):

    def test_zero_crossing_rate_of_known_signals(self):
        fs = 16000
        n = 400
        t = np.arange(n) / fs
        # A sine at f crosses zero 2f times per second; over n samples at rate fs the
        # crossing fraction is 2f/fs.
        for f in (100.0, 500.0, 2000.0):
            frame = np.sin(2 * np.pi * f * t)[None, :]
            expected = 2 * f / fs
            self.assertAlmostEqual(float(zero_crossing_rate(frame)[0]), expected,
                                   delta=2.0 / (n - 1))
        # Digital silence contains no sign changes -- signbit(0.0) is False for every
        # sample, which is the reason signbit is used instead of sign.
        self.assertEqual(float(zero_crossing_rate(np.zeros((1, 64)))[0]), 0.0)

    def test_robust_floor_ignores_a_minority_of_speech(self):
        """The floor estimator uses a median so an early talker cannot poison it."""
        noise = np.full(40, -50.0)
        contaminated = noise.copy()
        contaminated[:15] = -5.0  # 37% of the window is loud speech
        clean_med, _ = robust_floor(noise)
        dirty_med, _ = robust_floor(contaminated)
        self.assertEqual(clean_med, dirty_med)
        self.assertGreater(float(np.mean(contaminated)) - dirty_med, 15.0,
                           "the mean would have been dragged up; that is the point")

    def test_short_signal_fails_with_a_useful_message(self):
        with self.assertRaises(ValueError) as ctx:
            detect(np.zeros(100), 16000, VadConfig())
        self.assertIn("frame", str(ctx.exception))

    def test_mismatched_label_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            frame_metrics(np.zeros(5, bool), np.zeros(7, bool))


if __name__ == "__main__":
    unittest.main()
