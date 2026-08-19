"""WAV I/O and the framing convention.

Not the anchor test -- see `tests/test_vad.py` for that -- but everything the anchor
measures is expressed in these units, so an error here would move every boundary
number in the repo by a constant and nothing else would notice.
"""

import pathlib
import tempfile
import unittest

import numpy as np

from src.audio import (INT16_MAX, INT16_MIN, float_to_int16, frame_signal,
                       frame_span_s, int16_to_float, labels_from_spans, ms_to_samples,
                       n_frames, read_wav, segment_to_seconds, write_wav)


class TestScalingRoundTrip(unittest.TestCase):

    def test_int16_round_trip_is_bit_exact_over_the_whole_range(self):
        """Every one of the 65536 representable samples must survive unchanged.

        This holds only because both directions use 32768: a scale by 2^-15 and then
        by 2^15 is exact in binary floating point. The common shortcut of dividing by
        32768 and multiplying by 32767 fails this test on most of the range, by one
        LSB, which is inaudible once and a slow drift after repeated resampling.
        """
        every = np.arange(INT16_MIN, INT16_MAX + 1, dtype=np.int16)
        self.assertTrue(np.array_equal(float_to_int16(int16_to_float(every)), every))

    def test_float_round_trip_is_bounded_by_half_an_lsb(self):
        f = np.linspace(-0.9999, 0.9999, 20001)
        err = np.abs(int16_to_float(float_to_int16(f)) - f)
        self.assertLessEqual(float(err.max()), 1.0 / 32768.0 + 1e-15)

    def test_clipping_saturates_rather_than_wrapping(self):
        out = float_to_int16(np.array([-2.0, -1.0, 1.0, 2.0]))
        self.assertTrue(np.array_equal(out, np.array([INT16_MIN, INT16_MIN,
                                                      INT16_MAX, INT16_MAX],
                                                     dtype=np.int16)))


class TestWavIO(unittest.TestCase):

    def test_wav_round_trip_through_disk_is_lossless_in_int16(self):
        rng = np.random.default_rng(0)
        pcm = rng.integers(INT16_MIN, INT16_MAX + 1, size=4096).astype(np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rt.wav"
            write_wav(path, pcm, 16000)
            back, fs = read_wav(path)
            self.assertEqual(fs, 16000)
            self.assertTrue(np.array_equal(float_to_int16(back), pcm))

    def test_wav_round_trip_of_a_float_signal_is_within_one_lsb(self):
        rng = np.random.default_rng(1)
        x = 0.8 * rng.standard_normal(8000).clip(-1.0, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "rt.wav"
            write_wav(path, x, 16000)
            back, _ = read_wav(path)
            self.assertLessEqual(float(np.abs(back - x).max()), 1.0 / 32768.0 + 1e-15)

    def test_non_16_bit_input_is_rejected_by_name(self):
        import wave
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "eight.wav"
            with wave.open(str(path), "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(1)
                fh.setframerate(16000)
                fh.writeframes(b"\x80" * 100)
            with self.assertRaises(ValueError) as ctx:
                read_wav(path)
            self.assertIn("16-bit", str(ctx.exception))


class TestFramingConvention(unittest.TestCase):

    def test_frame_count_and_placement(self):
        # 100 samples, window 25, hop 10 -> frames at 0,10,...,70 = 8 frames; the
        # partial tail at 80 is dropped rather than zero-padded.
        self.assertEqual(n_frames(100, 25, 10), 8)
        self.assertEqual(n_frames(24, 25, 10), 0)
        frames = frame_signal(np.arange(100.0), 25, 10)
        self.assertEqual(frames.shape, (8, 25))
        self.assertEqual(frames[0, 0], 0.0)
        self.assertEqual(frames[3, 0], 30.0)
        self.assertEqual(frames[-1, -1], 94.0)

    def test_frame_and_segment_times_agree_with_the_documented_convention(self):
        fs, win, hop = 16000, 400, 160          # 25 ms / 10 ms
        self.assertEqual(frame_span_s(0, win, hop, fs), (0.0, 0.025))
        self.assertEqual(frame_span_s(10, win, hop, fs), (0.1, 0.125))
        # Frames [5, 8) span from the start of frame 5 to the end of frame 7.
        t0, t1 = segment_to_seconds(5, 8, win, hop, fs)
        self.assertAlmostEqual(t0, 0.05)
        self.assertAlmostEqual(t1, 0.095)
        with self.assertRaises(ValueError):
            segment_to_seconds(5, 5, win, hop, fs)

    def test_labels_use_the_same_convention_as_the_detector(self):
        """A ground-truth span of exactly one frame must light exactly the frames that
        overlap it by at least half, and no others."""
        fs, win, hop = 16000, 400, 160
        labels = labels_from_spans([(0.10, 0.125)], 20, win, hop, fs)
        lit = sorted(int(i) for i in np.flatnonzero(labels))
        # Frame m spans [10m, 10m+25) ms. Against a span of [100, 125) ms:
        #   frame  8: [ 80, 105)  ->  5 ms of 25   20%  no
        #   frame  9: [ 90, 115)  -> 15 ms of 25   60%  yes
        #   frame 10: [100, 125)  -> 25 ms of 25  100%  yes
        #   frame 11: [110, 135)  -> 15 ms of 25   60%  yes
        #   frame 12: [120, 145)  ->  5 ms of 25   20%  no
        # A 25 ms span therefore lights three frames at a 10 ms hop. That smearing is
        # the reason boundary error cannot be better than roughly half a window, and
        # it is why the sweep reports ~20 ms onset error as a good result rather than
        # as a defect.
        self.assertEqual(lit, [9, 10, 11])

    def test_ms_to_samples_never_collapses_a_window(self):
        self.assertEqual(ms_to_samples(25.0, 16000), 400)
        self.assertEqual(ms_to_samples(10.0, 16000), 160)
        self.assertEqual(ms_to_samples(0.0, 16000), 1)

    def test_bad_framing_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            n_frames(100, 0, 10)
        with self.assertRaises(ValueError):
            frame_signal(np.zeros((4, 4)), 2, 1)


if __name__ == "__main__":
    unittest.main()
