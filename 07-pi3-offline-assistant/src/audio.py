"""WAV I/O and the framing convention, with numpy and the standard library only.

Two things live here, and both are load-bearing for every number this repo reports.

**1. The sample scaling.** 16-bit PCM stores integers in [-32768, 32767]. This module
uses 32768 as the divisor in both directions:

    float  = int16 / 32768.0                         exact, lands in [-1, 1)
    int16  = clip(round(float * 32768.0), -32768, 32767)

so an int16 -> float -> int16 round trip is bit exact (a scale by 2^-15 followed by a
scale by 2^15 is exact in binary floating point), and a float -> int16 -> float round
trip is bounded by half an LSB, 2^-16. Using 32767 in one direction and 32768 in the
other -- a common shortcut -- breaks the bit-exact round trip and puts a slow drift
into any repeated resample. `tests/test_audio.py` asserts both bounds.

**2. The frame convention.** Frame `m` covers samples `[m*R, m*R + N)` for hop `R` and
window `N`, so in seconds it spans

    t0(m) = m*R / fs                    t1(m) = (m*R + N) / fs

and a segment running from frame `i` to frame `j` (exclusive) spans
`[i*R/fs, ((j-1)*R + N)/fs)`. Every conversion in the repo goes through the helpers
below rather than being open-coded. Ground-truth labels are converted with the same
functions the detector uses; mixing a frame-start convention with a frame-centre one
produces a boundary error pinned near N/2 that looks like a detector bug and is not.
"""

from __future__ import annotations

import pathlib
import wave

import numpy as np

INT16_SCALE = 32768.0
INT16_MIN = -32768
INT16_MAX = 32767


# --------------------------------------------------------------------------- scaling


def int16_to_float(x: np.ndarray) -> np.ndarray:
    """Exact: dividing by a power of two only shifts the exponent."""
    return np.asarray(x, dtype=np.int16).astype(np.float64) / INT16_SCALE


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest with clipping. +1.0 saturates to 32767 rather than wrapping."""
    scaled = np.rint(np.asarray(x, dtype=np.float64) * INT16_SCALE)
    return np.clip(scaled, INT16_MIN, INT16_MAX).astype(np.int16)


# ------------------------------------------------------------------------------- I/O


def read_wav(path: str | pathlib.Path, mono: bool = True) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV. Returns (float64 samples in [-1, 1), sample rate).

    Only 16-bit PCM is accepted, because that is what whisper.cpp wants and silently
    accepting 8- or 24-bit here would move the failure somewhere less obvious.
    """
    path = pathlib.Path(path)
    with wave.open(str(path), "rb") as fh:
        width = fh.getsampwidth()
        if width != 2:
            raise ValueError(
                f"{path}: sample width is {width} bytes; this reader handles 16-bit "
                "PCM only. Convert with `sox in.wav -b 16 out.wav` or ffmpeg."
            )
        n_channels = fh.getnchannels()
        fs = fh.getframerate()
        raw = fh.readframes(fh.getnframes())
    data = np.frombuffer(raw, dtype="<i2")
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        if not mono:
            return int16_to_float(data), fs
        # Average in float, not int16: summing two near-full-scale channels as int16
        # overflows.
        return int16_to_float(data).mean(axis=1), fs
    return int16_to_float(data), fs


def write_wav(path: str | pathlib.Path, x: np.ndarray, fs: int) -> pathlib.Path:
    """Write mono 16-bit PCM. Accepts float in [-1, 1) or int16 (written unchanged)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    pcm = x.astype(np.int16) if x.dtype == np.int16 else float_to_int16(x)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(fs))
        fh.writeframes(pcm.astype("<i2").tobytes())
    return path


# --------------------------------------------------------------------------- framing


def n_frames(n_samples: int, win: int, hop: int) -> int:
    """Number of whole frames of length `win` at stride `hop`. Partial tails dropped."""
    if win <= 0 or hop <= 0:
        raise ValueError(f"win and hop must be positive, got win={win} hop={hop}")
    if n_samples < win:
        return 0
    return 1 + (n_samples - win) // hop


def frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice into an (n_frames, win) read-only view. No copy, no padding.

    A trailing partial frame is dropped rather than zero-padded: padding would drop
    the short-time energy of the last frame and put a spurious offset at the end of
    every segment.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {x.shape}")
    m = n_frames(x.size, win, hop)
    if m == 0:
        return np.empty((0, win), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:m]


def ms_to_samples(ms: float, fs: int) -> int:
    """Round to the nearest sample; a window is never allowed to collapse to zero."""
    return max(1, int(round(ms * 1e-3 * fs)))


def frame_span_s(m: int, win: int, hop: int, fs: int) -> tuple[float, float]:
    """The time span [t0, t1) covered by frame `m`."""
    return (m * hop) / fs, (m * hop + win) / fs


def segment_to_seconds(i: int, j: int, win: int, hop: int, fs: int) -> tuple[float, float]:
    """Frames [i, j) -> seconds [t0, t1). `j` is exclusive, matching Python slicing."""
    if j <= i:
        raise ValueError(f"empty frame segment [{i}, {j})")
    return (i * hop) / fs, ((j - 1) * hop + win) / fs


def labels_from_spans(
    spans_s: list[tuple[float, float]],
    n_frame: int,
    win: int,
    hop: int,
    fs: int,
    min_overlap: float = 0.5,
) -> np.ndarray:
    """Ground-truth spans in seconds -> a boolean label per frame.

    A frame is labelled speech when at least `min_overlap` of its duration falls
    inside some span. The alternative -- any overlap at all -- inflates recall by
    handing the detector a free frame at each boundary, so the stricter rule is used
    and stated rather than left implicit.
    """
    labels = np.zeros(n_frame, dtype=bool)
    frame_dur = win / fs
    for m in range(n_frame):
        t0, t1 = frame_span_s(m, win, hop, fs)
        covered = 0.0
        for s0, s1 in spans_s:
            covered += max(0.0, min(t1, s1) - max(t0, s0))
        labels[m] = covered >= min_overlap * frame_dur
    return labels
