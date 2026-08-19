"""Synthetic speech-like signals with exact ground-truth boundaries.

The VAD has to be scored against something. Recording audio and hand-labelling it is
not reproducible on a machine with no microphone, so the test signal is generated:
that gives boundaries that are exact by construction rather than to within a human
annotator's reaction time, and an SNR that is set rather than inferred.

Three components, each chosen for what it does to the two features the detector uses:

  noise floor   pink-ish band noise, 80 Hz - 3.4 kHz, 1/f amplitude slope.
                Deliberately *not* white. White noise has the maximum possible
                zero-crossing rate, which would invert the usual "high ZCR means
                unvoiced" reasoning and make the ZCR feature useless-to-harmful. A
                band-limited floor sits at a middling ZCR, which is what a real room
                tone does.

  voiced core   harmonic complex, F0 in 100-160 Hz, 10 harmonics with 1/k amplitude
                and slight per-harmonic phase randomisation. High energy, LOW ZCR.

  unvoiced edge band noise 3-7 kHz at a fraction of the voiced amplitude. LOW energy --
                low enough to fall between the detector's two thresholds -- and HIGH
                ZCR. These sit at the start and end of a word, which is exactly the
                configuration the Rabiner-Sambur endpoint rule was designed for.

A "word" is [unvoiced edge][voiced core][unvoiced edge] and is labelled as one
segment spanning all three. So the energy channel finds the core and the ZCR channel
has to recover the edges; `tests/test_vad.py` ablates the ZCR channel and asserts that
the edge frames are what it buys.

SNR is defined over the speech regions only:

    SNR_dB = 10 log10( mean(speech^2 over labelled samples) / mean(noise^2) )

Measuring the numerator over the whole signal instead would make the reported SNR
depend on how much silence the utterance happens to contain.

Every draw goes through an explicit `np.random.default_rng(seed)`; there is no global
seeding anywhere in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_FS = 16_000  # whisper.cpp resamples everything to 16 kHz; start there.


@dataclass
class LabelledSignal:
    """A waveform plus the boundaries it was built from."""

    x: np.ndarray
    fs: int
    segments: list[tuple[float, float]]        # ground truth, seconds, [t0, t1)
    unvoiced_spans: list[tuple[float, float]]  # the low-energy/high-ZCR edges only
    snr_db: float
    lead_silence_s: float
    speech_rms: float = field(default=0.0)
    noise_rms: float = field(default=0.0)

    @property
    def duration_s(self) -> float:
        return self.x.size / self.fs


def _band_noise(
    rng: np.random.Generator, n: int, fs: int, f_lo: float, f_hi: float, slope: float = 0.0
) -> np.ndarray:
    """Gaussian noise shaped to a band, unit RMS.

    Shaping is done in the frequency domain with raised-cosine edges over a 20% skirt.
    A brick wall would ring; a first-order filter would not give a controllable band.
    `slope` applies an additional 1/f**slope amplitude tilt inside the band.
    """
    if n <= 0:
        return np.zeros(0)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    gain = np.zeros_like(freq)
    skirt_lo = max(f_lo * 0.2, 1.0)
    skirt_hi = max(f_hi * 0.2, 1.0)
    inside = (freq >= f_lo) & (freq <= f_hi)
    gain[inside] = 1.0
    rise = (freq < f_lo) & (freq > f_lo - skirt_lo)
    gain[rise] = 0.5 * (1.0 - np.cos(np.pi * (freq[rise] - (f_lo - skirt_lo)) / skirt_lo))
    fall = (freq > f_hi) & (freq < f_hi + skirt_hi)
    gain[fall] = 0.5 * (1.0 + np.cos(np.pi * (freq[fall] - f_hi) / skirt_hi))
    if slope:
        gain = gain / np.maximum(freq, 1.0) ** slope
    shaped = np.fft.irfft(spec * gain, n=n)
    rms = float(np.sqrt(np.mean(shaped ** 2)))
    return shaped / rms if rms > 0 else shaped


def _harmonic_complex(
    rng: np.random.Generator, n: int, fs: int, f0: float, n_harm: int = 10
) -> np.ndarray:
    """Unit-RMS harmonic stack with 1/k amplitudes -- a vowel-shaped spectrum."""
    t = np.arange(n) / fs
    out = np.zeros(n)
    for k in range(1, n_harm + 1):
        fk = f0 * k
        if fk >= 0.45 * fs:  # stay well clear of Nyquist so nothing aliases
            break
        out += (1.0 / k) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    rms = float(np.sqrt(np.mean(out ** 2)))
    return out / rms if rms > 0 else out


def _ramp(n: int, fs: int, ramp_ms: float = 8.0) -> np.ndarray:
    """Raised-cosine attack/decay. 8 ms is short enough that the labelled boundary is
    still meaningful and long enough not to put a click (broadband, all frames) at
    each edge."""
    r = min(int(ramp_ms * 1e-3 * fs), n // 2)
    env = np.ones(n)
    if r > 0:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        env[:r] = w
        env[-r:] = w[::-1]
    return env


def make_utterance(
    fs: int = DEFAULT_FS,
    snr_db: float = 20.0,
    seed: int = 0,
    n_words: int = 4,
    lead_silence_s: float = 0.6,
    word_core_ms: tuple[float, float] = (220.0, 420.0),
    edge_ms: tuple[float, float] = (70.0, 130.0),
    gap_ms: tuple[float, float] = (180.0, 420.0),
    unvoiced_gain: float = 0.14,
    word_gain_db: tuple[float, float] = (-6.0, 6.0),
    tail_silence_s: float = 0.4,
) -> LabelledSignal:
    """Build a noisy utterance and return it with its exact boundaries.

    Two parameters are set from measurement rather than taste:

    `unvoiced_gain` = 0.14 puts the unvoiced edges about 5 dB over the noise floor at
    20 dB SNR, against 22 dB for the voiced cores. That straddles the detector's two
    default thresholds (onset +8 dB, offset +3 dB): the energy channel triggers on
    cores only, and the edges are what the ZCR extension has to recover. If the edges
    were loud, the energy channel would find them and the ZCR ablation would measure
    nothing.

    `word_gain_db` = (-6, +6) gives each word an independent level, because real
    speech varies by well over 10 dB word to word. Without it every word crosses the
    detection threshold at the same SNR and the accuracy-vs-SNR curve is a cliff
    rather than a curve -- an artefact of the generator, not a property of the
    detector.
    """
    rng = np.random.default_rng(seed)
    lead = int(lead_silence_s * fs)
    tail = int(tail_silence_s * fs)

    pieces: list[np.ndarray] = [np.zeros(lead)]
    segments: list[tuple[float, float]] = []
    unvoiced: list[tuple[float, float]] = []
    cursor = lead

    for _ in range(n_words):
        n_pre = int(rng.uniform(*edge_ms) * 1e-3 * fs)
        n_core = int(rng.uniform(*word_core_ms) * 1e-3 * fs)
        n_post = int(rng.uniform(*edge_ms) * 1e-3 * fs)
        f0 = float(rng.uniform(100.0, 160.0))

        pre = unvoiced_gain * _band_noise(rng, n_pre, fs, 3000.0, 7000.0)
        core = _harmonic_complex(rng, n_core, fs, f0)
        post = unvoiced_gain * _band_noise(rng, n_post, fs, 3000.0, 7000.0)
        word = np.concatenate([pre, core, post])
        word *= _ramp(word.size, fs)
        word *= 10.0 ** (rng.uniform(*word_gain_db) / 20.0)

        t0 = cursor / fs
        t1 = (cursor + word.size) / fs
        segments.append((t0, t1))
        unvoiced.append((t0, (cursor + n_pre) / fs))
        unvoiced.append(((cursor + n_pre + n_core) / fs, t1))

        gap = int(rng.uniform(*gap_ms) * 1e-3 * fs)
        pieces.append(word)
        pieces.append(np.zeros(gap))
        cursor += word.size + gap

    pieces.append(np.zeros(tail))
    speech = np.concatenate(pieces)

    noise = _band_noise(rng, speech.size, fs, 80.0, 3400.0, slope=1.0)

    # SNR over the labelled regions only -- see the module docstring.
    mask = np.zeros(speech.size, dtype=bool)
    for t0, t1 in segments:
        mask[int(t0 * fs):int(t1 * fs)] = True
    speech_rms = float(np.sqrt(np.mean(speech[mask] ** 2)))
    noise_gain = speech_rms / (10.0 ** (snr_db / 20.0))
    x = speech + noise_gain * noise

    # Headroom check: keep the peak inside int16 range so a WAV round trip of the same
    # signal does not clip and quietly change the measurement.
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = 0.85 * x / peak
        scale = 0.85 / peak
    else:
        scale = 1.0

    return LabelledSignal(
        x=x,
        fs=fs,
        segments=segments,
        unvoiced_spans=unvoiced,
        snr_db=snr_db,
        lead_silence_s=lead_silence_s,
        speech_rms=speech_rms * scale,
        noise_rms=noise_gain * scale,
    )


def make_plateau(
    fs: int = DEFAULT_FS,
    seed: int = 0,
    level_db: float = -30.0,
    jitter_db: float = 3.0,
    plateau_s: float = 1.2,
    lead_silence_s: float = 0.6,
) -> np.ndarray:
    """A signal engineered to sit *on* a threshold and jitter across it.

    Used to show that hysteresis suppresses chatter. The level is held at `level_db`
    with a frame-rate random walk of +-`jitter_db`, so a single-threshold detector set
    at `level_db` toggles on essentially every frame. Nothing about this is speech; it
    is a test instrument.
    """
    rng = np.random.default_rng(seed)
    lead = int(lead_silence_s * fs)
    n = int(plateau_s * fs)
    block = int(0.010 * fs)  # jitter at the hop rate, so each frame gets a new draw
    n_blocks = n // block
    gains = 10.0 ** ((level_db + rng.uniform(-jitter_db, jitter_db, n_blocks)) / 20.0)
    carrier = _band_noise(rng, n_blocks * block, fs, 300.0, 3000.0)
    env = np.repeat(gains, block)
    plateau = carrier * env
    floor = 10.0 ** (-60.0 / 20.0) * _band_noise(rng, lead, fs, 80.0, 3400.0, slope=1.0)
    return np.concatenate([floor, plateau, np.zeros(int(0.2 * fs))])


def make_blip(
    fs: int = DEFAULT_FS,
    seed: int = 0,
    blip_ms: float = 25.0,
    lead_silence_s: float = 0.6,
    level_db: float = -6.0,
) -> np.ndarray:
    """One loud, very short burst on a quiet floor. Shorter than any real word, so a
    minimum-duration rule must discard it."""
    rng = np.random.default_rng(seed)
    lead = int(lead_silence_s * fs)
    n = int(blip_ms * 1e-3 * fs)
    floor_gain = 10.0 ** (-60.0 / 20.0)
    total = lead + n + int(0.5 * fs)
    x = floor_gain * _band_noise(rng, total, fs, 80.0, 3400.0, slope=1.0)
    burst = 10.0 ** (level_db / 20.0) * _harmonic_complex(rng, n, fs, 130.0)
    x[lead:lead + n] += burst * _ramp(n, fs, ramp_ms=2.0)
    return x
