"""Voice activity detection: short-time energy and zero-crossing rate, with an
adaptive noise floor, two-threshold hysteresis and minimum-duration smoothing.

This is the one component of the brief that is real signal processing rather than
process orchestration, so it is the component this repo's credibility rests on. It
uses numpy and nothing else.

The mathematics
---------------

Frame `m` is the window `x[mR : mR+N]`. Two features per frame:

    E_m = 10 log10( (1/N) sum_n x[mR+n]^2 + eps )                          (dB)

    Z_m = (1 / (N-1)) sum_{n=1}^{N-1} 1[ sgn(x[mR+n]) != sgn(x[mR+n-1]) ]

`E` in dB rather than linear because the decision is about *ratios* to a noise floor,
and a ratio is a difference in dB -- which is what makes a single threshold constant
transferable between a quiet room and a loud one. `eps` is 1e-12, i.e. -120 dB,
comfortably below 16-bit quantisation noise (-96 dB) so it never sets the floor for
real audio but keeps digital silence finite.

`Z` is a crude spectral-centroid proxy: for a zero-mean signal the expected crossing
rate relates to the autocorrelation at lag one, Z ~ (1/pi) arccos(rho_1). It is cheap
and it separates a low-pitched voiced sound (few crossings) from a high-frequency
fricative (many) even when their energies are similar.

Noise floor
-----------

The floor is estimated from a leading silence window of `noise_window_ms`, using the
*median* of `E` over those frames and a median absolute deviation for spread:

    floor = median(E_0..E_K)        sigma = 1.4826 * median(|E_k - floor|)

Median, not mean: if the speaker starts early, up to half the estimation window can
be speech before the estimate moves. The 1.4826 makes the MAD a consistent estimator
of the standard deviation under a Gaussian, which is the only reason that constant is
there.

The floor then keeps adapting, but only while the state machine is confidently in
silence:

    floor <- (1 - a) * floor + a * E_m ,    a = floor_update

so a fan spinning up mid-recording is tracked while the speaker's own voice can never
raise the floor and deafen the detector.

Hysteresis
----------

Two thresholds, `floor + onset_db` and `floor + offset_db` with
`offset_db < onset_db`. Speech is declared after `onset_frames` consecutive frames
above the upper threshold and released after `hangover_frames` consecutive frames
below the lower one. A single threshold on a signal that happens to sit near it
produces a burst of one-frame segments -- chatter -- and the hysteresis band is what
prevents that; `tests/test_vad.py` builds exactly that signal and asserts one segment
instead of dozens.

The released segment ends at the *first* frame that fell below the lower threshold,
not at the end of the hangover, so the hangover does not inflate every duration.

ZCR endpoint extension
----------------------

Energy finds the loud voiced core of a word. The unvoiced consonants at its edges can
be 10-15 dB quieter and are routinely cut off. After the energy pass, each segment is
walked outwards while

    |Z_m - Z_floor| > k * sigma_Z(floor)     and     E_m > floor + zcr_energy_db

for at most `zcr_extend_max_frames`. The deviation is two-sided on purpose: whether
a fricative reads as *high* ZCR depends on the spectrum of the room tone it is sitting
in, and against a white floor -- which has the maximum possible crossing rate -- an
unvoiced sound is the *lower* of the two. A one-sided "high ZCR means unvoiced" rule
is a statement about a particular noise spectrum masquerading as a statement about
speech.

This is the classical two-feature endpoint rule of Rabiner & Sambur, "An Algorithm
for Determining the Endpoints of Isolated Utterances" (Bell System Technical Journal,
1975), reimplemented here rather than cited as a black box. Its contribution is
ablated in the tests and reported as a separate row of `results/vad_snr_sweep.csv`;
if it stopped helping it would be deleted rather than left in for decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.audio import frame_signal, ms_to_samples, segment_to_seconds

EPS = 1e-12  # -120 dB: below 16-bit quantisation noise, above log(0).
MAD_TO_SIGMA = 1.4826  # consistency constant for the MAD under a Gaussian.


@dataclass(frozen=True)
class VadConfig:
    """Every constant the detector uses, with the reason it has that value.

    Defaults are for 16 kHz speech, which is what whisper.cpp consumes.
    """

    frame_ms: float = 25.0
    # 25 ms holds >=2.5 periods of a 100 Hz male pitch, so the energy estimate is not
    # dominated by where in the glottal cycle the window happened to land, and is
    # still short enough for the signal to be quasi-stationary across it.
    hop_ms: float = 10.0
    # 10 ms is the boundary resolution of the whole detector: no endpoint can be
    # reported more precisely than one hop.

    noise_window_ms: float = 400.0
    # Leading silence used for the floor estimate. 400 ms = 40 frames, enough for a
    # median and a MAD to be stable, short enough to demand only a moment of quiet.

    onset_db: float = 8.0
    offset_db: float = 3.0
    # The 5 dB hysteresis band. Placed from the measured feature separation in
    # `src/synth.py`: at 20 dB SNR voiced cores land ~22 dB over the floor and
    # unvoiced edges ~5 dB. +8 dB therefore triggers on voiced speech only, and +3 dB
    # holds the segment open across the quiet edges instead of clipping them off.
    # The floor of a 25 ms frame of band-limited noise has a robust sigma near 1 dB,
    # so +8 dB is an 8-sigma onset: false triggers on the floor are not the binding
    # constraint, missed quiet speech is.

    onset_frames: int = 3
    # 30 ms of evidence before declaring speech. One frame is not evidence.
    hangover_frames: int = 10
    # 100 ms below the lower threshold before releasing. Plosive closures inside a
    # word are 20-80 ms of near-silence; a shorter hangover splits words at them.
    # It must stay below the shortest inter-word gap or it merges words instead.

    min_speech_ms: float = 120.0
    # Shorter than the shortest real utterance; anything briefer is a door slam.
    min_gap_ms: float = 60.0
    # Segments separated by less are merged before the duration rule is applied.

    zcr_extend: bool = True
    zcr_mad_k: float = 4.0
    # How far Z must sit from the floor's Z, in robust sigmas, to count as evidence.
    zcr_min_dev: float = 0.05
    # ...but never less than this in absolute terms. If the floor is digital silence
    # or a pure tone its ZCR spread collapses, k*sigma collapses with it, and the
    # extension rule starts firing on 1% wobbles and creeping across the recording.
    # Honest note: on the synthetic floor used in the tests 4*sigma is already 0.067,
    # so this guard never binds there and setting it to 0 changes no number in
    # `results/vad_snr_sweep.csv`. It is insurance against a degenerate floor, not a
    # tuned parameter, and is documented as such rather than credited with a gain.
    zcr_extend_max_frames: int = 15
    # 150 ms cap. Without a cap, a noise floor with drifting spectrum lets a single
    # segment crawl across the whole recording.
    zcr_energy_db: float = 2.0
    # An extension frame must still be audible; ZCR alone on pure silence is noise.

    floor_update: float = 0.02
    # ~0.5 s time constant at a 10 ms hop. Fast enough for a fan, far too slow to be
    # dragged up by speech even if the state machine were wrong about a frame.

    abs_onset_db: float | None = None
    abs_offset_db: float | None = None
    # Explicit absolute thresholds in dBFS, bypassing the adaptive floor entirely.
    # Used by the hysteresis and minimum-duration tests, which need the threshold to
    # be known independently of the signal, and by any caller with a calibrated mic.


@dataclass
class VadResult:
    """Detector output plus everything needed to see why it decided that."""

    segments: list[tuple[float, float]]  # seconds, [t0, t1)
    frame_labels: np.ndarray             # bool, one per frame
    energy_db: np.ndarray
    zcr: np.ndarray
    floor_track_db: np.ndarray           # the adaptive floor, per frame
    noise_floor_db: float                # the initial estimate
    noise_floor_sigma_db: float
    zcr_floor: float
    zcr_sigma: float
    win: int
    hop: int
    fs: int
    n_extended_frames: int = 0
    dropped_segments: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def speech_fraction(self) -> float:
        return float(self.frame_labels.mean()) if self.frame_labels.size else 0.0


# ------------------------------------------------------------------------- features


def energy_db(frames: np.ndarray) -> np.ndarray:
    """Short-time energy in dB, one value per frame."""
    return 10.0 * np.log10(np.mean(np.asarray(frames, dtype=np.float64) ** 2, axis=1) + EPS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs that change sign, in [0, 1].

    `np.signbit` rather than `np.sign` so that an exact 0.0 sample -- common in a
    digitally silent lead-in -- is treated as positive rather than as a third state
    that manufactures two crossings out of one.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.shape[0] == 0:
        return np.zeros(0)
    if frames.shape[1] < 2:
        raise ValueError("a zero-crossing rate needs at least 2 samples per frame")
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def robust_floor(values: np.ndarray) -> tuple[float, float]:
    """(median, MAD-based sigma) of a 1-D array. Both zero for an empty array."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(values - med)))
    return med, sigma


# -------------------------------------------------------------------- state machine


def hysteresis_segments(
    e_db: np.ndarray,
    onset: np.ndarray | float,
    offset: np.ndarray | float,
    onset_frames: int,
    hangover_frames: int,
) -> list[tuple[int, int]]:
    """Two-threshold state machine over frame energies. Returns [i, j) frame indices.

    Kept separate from `detect` and given per-frame threshold arrays so that it can be
    exercised on its own with thresholds that do not depend on the signal.
    """
    n = int(np.asarray(e_db).size)
    on = np.full(n, onset, dtype=np.float64) if np.isscalar(onset) else np.asarray(onset)
    off = np.full(n, offset, dtype=np.float64) if np.isscalar(offset) else np.asarray(offset)
    if np.any(off > on):
        raise ValueError("offset threshold must not exceed onset threshold: that is "
                         "not hysteresis, it is a race")

    segments: list[tuple[int, int]] = []
    in_speech = False
    run = 0
    start = 0
    for m in range(n):
        if not in_speech:
            if e_db[m] > on[m]:
                run += 1
                if run >= onset_frames:
                    in_speech = True
                    start = m - onset_frames + 1  # credit the whole triggering run
                    run = 0
            else:
                run = 0
        else:
            if e_db[m] < off[m]:
                run += 1
                if run >= hangover_frames:
                    in_speech = False
                    # End at the first sub-threshold frame, so the hangover buys
                    # robustness without inflating every reported duration.
                    segments.append((start, m - hangover_frames + 1))
                    run = 0
            else:
                run = 0
    if in_speech:
        segments.append((start, n))
    return [(i, j) for i, j in segments if j > i]


def smooth_segments(
    segments: list[tuple[int, int]], min_gap_frames: int, min_len_frames: int
) -> tuple[list[tuple[int, int]], int]:
    """Merge near neighbours, then drop the too-short. Returns (segments, n_dropped).

    Merge before dropping: two 80 ms halves of one word separated by a 30 ms closure
    are one 190 ms word, and dropping first would delete both.
    """
    if not segments:
        return [], 0
    merged: list[list[int]] = [list(segments[0])]
    for i, j in segments[1:]:
        if i - merged[-1][1] <= min_gap_frames:
            merged[-1][1] = max(merged[-1][1], j)
        else:
            merged.append([i, j])
    kept = [(i, j) for i, j in merged if (j - i) >= min_len_frames]
    return kept, len(merged) - len(kept)


def _extend_by_zcr(
    segments: list[tuple[int, int]],
    zcr: np.ndarray,
    e_db: np.ndarray,
    floor_track: np.ndarray,
    cfg: VadConfig,
    zcr_floor: float,
    zcr_sigma: float,
) -> tuple[list[tuple[int, int]], int]:
    """Rabiner-Sambur style endpoint extension. Returns (segments, frames_added)."""
    n = e_db.size
    thresh = max(cfg.zcr_mad_k * zcr_sigma, cfg.zcr_min_dev)
    added = 0
    out: list[tuple[int, int]] = []
    for i, j in segments:
        new_i, new_j = i, j
        for _ in range(cfg.zcr_extend_max_frames):
            m = new_i - 1
            if m < 0:
                break
            if abs(zcr[m] - zcr_floor) > thresh and e_db[m] > floor_track[m] + cfg.zcr_energy_db:
                new_i = m
                added += 1
            else:
                break
        for _ in range(cfg.zcr_extend_max_frames):
            m = new_j
            if m >= n:
                break
            if abs(zcr[m] - zcr_floor) > thresh and e_db[m] > floor_track[m] + cfg.zcr_energy_db:
                new_j = m + 1
                added += 1
            else:
                break
        out.append((new_i, new_j))
    return out, added


# ---------------------------------------------------------------------------- detect


def detect(x: np.ndarray, fs: int, cfg: VadConfig | None = None) -> VadResult:
    """Run the detector over a mono float signal. The only entry point callers need."""
    cfg = cfg or VadConfig()
    x = np.asarray(x, dtype=np.float64)
    win = ms_to_samples(cfg.frame_ms, fs)
    hop = ms_to_samples(cfg.hop_ms, fs)
    frames = frame_signal(x, win, hop)
    if frames.shape[0] == 0:
        raise ValueError(
            f"signal is {x.size} samples ({x.size / fs:.3f} s) but one frame needs "
            f"{win} samples ({cfg.frame_ms} ms). Nothing to detect."
        )

    e_db = energy_db(frames)
    zcr = zero_crossing_rate(frames)
    n = e_db.size

    n_noise = max(1, min(n, int(round(cfg.noise_window_ms / cfg.hop_ms))))
    if n_noise < 5 and cfg.abs_onset_db is None:
        raise ValueError(
            f"only {n_noise} frames of leading audio available for the noise-floor "
            f"estimate; need at least 5. Either give the detector "
            f"{5 * cfg.hop_ms:.0f} ms of leading silence or set abs_onset_db / "
            "abs_offset_db to use fixed thresholds."
        )
    floor0, floor_sigma = robust_floor(e_db[:n_noise])
    zcr_floor, zcr_sigma = robust_floor(zcr[:n_noise])

    absolute = cfg.abs_onset_db is not None and cfg.abs_offset_db is not None
    if (cfg.abs_onset_db is None) != (cfg.abs_offset_db is None):
        raise ValueError("give both abs_onset_db and abs_offset_db, or neither")

    # The floor adapts inside the state machine, so the thresholds are per-frame and
    # the two loops have to be the same loop. Run it once to produce the tracks.
    floor_track = np.empty(n)
    if absolute:
        floor_track[:] = 0.0
        onset_track = np.full(n, float(cfg.abs_onset_db))
        offset_track = np.full(n, float(cfg.abs_offset_db))
    else:
        floor = floor0
        in_speech = False
        run = 0
        for m in range(n):
            floor_track[m] = floor
            above = e_db[m] > floor + cfg.onset_db
            below = e_db[m] < floor + cfg.offset_db
            if not in_speech:
                run = run + 1 if above else 0
                if run >= cfg.onset_frames:
                    in_speech, run = True, 0
                elif not above:
                    # Adapt only in confirmed silence: speech can never raise the
                    # floor and deafen the detector.
                    floor = (1.0 - cfg.floor_update) * floor + cfg.floor_update * e_db[m]
            else:
                run = run + 1 if below else 0
                if run >= cfg.hangover_frames:
                    in_speech, run = False, 0
        onset_track = floor_track + cfg.onset_db
        offset_track = floor_track + cfg.offset_db

    raw = hysteresis_segments(e_db, onset_track, offset_track,
                              cfg.onset_frames, cfg.hangover_frames)

    extended = 0
    if cfg.zcr_extend and raw:
        raw, extended = _extend_by_zcr(raw, zcr, e_db, floor_track, cfg,
                                       zcr_floor, zcr_sigma)

    min_gap_frames = max(0, int(round(cfg.min_gap_ms / cfg.hop_ms)))
    min_len_frames = max(1, int(round(cfg.min_speech_ms / cfg.hop_ms)))
    kept, dropped = smooth_segments(raw, min_gap_frames, min_len_frames)

    labels = np.zeros(n, dtype=bool)
    for i, j in kept:
        labels[i:j] = True

    return VadResult(
        segments=[segment_to_seconds(i, j, win, hop, fs) for i, j in kept],
        frame_labels=labels,
        energy_db=e_db,
        zcr=zcr,
        floor_track_db=floor_track,
        noise_floor_db=floor0,
        noise_floor_sigma_db=floor_sigma,
        zcr_floor=zcr_floor,
        zcr_sigma=zcr_sigma,
        win=win,
        hop=hop,
        fs=fs,
        n_extended_frames=extended,
        dropped_segments=dropped,
        meta={"n_noise_frames": n_noise, "absolute_thresholds": absolute},
    )


def extract_speech(x: np.ndarray, fs: int, result: VadResult) -> np.ndarray:
    """Concatenate the detected segments. What actually gets handed to the ASR stage."""
    if not result.segments:
        return np.zeros(0)
    return np.concatenate([x[int(t0 * fs):int(t1 * fs)] for t0, t1 in result.segments])


# --------------------------------------------------------------------------- scoring


def frame_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-frame precision / recall / F1 against a ground-truth label array.

    Precision and recall are reported separately and never collapsed into accuracy:
    speech occupies well under half the frames of a typical utterance, so a detector
    that outputs nothing at all already scores a respectable accuracy.
    """
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"label arrays disagree: {pred.shape} vs {truth.shape}")
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


def boundary_errors_ms(
    pred_segments: list[tuple[float, float]], true_segments: list[tuple[float, float]]
) -> dict[str, float]:
    """Onset/offset error against ground truth, in milliseconds.

    Each true segment is matched to the predicted segment it overlaps most; true
    segments with no overlapping prediction are counted as misses and excluded from
    the error average rather than being given an arbitrary large penalty, so
    `match_rate` must always be read alongside the errors.
    """
    starts, ends = [], []
    matched = 0
    for t0, t1 in true_segments:
        best, best_ov = None, 0.0
        for p0, p1 in pred_segments:
            ov = max(0.0, min(t1, p1) - max(t0, p0))
            if ov > best_ov:
                best, best_ov = (p0, p1), ov
        if best is None:
            continue
        matched += 1
        starts.append(abs(best[0] - t0) * 1e3)
        ends.append(abs(best[1] - t1) * 1e3)
    n_true = len(true_segments)
    return {
        "match_rate": matched / n_true if n_true else 0.0,
        "start_mae_ms": float(np.mean(starts)) if starts else float("nan"),
        "end_mae_ms": float(np.mean(ends)) if ends else float("nan"),
        "n_matched": matched,
        "n_true": n_true,
        "n_pred": len(pred_segments),
    }
