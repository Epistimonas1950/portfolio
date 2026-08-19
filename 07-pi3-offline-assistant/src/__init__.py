"""An offline voice-assistant pipeline, instrumented for a latency budget.

This package contains the parts of the brief that can be built and tested without a
Raspberry Pi and without the three inference binaries: the voice-activity detector,
the WAV/framing layer, the stage abstraction with its timing instrumentation, and the
orchestration state machine.

The signal-processing core is the VAD. A frame of audio is judged speech or silence
from two short-time features,

    E_m  = 10 log10( (1/N) sum_n x[mR + n]^2 + eps )        short-time energy, dB
    Z_m  = (1 / 2(N-1)) sum_n |sgn(x[mR+n]) - sgn(x[mR+n-1])|   zero-crossing rate

with N the window length and R the hop. Thresholds are placed relative to a noise
floor estimated from a leading silence window, and the frame-to-frame decision is
smoothed by a two-threshold (hysteresis) state machine plus a minimum-duration rule.
See `src/vad.py` for the derivation and the parameter reasoning.

Nothing in this package fabricates a latency measurement. `SimulatedStage` draws from
an explicit distribution and labels every value it produces as simulated.
"""
