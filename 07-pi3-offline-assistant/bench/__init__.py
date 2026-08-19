"""Measurement harnesses: latency statistics and thermal/throughput logging.

Nothing in here produces a Raspberry Pi 3 measurement, because there is no Raspberry
Pi 3. What it does produce, on any host:

  - `latency.py`  exercises the timing and percentile arithmetic against a
                  distribution with closed-form quantiles, and writes
                  `results/latency_simulated.csv` -- named so that no reader can
                  mistake it for hardware data.
  - `thermal.py`  a real temperature-versus-throughput log from this host's
                  `/sys/class/thermal` zones, which validates the instrument the Pi
                  throttling study will use.
"""
