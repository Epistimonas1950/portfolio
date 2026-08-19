PY ?= python3

.PHONY: help test demo results thermal plots clean

help:
	@echo "make test     - run the unittest suite (numpy only, ~8 s)"
	@echo "make demo     - one assistant turn: real capture + real VAD + SIMULATED inference"
	@echo "make results  - regenerate the deterministic CSVs in results/"
	@echo "make thermal  - re-run the host thermal log (UNSEEDED: answer differs every run)"
	@echo "make plots    - draw the figures (needs matplotlib in a venv, see requirements.txt)"
	@echo "make clean    - remove __pycache__ and generated audio"

test:
	$(PY) -m unittest discover -s tests -t . -v

demo:
	$(PY) -m src.orchestrate --turns 3

# Deterministic: both scripts are seeded, so re-running produces byte-identical
# CSVs apart from the wall-clock columns of the capture/vad rows, which are real
# measurements of a shared machine and are labelled as such.
results:
	$(PY) bench/vad_snr_sweep.py --seeds 20
	$(PY) bench/latency.py --turns 400

# Deliberately NOT part of `results`. This one has no seed and cannot have one: it
# records what this machine's temperature did during one particular 15 seconds, so
# re-running it always rewrites the committed CSV with a different answer.
thermal:
	$(PY) bench/thermal.py --seconds 15

plots:
	$(PY) bench/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ bench/__pycache__
	rm -f data/*.wav
