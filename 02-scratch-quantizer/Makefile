PY ?= python3

.PHONY: test results clean help

help:
	@echo "make test     - run the unittest suite (numpy only)"
	@echo "make results  - regenerate every CSV in results/"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"

test:
	$(PY) -m unittest discover -s tests -t . -v

results:
	$(PY) analysis/bits_vs_error.py
	$(PY) analysis/damping_sweep.py
	$(PY) analysis/error_bound.py

plots:
	$(PY) analysis/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ analysis/__pycache__
