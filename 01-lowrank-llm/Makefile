PY ?= python3

.PHONY: help test results plots clean

help:
	@echo "make test     - run the unittest suite (numpy only, ~1 s)"
	@echo "make results  - regenerate every CSV in results/ (~4 min)"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"
	@echo "make clean    - remove __pycache__ directories"

test:
	$(PY) -m unittest discover -s tests -t . -v

results:
	$(PY) analysis/spectra.py
	$(PY) analysis/allocation_compare.py
	$(PY) analysis/pareto.py
	$(PY) analysis/composition.py

plots:
	$(PY) analysis/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ analysis/__pycache__
