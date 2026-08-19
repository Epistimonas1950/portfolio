PY ?= python3

.PHONY: test results plots clean help

help:
	@echo "make test     - run the unittest suite (numpy only, ~7 s)"
	@echo "make results  - regenerate every CSV in results/ (~45 s)"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"
	@echo "make clean    - remove __pycache__"

test:
	$(PY) -m unittest discover -s tests -t . -v

results:
	$(PY) analysis/convergence_order.py
	$(PY) analysis/nfe_vs_quality.py
	$(PY) analysis/stochastic_vs_deterministic.py
	$(PY) analysis/stability.py

plots:
	$(PY) analysis/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ src/samplers/__pycache__ tests/__pycache__ analysis/__pycache__
