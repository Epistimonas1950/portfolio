PY ?= python3

.PHONY: help test results plots clean demo

help:
	@echo "make test     - run the unittest suite (numpy only, ~5 s)"
	@echo "make results  - regenerate every CSV in results/ (~4 min)"
	@echo "make demo     - one natural-language request, end to end"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"

test:
	$(PY) -m unittest discover -s tests -t . -v

results:
	$(PY) bench/optimality.py
	$(PY) bench/dominance.py
	$(PY) bench/adversarial.py
	$(PY) bench/translation_accuracy.py

demo:
	$(PY) -m src.demo

plots:
	$(PY) analysis/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ src/search/__pycache__ tests/__pycache__ bench/__pycache__ analysis/__pycache__
