PY ?= python3 -u

.PHONY: help test results plots clean

help:
	@echo "make test     - run the unittest suite (numpy + PyYAML only, ~80 s)"
	@echo "make results  - regenerate every CSV in results/ (~22 min)"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"
	@echo "make clean    - remove __pycache__ and the generated agent corpus"

test:
	$(PY) -m unittest discover -s tests -t . -v

results:
	$(PY) eval/regret.py
	$(PY) eval/coverage.py
	$(PY) eval/pareto.py
	$(PY) eval/compounding.py
	$(PY) eval/surrogate_bias.py

plots:
	$(PY) analysis/plot_results.py

clean:
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__ tests/__pycache__ \
	       eval/__pycache__ analysis/__pycache__ data/corpus
