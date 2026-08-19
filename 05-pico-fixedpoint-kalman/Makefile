# 05 · Pico fixed-point Kalman filter.
#
# Two build targets that do very different things:
#
#   host  plain gcc, runs here, produces every number in results/. The filter source in
#         firmware/ is portable C11 with no SDK dependency and no floating point, so the
#         arithmetic the host runs is bit-for-bit the arithmetic an RP2040 would run.
#         That is the whole reason the filter is written in fixed point rather than
#         leaning on a compiler's soft-float.
#
#   pico  cross-compile for the RP2040. CANNOT RUN ON THIS MACHINE: no Pico SDK, no
#         arm-none-eabi toolchain, no board. The target checks for both and explains
#         itself rather than failing with a confusing error. See STATUS.md.

PY      ?= python3
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
INC     := -Ifirmware
LDLIBS  := -lm

BUILD   := build
FILTER  := firmware/kalman_fixed.c firmware/kalman_joseph.c firmware/qformat.c \
           firmware/imu.c
TRACE   := data/imu_capture.csv

.PHONY: help host test results plots trace clean pico

help:
	@echo "make host     - build build/kfhost and build/qtest with plain gcc (runs here)"
	@echo "make trace    - regenerate data/imu_capture.csv (seeded, deterministic)"
	@echo "make test     - build, then run the unittest suite"
	@echo "make results  - regenerate every CSV in results/"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"
	@echo "make pico     - cross-compile for RP2040 (NOT possible on this machine)"
	@echo "make clean    - remove build/ and __pycache__"

host: $(BUILD)/kfhost $(BUILD)/qtest

$(BUILD)/kfhost: host/main.c $(FILTER) firmware/qformat.h firmware/kalman_fixed.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ host/main.c $(FILTER) $(LDLIBS)

$(BUILD)/qtest: host/qformat_selftest.c firmware/qformat.c firmware/qformat.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ host/qformat_selftest.c firmware/qformat.c $(LDLIBS)

trace $(TRACE):
	$(PY) reference/generate_trace.py

test: host $(TRACE)
	$(PY) -m unittest discover -s tests -t . -v

# Order matters and is not cosmetic: the error budget is written and run BEFORE the
# comparison, so its predictions cannot be adjusted to fit what was measured.
results: host $(TRACE)
	$(PY) reference/error_budget.py
	$(PY) reference/compare.py
	$(PY) analysis/gain_bits_sweep.py
	$(PY) analysis/cov_bits_sweep.py
	$(PY) analysis/dt_study.py

plots:
	$(PY) analysis/plot_results.py

# ---------------------------------------------------------------------------------
# RP2040 target. Not runnable here. Documented so the gap is visible, not hidden.
#
# The filter objects themselves are freestanding: they include only <stdint.h> and
# would cross-compile with
#
#   arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -O2 -ffreestanding -c \
#       firmware/kalman_fixed.c firmware/kalman_joseph.c firmware/qformat.c
#
# The application around them -- I2C, a repeating hardware timer at 200 Hz, USB CDC
# logging -- needs the SDK, and firmware/imu.c is a stub for exactly that reason.
# Cycle counts and RAM high-water marks are NOT reported anywhere in this repo because
# they cannot be measured here, and an estimate presented as a measurement would be
# worse than an empty cell.
# ---------------------------------------------------------------------------------
pico:
	@if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then \
	  echo "arm-none-eabi-gcc not found: cannot cross-compile for the RP2040."; \
	  echo "The filter is portable C11 and 'make host' runs the identical arithmetic."; \
	  echo "See STATUS.md for what needs hardware and what does not."; \
	  exit 1; \
	fi; \
	if [ -z "$$PICO_SDK_PATH" ]; then \
	  echo "PICO_SDK_PATH is not set: the application layer (I2C, timer, USB) needs it."; \
	  echo "The filter objects alone can still be built:"; \
	  echo "  arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -O2 -ffreestanding \\"; \
	  echo "      -Ifirmware -c firmware/kalman_fixed.c firmware/kalman_joseph.c"; \
	  exit 1; \
	fi; \
	echo "SDK found. This path has never been exercised -- see STATUS.md."; exit 1

clean:
	rm -rf $(BUILD) __pycache__ */__pycache__
