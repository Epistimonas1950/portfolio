PY      ?= python3
CC       = gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDLIBS  ?= -lm

SRC     := src/linalg.c src/rangefinder.c src/incsvd.c src/reorth.c src/forget.c \
           src/detect.c src/rank.c host/main.c
BUILD   := build

.PHONY: host test demo data results plots clean help cross-note asan

help:
	@echo "make host     - build build/tracker (double) and build/tracker32 (float) with plain gcc"
	@echo "make test     - unittest suite; builds the C first, then shells out to it"
	@echo "make demo     - run the tracker on the labelled stream and print its summary"
	@echo "make data     - regenerate the synthetic streams in data/"
	@echo "make results  - regenerate every CSV in results/ (implies host + data)"
	@echo "make plots    - draw the figures (needs matplotlib in a venv)"
	@echo "make asan     - rebuild under AddressSanitizer + UBSan and exercise all three modes"
	@echo "make cross-note - print the cross-compile recipe for the Pi Zero"

# The whole C build. No BLAS, no LAPACK, no third-party header: one gcc invocation.
# The float32 binary is the same source with a different scalar_t; it exists because
# the orthogonality drift it produces is two orders of magnitude larger than the double
# build's, and reporting the same experiment at two precisions is worth more than
# reporting either alone.
host: $(BUILD)/tracker $(BUILD)/tracker32

$(BUILD)/tracker: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

$(BUILD)/tracker32: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -DSUBSPACE_FLOAT32 -o $@ $(SRC) $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

# Hand-written linear algebra with hand-managed scratch buffers is exactly the code that
# should be run under a sanitizer before anyone believes its numbers. This target rebuilds
# at -O1 with AddressSanitizer and UBSan and exercises every mode; it passes clean, which
# is why the results in results/ are worth reading.
asan: | $(BUILD)
	$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined \
	    -o $(BUILD)/tracker-asan $(SRC) $(LDLIBS)
	$(BUILD)/tracker-asan selftest
	$(BUILD)/tracker-asan track --input data/anomalous.csv \
	    --output $(BUILD)/asan_scores.csv --repeat 2 --basis $(BUILD)/asan_basis.csv
	$(BUILD)/tracker-asan rangefind --input data/normal.csv --rank 8 \
	    --oversampling 6 --power-iters 2
	@echo "asan: clean"

test: host
	$(PY) -m unittest discover -s tests -t . -v

# The first thing to run. Streams the labelled anomaly file through the compiled tracker
# and prints what it found: chosen rank, calibrated threshold, how many samples it flagged,
# the orthogonality drift it ended on, and its own accounting of its heap.
demo: host data
	./$(BUILD)/tracker track --input data/anomalous.csv \
	    --output $(BUILD)/demo_scores.csv --lambda 0.99875
	@echo "per-sample scores -> $(BUILD)/demo_scores.csv"

data:
	$(PY) oracle/generate_data.py

results: host data
	$(PY) oracle/batch_svd.py
	$(PY) oracle/roc.py
	$(PY) oracle/compare.py
	$(PY) analysis/drift_study.py
	$(PY) analysis/multimode_study.py
	$(PY) analysis/rangefinder_study.py
	$(PY) analysis/forgetting_study.py

plots:
	$(PY) analysis/plot_results.py

# The board is armv6 (ARM11) on a Zero / Zero W, and armv7/aarch64 on a Zero 2 W. The
# recipe below is what would build this for the original Zero. It is NOT run here: no
# board is attached and no cross toolchain is installed on this machine, so every number
# in README.md and results/ is from the x86 host build. STATUS.md says the same.
#
#   arm-linux-gnueabihf-gcc -O2 -Wall -Wextra -std=c11 \
#       -march=armv6zk -mfpu=vfp -mfloat-abi=hard -mtune=arm1176jzf-s \
#       -o build/tracker-armv6 $(SRC) -lm
#
# Nothing in src/ uses an intrinsic, an extension or a library beyond libm, so the only
# thing that changes is the triple. That portability is the point of the project: the
# framework route (a TFLite or ONNX wheel) has no armv6 build at all, which is why the
# comparison row in README.md's table is empty rather than estimated.
cross-note:
	@sed -n '/^# The board is armv6/,/^cross-note:/p' Makefile | sed 's/^# \{0,1\}//' | head -n -1

clean:
	rm -rf $(BUILD) __pycache__ src/__pycache__ tests/__pycache__ \
	       oracle/__pycache__ analysis/__pycache__
