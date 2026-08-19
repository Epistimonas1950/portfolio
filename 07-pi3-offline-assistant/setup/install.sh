#!/usr/bin/env bash
#
# Reproducible setup from a fresh 64-bit Raspberry Pi OS Lite image.
#
# Run on the board, as a normal user with sudo rights:
#
#     git clone <this repo> ~/assistant && cd ~/assistant
#     ./setup/install.sh
#
# It is idempotent: re-running skips anything already built, so a failed run can be
# resumed rather than restarted. It never runs as root; it calls sudo for the three
# things that need it (apt, the swap change, the systemd unit) and nothing else.
#
# NOT RUN ANYWHERE YET. There is no Raspberry Pi attached to the machine this repo was
# developed on, so every line below is reasoned rather than verified. It is written to
# be readable as a specification of the build, and it refuses to run on the wrong
# architecture rather than half-succeeding. See STATUS.md.

set -euo pipefail

readonly PREFIX="${PREFIX:-${HOME}/assistant}"
readonly SRC_DIR="${PREFIX}/src"
readonly BIN_DIR="${PREFIX}/bin"
readonly MODEL_DIR="${PREFIX}/models"
readonly VOICE_DIR="${PREFIX}/voices"
readonly SERVICE_USER="${SERVICE_USER:-assistant}"
readonly JOBS="${JOBS:-4}"          # Pi 3 has 4 cores; -j4 with 1 GB RAM already swaps
readonly WHISPER_MODEL="${WHISPER_MODEL:-tiny.en-q5_1}"
readonly PIPER_VOICE="${PIPER_VOICE:-en_US-lessac-low}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxxx\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------- preflight

preflight() {
  [[ ${EUID} -ne 0 ]] || die "do not run this as root; it calls sudo where needed"

  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "aarch64" ]]; then
    die "architecture is '${arch}', not aarch64.

  This script builds for 64-bit ARM. Two reasons it refuses rather than adapting:

    1. On a Pi 3 the 32-bit image measurably costs inference throughput. ARMv8-A
       AArch64 has 32 128-bit NEON registers against AArch32's 16, and llama.cpp's
       quantized kernels are register-pressure bound. Running armhf here is the
       single easiest way to lose performance and not notice.
    2. If you are on x86_64 you are on your laptop. Do the training and the
       quantization here (see projects 01 and 02) and copy the GGUF over. Never
       train on the board.

  Flash Raspberry Pi OS Lite (64-bit) and try again. See setup/build_flags.md."
  fi

  local total_mb
  total_mb="$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)"
  log "detected aarch64, ${total_mb} MB RAM, ${JOBS} build jobs"
  if (( total_mb < 900 )); then
    warn "under 900 MB of RAM. The build will need swap; see ensure_swap below."
  fi

  command -v sudo >/dev/null || die "sudo is not installed"
}

# ------------------------------------------------------------------------ system

install_packages() {
  log "installing build dependencies"
  sudo apt-get update -qq
  # No build-essential meta-package: it pulls in more than is needed and on a 1 GB
  # board every avoided package is avoided disk and avoided apt time.
  sudo apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    gcc g++ make cmake pkg-config \
    libopenblas-dev \
    alsa-utils \
    python3 python3-numpy
}

ensure_swap() {
  # llama.cpp's C++ compile units peak well over 1 GB with -O3. Pi OS Lite ships
  # 100 MB of swap on the SD card, which is not enough to link. 1 GB is enough and is
  # switched off again at the end of the build: leaving a large SD-card swapfile
  # enabled is a good way to wear the card out and to make inference latency spiky
  # in a way that would pollute exactly the measurements this project is about.
  local conf=/etc/dphys-swapfile
  [[ -f ${conf} ]] || { warn "no dphys-swapfile; skipping swap resize"; return 0; }
  log "temporarily raising swap to 1024 MB for the build"
  sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' "${conf}"
  sudo systemctl restart dphys-swapfile
}

restore_swap() {
  local conf=/etc/dphys-swapfile
  [[ -f ${conf} ]] || return 0
  log "restoring swap to 100 MB"
  sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' "${conf}"
  sudo systemctl restart dphys-swapfile
}

# ------------------------------------------------------------------------- build

clone_or_update() {
  local url="$1" dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    log "updating $(basename "${dest}")"
    git -C "${dest}" pull --ff-only
  else
    log "cloning $(basename "${dest}")"
    git clone --depth 1 "${url}" "${dest}"
  fi
}

build_llama_cpp() {
  local dir="${SRC_DIR}/llama.cpp"
  clone_or_update https://github.com/ggerganov/llama.cpp "${dir}"
  log "building llama.cpp (see setup/build_flags.md for every flag)"
  cmake -S "${dir}" -B "${dir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH="armv8-a+crc+simd" \
    -DCMAKE_C_FLAGS="-mcpu=cortex-a53 -mtune=cortex-a53 -O3" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-a53 -mtune=cortex-a53 -O3" \
    -DLLAMA_CURL=OFF \
    -DGGML_OPENMP=ON
  cmake --build "${dir}/build" --config Release -j "${JOBS}"
  install -Dm755 "${dir}/build/bin/llama-cli" "${BIN_DIR}/llama-cli"
  install -Dm755 "${dir}/build/bin/llama-bench" "${BIN_DIR}/llama-bench"
}

build_whisper_cpp() {
  local dir="${SRC_DIR}/whisper.cpp"
  clone_or_update https://github.com/ggerganov/whisper.cpp "${dir}"
  log "building whisper.cpp"
  cmake -S "${dir}" -B "${dir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DCMAKE_C_FLAGS="-mcpu=cortex-a53 -mtune=cortex-a53 -O3" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-a53 -mtune=cortex-a53 -O3" \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON
  cmake --build "${dir}/build" --config Release -j "${JOBS}"
  install -Dm755 "${dir}/build/bin/whisper-cli" "${BIN_DIR}/whisper-cli"
  log "fetching whisper model ${WHISPER_MODEL}"
  mkdir -p "${MODEL_DIR}"
  bash "${dir}/models/download-ggml-model.sh" "${WHISPER_MODEL}" "${MODEL_DIR}"
}

install_piper() {
  # Piper ships prebuilt aarch64 binaries. Building it from source drags in onnxruntime,
  # which does not build comfortably on a 1 GB board; the release tarball is the
  # pragmatic choice and the reason is worth recording rather than hiding.
  log "installing Piper (prebuilt aarch64 release)"
  mkdir -p "${BIN_DIR}" "${VOICE_DIR}"
  if [[ ! -x "${BIN_DIR}/piper" ]]; then
    local tarball="${SRC_DIR}/piper_linux_aarch64.tar.gz"
    mkdir -p "${SRC_DIR}"
    curl -fsSL -o "${tarball}" \
      "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz"
    tar -xzf "${tarball}" -C "${SRC_DIR}"
    install -Dm755 "${SRC_DIR}/piper/piper" "${BIN_DIR}/piper"
  fi
  local base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low"
  for suffix in ".onnx" ".onnx.json"; do
    local dest="${VOICE_DIR}/${PIPER_VOICE}${suffix}"
    [[ -f "${dest}" ]] || curl -fsSL -o "${dest}" "${base}/${PIPER_VOICE}${suffix}"
  done
}

check_model() {
  # The GGUF is produced OFF the board by projects 01 and 02 and copied across. This
  # script will not download a stand-in: quietly substituting somebody else's model
  # would invalidate every number the latency budget is for.
  log "checking for the compressed model"
  shopt -s nullglob
  local ggufs=("${MODEL_DIR}"/*.gguf)
  shopt -u nullglob
  if (( ${#ggufs[@]} == 0 )); then
    warn "no .gguf in ${MODEL_DIR}."
    warn "Export it from projects 01/02 on your laptop and copy it over:"
    warn "    scp model-q4_k_m.gguf pi@raspberrypi.local:${MODEL_DIR}/"
    warn "This script will not download a substitute model."
  else
    log "found: ${ggufs[*]}"
  fi
}

# ------------------------------------------------------------------------ service

install_service() {
  log "installing the systemd unit"
  if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    # System account, no login shell, no home: the assistant needs to read models and
    # open the sound device, nothing else.
    sudo useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
  fi
  sudo usermod -aG audio "${SERVICE_USER}"
  sudo install -Dm644 "$(dirname "$0")/assistant.service" \
    /etc/systemd/system/assistant.service
  sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${PREFIX}"
  sudo systemctl daemon-reload
  sudo systemctl enable assistant.service
  log "enabled. start it with: sudo systemctl start assistant"
}

record_provenance() {
  # A build with no record of its flags is not reproducible, and the flags are half
  # the point of this project.
  local out="${PREFIX}/build-provenance.txt"
  log "recording build provenance to ${out}"
  {
    echo "date:       $(date -Is)"
    echo "uname:      $(uname -a)"
    echo "gcc:        $(gcc --version | head -1)"
    echo "cmake:      $(cmake --version | head -1)"
    echo "os:         $(sed -n 's/^PRETTY_NAME=//p' /etc/os-release)"
    echo "cpuinfo:    $(sed -n 's/^Model\s*:\s*//p' /proc/cpuinfo | head -1)"
    echo "llama.cpp:  $(git -C "${SRC_DIR}/llama.cpp" rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "whisper.cpp:$(git -C "${SRC_DIR}/whisper.cpp" rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "cflags:     -mcpu=cortex-a53 -mtune=cortex-a53 -O3"
  } | sudo tee "${out}" >/dev/null
}

main() {
  preflight
  mkdir -p "${SRC_DIR}" "${BIN_DIR}" "${MODEL_DIR}" "${VOICE_DIR}"
  install_packages
  ensure_swap
  trap restore_swap EXIT       # restore swap even if a build fails
  build_llama_cpp
  build_whisper_cpp
  install_piper
  check_model
  record_provenance
  install_service
  log "done. Binaries in ${BIN_DIR}; add it to PATH for src/orchestrate.py --real."
  log "Then measure: python3 bench/latency.py --real  (writes the real budget)"
}

main "$@"
