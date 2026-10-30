#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/workspace/actlize-la-ppu15-build}"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
PYTHON="${PYTHON:-python}"
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DGDN_QSA_TARGET=ppu15 -DGDN_QSA_ENABLE_PPU=OFF \
  -DGDN_QSA_BUILD_PPU_TESTS=OFF -DGDN_QSA_BUILD_PPU15_TESTS=OFF -DGDN_QSA_PPU15_DEVICE_TESTS=ON \
  -DPPU_SDK_ROOT="$SDK" -DPython3_EXECUTABLE="$(command -v "$PYTHON")" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target _gdn_residual_ppu15 test_ppu15_primitives -j"${JOBS:-8}"
"$PYTHON" "$ROOT/dev/ppu15/check_native.py" --sdk "$SDK" --out "$BUILD_DIR/native" "$BUILD_DIR/libactlize_la_ppu15.so"
sha256sum "$BUILD_DIR/libactlize_la_ppu15.so" "$BUILD_DIR"/_gdn_residual_ppu15*.so
echo "[ppu15 build] PASS: native compile/link; device numerics NOT_RUN"
