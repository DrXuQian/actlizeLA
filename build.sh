#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-/workspace/actlizeLA-build}"
JOBS="${JOBS:-16}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"

mkdir -p "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DACTLIZELA_ENABLE_PPU=ON \
  -DPPU_SDK_ROOT="$PPU_SDK_ROOT" \
  -DCUTLASS_PPU_ARCHS=ppu0010 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" \
  --target actlize_la_ppu test_ppu_chunked_gdn_abi \
  -j"$JOBS"

echo "library: $BUILD_DIR/libactlize_la_ppu.so"
echo "test:    $BUILD_DIR/test_ppu_chunked_gdn_abi"
