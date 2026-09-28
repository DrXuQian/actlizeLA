#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/workspace/gdn-qsa-ppu-build}"
JOBS="${JOBS:-16}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"

mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DGDN_QSA_ENABLE_PPU=ON \
  -DGDN_QSA_TARGET="${GDN_QSA_TARGET:-ppu10}" \
  -DPPU_SDK_ROOT="$PPU_SDK_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target _gdn_chunk_ppu -j"$JOBS"

LIB="$BUILD_DIR/libgdn_qsa_ppu.so"
HGOBJDUMP="$PPU_SDK_ROOT/bin/hgobjdump"
test -s "$LIB"
if [[ ! -x "$HGOBJDUMP" ]]; then
  echo "[PPU binary audit] FAIL: hgobjdump unavailable at $HGOBJDUMP" >&2
  exit 1
fi
"$HGOBJDUMP" --arch=ppu1.0 --dump-isa "$LIB" >"$BUILD_DIR/gdn_qsa_ppu.isa"
"$HGOBJDUMP" --dump-resource-usage=all "$LIB" \
  >"$BUILD_DIR/gdn_qsa_ppu.resources"
python "$ROOT/dev/ppu/check_binary.py" "$BUILD_DIR"
echo "library: $LIB"
find "$BUILD_DIR" -maxdepth 1 -name '_gdn_chunk_ppu*.so' -print
