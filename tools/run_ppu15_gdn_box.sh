#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/actlize-la-zw890-${SHA}-${STAMP}}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT"
trap 'rc=$?; echo "[ppu15 box] FAIL rc=$rc; artifacts=$OUT" >&2; exit "$rc"' ERR
git -C "$ROOT" submodule update --init third_party/actlize
git -C "$ROOT" rev-parse HEAD > "$OUT/sha.txt"
git -C "$ROOT" diff --binary > "$OUT/source.diff"
git -C "$ROOT" submodule status third_party/actlize > "$OUT/actlize.txt"
"$SDK/bin/hgcc" --version > "$OUT/hgcc.txt"
PPU_SDK="$SDK" BUILD_DIR="$OUT/build" PYTHON="$PYTHON" JOBS="${JOBS:-8}" \
  bash "$ROOT/scripts/build_ppu15.sh" 2>&1 | tee "$OUT/build.log"
shopt -s nullglob
bindings=("$OUT/build"/_gdn_residual_ppu15*.so)
if [[ ${#bindings[@]} != 1 ]]; then
  echo "[ppu15 box] FAIL: expected exactly one PPU1.5 binding, got ${#bindings[@]}" >&2
  exit 1
fi
export GDN_QSA_PPU15_EXTENSION="${bindings[0]}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/CUDA_SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
sha256sum "$OUT/build/libactlize_la_ppu15.so" "$GDN_QSA_PPU15_EXTENSION" "$OUT/build/test_ppu15_primitives" > "$OUT/binaries.sha256"
"$OUT/build/test_ppu15_primitives" 2>&1 | tee "$OUT/primitives.log"
"$PYTHON" "$ROOT/tests/test_ppu15_device.py" --extension "$GDN_QSA_PPU15_EXTENSION" --out "$OUT" \
  --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-20}" --warmup "${WARMUP:-5}" 2>&1 | tee "$OUT/device.log"
# Reports and native evidence only; do not upload the build directory or SDK.
tar -czf "$OUT/results.tar.gz" -C "$OUT" sha.txt source.diff actlize.txt hgcc.txt \
  binaries.sha256 build.log primitives.log device.log device.json build/native
echo "[ppu15 box] PASS: $OUT; upload $OUT/results.tar.gz"
