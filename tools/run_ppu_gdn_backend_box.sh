#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPU_SDK_ROOT="${PPU_SDK:-/usr/local/PPU_SDK}"
JOBS="${JOBS:-16}"
DEVICE="${DEVICE:-0}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/gdn-qsa-ppu-${SHA}-${STAMP}}"
BUILD_DIR="$OUT/build"

mkdir -p "$OUT" "$BUILD_DIR"
git -C "$ROOT" submodule update --init --recursive

BUILD_DIR="$BUILD_DIR" PPU_SDK="$PPU_SDK_ROOT" JOBS="$JOBS" \
  bash "$ROOT/scripts/build_ppu.sh" 2>&1 | tee "$OUT/build.log"

LIB="$BUILD_DIR/libgdn_qsa_ppu.so"
bindings=("$BUILD_DIR"/_gdn_chunk_ppu*.so)
if [[ ${#bindings[@]} -ne 1 || ! -f "${bindings[0]}" ]]; then
  echo "[PPU GDN box] FAIL: expected one original-dispatch Python binding" >&2
  exit 1
fi
EXTENSION="${bindings[0]}"
git -C "$ROOT" rev-parse HEAD | tee "$OUT/sha.txt"
git -C "$ROOT" diff --binary > "$OUT/source.diff"
sha256sum "$LIB" "$EXTENSION" | tee "$OUT/library.sha256"
if [[ "${WITH_FLA:-0}" == 1 ]]; then
  if [[ -n "${FLA_ROOT:-}" && ! -f "$FLA_ROOT/fla/__init__.py" ]]; then
    echo "[PPU GDN FLA] FAIL: FLA_ROOT is not an FLA checkout: $FLA_ROOT" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$DEVICE" \
  LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    python "$ROOT/benchmarks/bench_ppu_gdn_fla.py" \
      --extension "$EXTENSION" --device 0 --fla-heads "${FLA_HEADS:-native}" \
      --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-10}" --warmup "${WARMUP:-5}" \
      --results "$OUT/fla-comparison.json" 2>&1 | tee "$OUT/fla-comparison.log"
  echo "[PPU GDN FLA box] PASS: artifacts=$OUT"
  exit 0
fi
perf_args=()
if [[ "${PERF:-1}" == 1 ]]; then
  perf_args=(--perf --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-5}" --warmup "${WARMUP:-2}")
fi
CUDA_VISIBLE_DEVICES="$DEVICE" \
LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python "$ROOT/tests/test_ppu_gdn_backend.py" \
    --extension "$EXTENSION" --device 0 "${perf_args[@]}" 2>&1 | tee "$OUT/device.log"

echo "[PPU GDN box] PASS: artifacts=$OUT"
