#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
families=0
for flag in "${DELIVERY_AB:-0}" "${TILE_AB:-0}" "${STATE_AB:-0}" "${STAGE_AB:-0}" "${PREPARE_ROWS_AB:-0}" "${AIU_AB:-0}" "${SPLIT_PREPARE_AB:-0}" "${STATE_PIPELINE_AB:-0}"; do
  if [[ "$flag" != 0 && "$flag" != 1 ]]; then
    echo "[WY box] FAIL: candidate family flags must be 0 or 1" >&2
    exit 1
  fi
  families=$((families + flag))
done
if (( families > 1 )); then
  echo "[WY box] FAIL: choose only one of DELIVERY_AB / TILE_AB / STATE_AB / STAGE_AB / PREPARE_ROWS_AB / AIU_AB / SPLIT_PREPARE_AB / STATE_PIPELINE_AB" >&2
  exit 1
fi
PPU_SDK_ROOT="${PPU_SDK:-/usr/local/PPU_SDK}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
OUT="${OUT:-/workspace/gdn-wy-fla-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/sha.txt"
git -C "$ROOT" diff --binary > "$OUT/source.diff"
git -C "$ROOT" submodule update --init --recursive
git -C "$ROOT" submodule status > "$OUT/submodules.txt"
BUILD_DIR="$OUT/build" PPU_SDK="$PPU_SDK_ROOT" JOBS="${JOBS:-16}" \
  bash "$ROOT/scripts/build_ppu.sh" 2>&1 | tee "$OUT/build-original.log"
cmake --build "$OUT/build" --target _gdn_wy_ppu -j"${JOBS:-16}" 2>&1 | tee "$OUT/build-wy.log"
"$PPU_SDK_ROOT/bin/hgobjdump" --arch=ppu1.0 --dump-isa "$OUT/build/libgdn_wy_ppu.so" > "$OUT/build/gdn_wy_ppu.isa"
"$PPU_SDK_ROOT/bin/hgobjdump" --dump-resource-usage=all "$OUT/build/libgdn_wy_ppu.so" > "$OUT/build/gdn_wy_ppu.resources"
python "$ROOT/dev/ppu/check_wy_binary.py" "$OUT/build" --self-test | tee "$OUT/codegen.log"
old=("$OUT/build"/_gdn_chunk_ppu*.so)
new=("$OUT/build"/_gdn_wy_ppu*.so)
if [[ ${#old[@]} != 1 || ${#new[@]} != 1 || ! -f "${old[0]}" || ! -f "${new[0]}" ]]; then
  echo "[WY box] FAIL: exact Python bindings missing" >&2
  exit 1
fi
sha256sum "${old[0]}" "${new[0]}" "$OUT/build/libgdn_qsa_ppu.so" "$OUT/build/libgdn_wy_ppu.so" > "$OUT/binaries.sha256"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
variant_flags=()
sample_flags=()
if [[ -n "${SAMPLES+x}" ]]; then
  sample_flags+=(--samples "$SAMPLES")
fi
if [[ "${ADMISSION_ONLY:-0}" == 1 ]]; then
  sample_flags+=(--admission-only)
fi
if [[ "${DELIVERY_AB:-0}" == 1 ]]; then
  variant_flags+=(--delivery-ab)
fi
if [[ "${TILE_AB:-0}" == 1 ]]; then
  variant_flags+=(--tile-ab)
fi
if [[ "${STATE_AB:-0}" == 1 ]]; then
  variant_flags+=(--state-ab)
fi
if [[ "${STAGE_AB:-0}" == 1 ]]; then
  variant_flags+=(--stage-ab)
fi
if [[ "${PREPARE_ROWS_AB:-0}" == 1 ]]; then
  variant_flags+=(--prepare-rows-ab)
fi
if [[ "${AIU_AB:-0}" == 1 ]]; then
  variant_flags+=(--aiu-ab)
fi
if [[ "${SPLIT_PREPARE_AB:-0}" == 1 ]]; then
  variant_flags+=(--split-prepare-ab)
fi
if [[ "${STATE_PIPELINE_AB:-0}" == 1 ]]; then
  variant_flags+=(--state-pipeline-ab)
fi
python "$ROOT/tests/test_ppu_gdn_backend.py" --extension "${old[0]}" --device 0 2>&1 | tee "$OUT/original-correctness.log"
python "$ROOT/tests/test_ppu_wy_backend.py" --wy-extension "${new[0]}" --device 0 "${variant_flags[@]}" 2>&1 | tee "$OUT/wy-correctness.log"
if [[ "${PERF:-1}" == 1 ]]; then
  python "$ROOT/benchmarks/bench_ppu_wy_fla.py" --extension "${old[0]}" --wy-extension "${new[0]}" \
    --device 0 "${sample_flags[@]}" --launches "${LAUNCHES:-10}" --warmup "${WARMUP:-5}" "${variant_flags[@]}" \
    --results "$OUT/comparison.json" 2>&1 | tee "$OUT/comparison.log"
fi
echo "[WY box] PASS: artifacts=$OUT auto-routing=UNCHANGED"
