#!/usr/bin/env bash
# One build/admission, twelve five-arm captures, one upload archive.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-geometry-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
if [[ -e "$RUN" || -L "$RUN" || ! -x "$ACU" ]]; then
  echo "[geometry ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
echo "[geometry ACU] 6 shapes x 2 gates x (control + V32W8 + V32W4 + V16W4 + FLA)"
echo "[geometry ACU] all-kernel sums; no API timing, default routing unchanged"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" GDN_QSA_TARGET=ppu10 PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
cmake --build "$RUN/build" --target l040_wy_geometry -j"${JOBS:-16}" 2>&1 | tee "$RUN/geometry-host-build.log"
"$RUN/build/l040_wy_geometry" | tee "$RUN/l040_wy_geometry.log"
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[geometry ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
python "$ROOT/dev/ppu/check_geometry.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  --library "$RUN/build/libgdn_wy_ppu.so" --binding "${extensions[0]}" \
  | tee "$RUN/geometry-native.log"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" \
  --deliveries full-chunk geometry-v32-w8 geometry-v32-w4 geometry-v16-w4 \
  2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/tests/test_ppu_geometry_backend.py" --extension "${extensions[0]}" \
  2>&1 | tee "$RUN/geometry-edge.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" \
  --shapes "$ROOT/dev/ppu/geometry_shapes.json" \
  --deliveries full-chunk geometry-v32-w8 geometry-v32-w4 geometry-v16-w4 \
  --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"
PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" python "$ROOT/tools/sweep_ppu10_geometry.py" "$RUN"
