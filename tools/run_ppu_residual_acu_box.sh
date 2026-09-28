#!/usr/bin/env bash
# One candidate, one admitted incumbent, installed FLA; all four kernel stages.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/gdn-residual-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
if [[ -e "$RUN" || ! -x "$ACU" ]]; then
  echo "[residual ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
echo "[residual ACU] control=state-pipeline subject=residual reference=FLA target=1.5x"
echo "[residual ACU] metric=ALL-KERNEL-ACU-SUM new-BF16-association=EXPLICIT routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[residual ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" \
  2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" \
  --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"
env -u EXTENSION OUT="$RUN/acu" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
  bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
  --wy-control state-pipeline --wy-delivery residual --gate "${GATE:--1.0}"
echo "[residual ACU] completed; upload: $RUN/acu/$(basename "$RUN/acu").tar.gz"
