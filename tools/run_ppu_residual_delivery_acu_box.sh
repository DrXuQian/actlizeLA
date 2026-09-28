#!/usr/bin/env bash
# One same-math residual delivery candidate; FLA remains the external target.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/gdn-residual-delivery-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
CANDIDATE="${CANDIDATE:-residual-v16}"
case "$CANDIDATE" in
  residual-prefetch|residual-operands|residual-v16|residual-blayout|residual-warps8|residual-warps8-blayout|residual-warps8-operands|residual-warps8-hlayout|residual-warps8-hvlayout|residual-warps8-metadata|residual-solve-static|residual-full-chunk|residual-gate-cache|residual-gate-cache-solve-static) DELIVERY="${CANDIDATE#residual-}" ;;
  *) echo "[residual delivery ACU] FAIL: unknown CANDIDATE=$CANDIDATE" >&2; exit 1 ;;
esac
if [[ -e "$RUN" || ! -x "$ACU" ]]; then
  echo "[residual delivery ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
CONTROL=$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -c \
  'import sys; from actlize_la.gdn_residual_interface import RESIDUAL_CONTROLS; print(RESIDUAL_CONTROLS[sys.argv[1]])' "$CANDIDATE")
DELIVERIES=("$DELIVERY")
if [[ "$CONTROL" != residual ]]; then
  DELIVERIES=("${CONTROL#residual-}" "$DELIVERY")
fi
echo "[residual delivery ACU] control=$CONTROL subject=$CANDIDATE reference=FLA target=1.5x"
echo "[residual delivery ACU] metric=ALL-KERNEL-ACU-SUM admission=RAW-BIT+2%-oracle routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
if [[ "$CANDIDATE" == residual-blayout ]]; then
  cmake --build "$RUN/build" --target l024_wy_residual_blayout -j"${JOBS:-16}" \
    | tee "$RUN/blayout-host-build.log"
  "$RUN/build/l024_wy_residual_blayout" | tee "$RUN/blayout-layout.log"
  python "$ROOT/dev/ppu/check_residual_blayout.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/blayout-native.log"
fi
if [[ "$CANDIDATE" == residual-warps8 ]]; then
  cmake --build "$RUN/build" --target l025_wy_residual_warps8 -j"${JOBS:-16}" \
    | tee "$RUN/warps8-host-build.log"
  "$RUN/build/l025_wy_residual_warps8" | tee "$RUN/warps8-layout.log"
  python "$ROOT/dev/ppu/check_residual_warps8.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/warps8-native.log"
fi
if [[ "$CANDIDATE" == residual-warps8-blayout ]]; then
  cmake --build "$RUN/build" --target l027_wy_residual_warps8_blayout -j"${JOBS:-16}" \
    | tee "$RUN/warps8-blayout-host-build.log"
  "$RUN/build/l027_wy_residual_warps8_blayout" | tee "$RUN/warps8-blayout-layout.log"
  python "$ROOT/dev/ppu/check_residual_warps8_blayout.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/warps8-blayout-native.log"
fi
if [[ "$CANDIDATE" == residual-warps8-operands ]]; then
  cmake --build "$RUN/build" --target l028_wy_residual_warps8_operands -j"${JOBS:-16}" \
    | tee "$RUN/warps8-operands-host-build.log"
  "$RUN/build/l028_wy_residual_warps8_operands" | tee "$RUN/warps8-operands-schedule.log"
  python "$ROOT/dev/ppu/check_residual_warps8_operands.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/warps8-operands-native.log"
fi
if [[ "$CANDIDATE" == residual-warps8-hlayout ]]; then
  cmake --build "$RUN/build" --target l029_wy_residual_warps8_hlayout -j"${JOBS:-16}" \
    | tee "$RUN/warps8-hlayout-host-build.log"
  "$RUN/build/l029_wy_residual_warps8_hlayout" | tee "$RUN/warps8-hlayout-paired-map.log"
  python "$ROOT/dev/ppu/check_residual_warps8_hlayout.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/warps8-hlayout-native.log"
fi
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ "$CANDIDATE" == residual-warps8-hvlayout ]]; then
  cmake --build "$RUN/build" --target l030_wy_residual_warps8_hvlayout -j"${JOBS:-16}" \
    | tee "$RUN/warps8-hvlayout-host-build.log"
  "$RUN/build/l030_wy_residual_warps8_hvlayout" | tee "$RUN/warps8-hvlayout-paired-map.log"
  python "$ROOT/dev/ppu/check_residual_warps8_hvlayout.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/warps8-hvlayout-native.log"
fi
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[residual delivery ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
if [[ "$CANDIDATE" == residual-warps8-metadata ]]; then
  cmake --build "$RUN/build" --target l031_wy_residual_metadata -j"${JOBS:-16}" \
    | tee "$RUN/metadata-host-build.log"
  "$RUN/build/l031_wy_residual_metadata" | tee "$RUN/metadata-lifetime.log"
  python "$ROOT/dev/ppu/check_residual_metadata.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/metadata-native.log"
fi
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
if [[ "$CANDIDATE" == residual-gate-cache || "$CANDIDATE" == residual-gate-cache-solve-static || "$CANDIDATE" == residual-full-chunk ]]; then
  cmake --build "$RUN/build" --target l033_wy_gate_cache -j"${JOBS:-16}" | tee "$RUN/gate-cache-host-build.log"
  "$RUN/build/l033_wy_gate_cache" | tee "$RUN/gate-cache-host.log"
  python "$ROOT/dev/ppu/check_gate_cache.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" | tee "$RUN/gate-cache-native.log"
fi
if [[ "$CANDIDATE" == residual-solve-static || "$CANDIDATE" == residual-gate-cache-solve-static || "$CANDIDATE" == residual-full-chunk ]]; then
  cmake --build "$RUN/build" --target l032_wy_solve_static -j"${JOBS:-16}" \
    | tee "$RUN/solve-static-host-build.log"
  python "$ROOT/dev/ppu/check_solve_static.py" --self-test --host "$RUN/build/l032_wy_solve_static" \
    --isa "$RUN/build/gdn_wy_ppu.isa" \
    | tee "$RUN/solve-static-native.log"
fi
if [[ "$CANDIDATE" == residual-gate-cache-solve-static || "$CANDIDATE" == residual-full-chunk ]]; then
  python "$ROOT/dev/ppu/check_gate_cache_solve.py" --self-test \
    --library "$RUN/build/libgdn_wy_ppu.so" | tee "$RUN/gate-cache-solve-composition.log"
fi
if [[ "$CANDIDATE" == residual-full-chunk ]]; then
  cmake --build "$RUN/build" --target l034_wy_full_chunk -j"${JOBS:-16}" | tee "$RUN/full-chunk-host-build.log"
  "$RUN/build/l034_wy_full_chunk" | tee "$RUN/full-chunk-host.log"
  python "$ROOT/dev/ppu/check_full_chunk.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
    --library "$RUN/build/libgdn_wy_ppu.so" | tee "$RUN/full-chunk-native.log"
fi
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" --deliveries "${DELIVERIES[@]}" \
  2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" --deliveries "${DELIVERIES[@]}" \
  --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"
env -u EXTENSION OUT="$RUN/acu" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
  bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
  --wy-control "$CONTROL" --wy-delivery "$CANDIDATE" --gate "${GATE:--1.0}"
echo "[residual delivery ACU] completed; upload: $RUN/acu/$(basename "$RUN/acu").tar.gz"
