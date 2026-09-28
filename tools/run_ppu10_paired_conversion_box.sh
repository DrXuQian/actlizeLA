#!/usr/bin/env bash
# One build, same-binary admission, two complete ACU captures, one upload tar.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-paired-conversion-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
if [[ -e "$RUN" || -L "$RUN" || ! -x "$ACU" ]]; then
  echo "[paired conversion ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
echo "[paired conversion ACU] control=residual-full-chunk candidate=residual-paired-conversion reference=FLA"
echo "[paired conversion ACU] metric=ALL-KERNEL-ACU-SUM S=2048 initial=None admission=RAW-BIT+2%-oracle routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" GDN_QSA_TARGET=ppu10 PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
cmake --build "$RUN/build" --target l039_wy_paired_conversion -j"${JOBS:-16}" \
  2>&1 | tee "$RUN/paired-conversion-host-build.log"
"$RUN/build/l039_wy_paired_conversion" | tee "$RUN/l039_wy_paired_conversion.log"
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[paired conversion ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
python "$ROOT/dev/ppu/check_paired_conversion.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  --host "$RUN/build/l039_wy_paired_conversion" --library "$RUN/build/libgdn_wy_ppu.so" \
  --binding "${extensions[0]}" | tee "$RUN/paired-conversion-native.log"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" \
  --deliveries full-chunk paired-conversion 2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/tests/test_ppu_paired_conversion_backend.py" --extension "${extensions[0]}" \
  2>&1 | tee "$RUN/paired-conversion-edge.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" \
  --deliveries full-chunk paired-conversion --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"
for gate in -1.0 -0.1; do
  capture="$RUN/acu-paired-g${gate}"
  env -u EXTENSION OUT="$capture" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
    bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
    --wy-control residual-full-chunk --wy-delivery residual-paired-conversion --gate "$gate"
  printf '[paired conversion ACU] captured gate=%s\n' "$gate"
done
bash "$ROOT/tools/pack_ppu10_paired_conversion.sh" "$RUN"
printf '[paired conversion ACU] admission and two captures complete; speed requires ACU analysis\n'
