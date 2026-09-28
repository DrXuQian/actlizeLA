#!/usr/bin/env bash
# One build, same-binary admission, two paired ACU captures and one upload tar.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-inverse-register-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
if [[ -e "$RUN" || -L "$RUN" || ! -x "$ACU" ]]; then
  echo "[inverse register ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
echo "[inverse register ACU] control=residual-full-chunk candidate=residual-inverse-register reference=FLA"
echo "[inverse register ACU] metric=ALL-KERNEL-ACU-SUM S=2048 initial=None admission=RAW-BIT+2%-oracle routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" GDN_QSA_TARGET=ppu10 PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
cmake --build "$RUN/build" --target l038_wy_inverse_register -j"${JOBS:-16}" \
  2>&1 | tee "$RUN/inverse-register-host-build.log"
"$RUN/build/l038_wy_inverse_register" | tee "$RUN/l038_wy_inverse_register.log"
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[inverse register ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
python "$ROOT/dev/ppu/check_inverse_register.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  --host "$RUN/build/l038_wy_inverse_register" --library "$RUN/build/libgdn_wy_ppu.so" \
  --binding "${extensions[0]}" | tee "$RUN/inverse-register-native.log"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" \
  --deliveries full-chunk inverse-register 2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/tests/test_ppu_inverse_register_backend.py" --extension "${extensions[0]}" \
  2>&1 | tee "$RUN/inverse-register-edge.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" \
  --deliveries full-chunk inverse-register --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"
for gate in -1.0 -0.1; do
  capture="$RUN/acu-inverse-g${gate}"
  env -u EXTENSION OUT="$capture" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
    bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
    --wy-control residual-full-chunk --wy-delivery residual-inverse-register --gate "$gate"
  printf '[inverse register ACU] captured gate=%s\n' "$gate"
done
bash "$ROOT/tools/pack_ppu10_inverse_register.sh" "$RUN"
printf '[inverse register ACU] admission and two captures complete; speed requires ACU analysis\n'
