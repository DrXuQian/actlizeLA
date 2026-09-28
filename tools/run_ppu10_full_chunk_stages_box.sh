#!/usr/bin/env bash
# One immutable build; admit first, then paired complete-call ACU captures.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-full-chunk-stages-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
if [[ -e "$RUN" || ! -x "$ACU" ]]; then
  echo "[full stages ACU] FAIL: require fresh OUT=$RUN and executable ACU=$ACU" >&2
  exit 1
fi
echo "[full stages ACU] control=residual-full-chunk candidates=solve,output,both reference=FLA"
echo "[full stages ACU] metric=ALL-KERNEL-ACU-SUM admission=RAW-BIT+2%-oracle routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" PERF=0 STATE_PIPELINE_AB=1 \
  SPLIT_PREPARE_AB=0 AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"

checks=(l032_wy_solve_static l033_wy_gate_cache l034_wy_full_chunk l035_wy_full_chunk_stages)
cmake --build "$RUN/build" --target "${checks[@]}" -j"${JOBS:-16}" \
  2>&1 | tee "$RUN/full-stages-host-build.log"
for check in "${checks[@]}"; do
  "$RUN/build/$check" | tee "$RUN/$check.log"
done
python "$ROOT/dev/ppu/check_solve_static.py" --self-test --host "$RUN/build/l032_wy_solve_static" \
  --isa "$RUN/build/gdn_wy_ppu.isa" | tee "$RUN/solve-static-native.log"
python "$ROOT/dev/ppu/check_gate_cache.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  | tee "$RUN/gate-cache-native.log"
python "$ROOT/dev/ppu/check_gate_cache_solve.py" --self-test --library "$RUN/build/libgdn_wy_ppu.so" \
  | tee "$RUN/gate-cache-solve-composition.log"
python "$ROOT/dev/ppu/check_full_chunk.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  --library "$RUN/build/libgdn_wy_ppu.so" | tee "$RUN/full-chunk-native.log"
extensions=("$RUN/build"/_gdn_wy_ppu*.so)
if [[ ${#extensions[@]} != 1 || ! -f "${extensions[0]}" ]]; then
  echo "[full stages ACU] FAIL: unique WY extension missing" >&2
  exit 1
fi
python "$ROOT/dev/ppu/check_full_chunk_stages.py" --self-test --isa "$RUN/build/gdn_wy_ppu.isa" \
  --library "$RUN/build/libgdn_wy_ppu.so" --binding "${extensions[0]}" \
  | tee "$RUN/full-stages-native.log"

export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
deliveries=(full-chunk full-chunk-solve full-chunk-output full-chunk-both)
python "$ROOT/tests/test_ppu_residual_backend.py" --extension "${extensions[0]}" \
  --deliveries "${deliveries[@]}" 2>&1 | tee "$RUN/residual-correctness.log"
python "$ROOT/benchmarks/admit_ppu_residual_fla.py" --extension "${extensions[0]}" \
  --deliveries "${deliveries[@]}" --results "$RUN/comparison.json" 2>&1 | tee "$RUN/comparison.log"

# Fresh control and FLA alongside every candidate, not an old timing reused.
for stage in solve output both; do
  for gate in -1.0 -0.1; do
    capture="$RUN/acu-$stage-g${gate}"
    env -u EXTENSION OUT="$capture" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
      bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
      --wy-control residual-full-chunk --wy-delivery "residual-full-chunk-$stage" --gate "$gate"
    printf '[full stages ACU] captured stage=%s gate=%s\n' "$stage" "$gate"
  done
done
bash "$ROOT/tools/pack_ppu10_full_chunk_stages.sh" "$RUN"
printf '[full stages ACU] admission and six captures complete; speed requires ACU analysis\n'
