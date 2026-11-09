#!/usr/bin/env bash
# Run with bash in a child shell. Recheck eight shapes, not the ZW610 matrix.
# Kernel sources/dispatch are unchanged; all builds and captures stay on box.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PPU_SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
export DEVICE="${DEVICE:-0}" PYTHON="${PYTHON:-python3}"
export ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
export CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/tests:$ROOT/tools${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PPU_SDK/CUDA_SDK/bin:$PPU_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$PPU_SDK/CUDA_SDK/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SHA="$(git -C "$ROOT" rev-parse --short=8 HEAD)"
OUT="${OUT:-/workspace/actlize-la-zw810-recheck-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
export TMPDIR="$OUT/scratch" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUT/triton-cache}"
mkdir -p -- "$TMPDIR" "$TRITON_CACHE_DIR"

finish() {
  local rc=$?
  trap - EXIT
  local files=() name
  for name in runner.log identity.log routes.log build-run.log capture.log \
      build-run/profiler.log build-run/profiler/profiler.json \
      build-run/profiler/acu-version.log build-run/profiler/acu-version.log.command \
      build-run/profiler/acu-section-check.log build-run/profiler/acu-section-check.log.command \
      build-run/identity.txt build-run/binary.json build-run/build.log \
      build-run/results.json build-run/aux-results.json build-run/device-tests.log \
      build-run/aux-device.log build-run/native-graph.log acu/results.tar.gz; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  if ((rc == 0)); then
    echo "[ZW810 recheck] COLLECTED: numerics and measurement completed; speed verdict is in acu/summary.json"
  else
    echo "[ZW810 recheck] NOT_ADMITTED: rc=$rc; SKIP is not PASS" >&2
  fi
  echo "[ZW810 recheck] artifacts=$OUT upload=$OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT

printf '[ZW810 recheck] sha=%s out=%s cells=8 B=1 Hk=16 Hv=32,48 T=2048,8192 g=-0.1,-1.0 acu=%s\n' \
  "$SHA" "$OUT" "$ACU" | tee "$OUT/runner.log"

if [[ ! -x "$PPU_SDK/bin/hgcc" || ! -x "$ACU" ]]; then
  echo "ERROR: set matching PPU_SDK and executable ACU; SDK=$PPU_SDK ACU=$ACU" >&2
  exit 2
fi
if [[ ! -f "$ROOT/third_party/actlize/CMakeLists.txt" ]]; then
  echo "ERROR: initialize the pinned third_party/actlize submodule before running" >&2
  exit 2
fi
"$PYTHON" "$ROOT/tools/check_ppu810_gdn_recheck.py" 2>&1 | tee "$OUT/identity.log"
# Reuse the maintained builder and full serving/aux/native graph gates.
# Do not accidentally attach an unrelated old H1 experiment from the environment.
VALIDATE_ONLY=1 BASELINE_SERVING_RUN="" OUT="$OUT/build-run" \
  bash "$ROOT/tools/run_ppu10_gdn_tuning_box.sh" 2>&1 | tee "$OUT/build-run.log"
"$PYTHON" "$ROOT/tools/check_ppu810_gdn_recheck.py" \
  --library "$OUT/build-run/build/libactlize_la_serving.so" 2>&1 | tee "$OUT/routes.log"
# Same saved processed operands for current public core and FLA Triton.
# The current collector also records complete serving and all helper costs.
# Do not relabel the ZW610-only manifest as a ZW810 measurement.
OUT="$OUT/acu" "$PYTHON" "$ROOT/tools/collect_serving_prefill_acu.py" \
  --serving-run "$OUT/build-run" --device "$DEVICE" --acu "$ACU" \
  --batches 1 --lengths 2048 8192 --value-heads 32 48 --gates -0.1 -1.0 \
  --repeats 3 2>&1 | tee "$OUT/capture.log"
