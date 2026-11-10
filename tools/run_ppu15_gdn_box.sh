#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/actlize-la-zw890-${SHA}-${STAMP}-$$}"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS=()
case "${NO_PERF:-0}" in
  0) ;;
  1) EXTRA_ARGS+=(--no-perf) ;;
  *) echo '[ppu15 box] NO_PERF must be0 or1' >&2; exit 2 ;;
esac
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
finish() {
  local rc=$? name
  trap - EXIT
  local files=()
  for name in sha.txt source.diff actlize.txt hgcc.txt runtime.txt identity.log binaries.sha256 \
      build.log primitives.log device.log device.json build/native; do
    if [[ -e "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -czf "$OUT/results.tar.gz" -C "$OUT" -- "${files[@]}"; fi
  if ((rc == 0)); then
    echo "[ppu15 box] PASS; NO_PERF=${NO_PERF:-0}"
  else
    echo "[ppu15 box] NOT_ADMITTED rc=$rc; SKIP is not PASS" >&2
  fi
  echo "[ppu15 box] artifacts=$OUT; upload $OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES="${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
# HGGC objects need this SDK's PPU runtime. Keep the container's paired CUDA
# facade for Torch: injecting CUDA_SDK/lib64 can break cudaGetDeviceProperties_v2
# before any GDN kernel is loaded. Compiler paths are not runtime overrides.
export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
git -C "$ROOT" rev-parse HEAD > "$OUT/sha.txt"
printf 'compile_sdk=%s\ncuda_facade=CONTAINER_ENV\nLD_LIBRARY_PATH=%s\n' \
  "$SDK" "$LD_LIBRARY_PATH" > "$OUT/runtime.txt"
echo "[ppu15 box] stage=device-identity sdk=$SDK cuda_facade=CONTAINER_ENV"
"$PYTHON" -m actlize_la.serving.device --target ppu15 2>&1 | tee "$OUT/identity.log"
git -C "$ROOT" submodule update --init third_party/actlize
git -C "$ROOT" diff --binary > "$OUT/source.diff"
git -C "$ROOT" submodule status third_party/actlize > "$OUT/actlize.txt"
"$PYTHON" "$ROOT/tools/hgcc_identity.py" --compiler "$SDK/bin/hgcc" > "$OUT/hgcc.txt" 2>&1
PPU_SDK="$SDK" BUILD_DIR="$OUT/build" PYTHON="$PYTHON" JOBS="${JOBS:-8}" \
  bash "$ROOT/scripts/build_ppu15.sh" 2>&1 | tee "$OUT/build.log"
shopt -s nullglob
bindings=("$OUT/build"/_gdn_residual_ppu15*.so)
if [[ ${#bindings[@]} != 1 ]]; then
  echo "[ppu15 box] FAIL: expected exactly one PPU1.5 binding, got ${#bindings[@]}" >&2
  exit 1
fi
export GDN_QSA_PPU15_EXTENSION="${bindings[0]}"
sha256sum "$OUT/build/libactlize_la_ppu15.so" "$GDN_QSA_PPU15_EXTENSION" "$OUT/build/test_ppu15_primitives" > "$OUT/binaries.sha256"
"$OUT/build/test_ppu15_primitives" 2>&1 | tee "$OUT/primitives.log"
"$PYTHON" "$ROOT/tests/test_ppu15_device.py" --extension "$GDN_QSA_PPU15_EXTENSION" --out "$OUT" \
  --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-20}" --warmup "${WARMUP:-5}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT/device.log"
