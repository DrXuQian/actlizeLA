#!/usr/bin/env bash
# Execute with bash; never source this script in the container shell.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
OUT="${OUT:-/workspace/actlize-la-native-default-${SHA:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
finish() {
  local rc=$? name
  trap - EXIT
  local files=()
  for name in identity.log build.log device.log results.json; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  echo "[native default box] rc=$rc upload=$OUT/results.tar.gz (no performance measurement)"
  exit "$rc"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export PYTHONPATH="$ROOT:$ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
{
  date -u '+utc=%Y-%m-%d %H:%M:%S UTC'
  echo "sha=$SHA target=ppu10"
  git -C "$ROOT" status --short
  git -C "$ROOT" submodule status -- third_party/actlize
} >"$OUT/identity.log"
# A supplied locally built LIB is executed as-is, never silently rebuilt.
if [[ -z "${LIB:-}" ]]; then
  test -x "$SDK/bin/hgcc"
  "$PYTHON" "$ROOT/tools/hgcc_identity.py" --compiler "$SDK/bin/hgcc" >>"$OUT/identity.log" 2>&1
  cmake -S "$ROOT/csrc/serving" -B "$OUT/build" -DALA_TARGET=ppu10 \
    -DPPU_SDK_ROOT="$SDK" -DALA_BUILD_SERVING_EXAMPLE=OFF \
    -DCMAKE_BUILD_TYPE=Release >"$OUT/build.log" 2>&1
  cmake --build "$OUT/build" -j"${JOBS:-8}" >>"$OUT/build.log" 2>&1
  LIB="$OUT/build/libactlize_la_serving.so"
fi
test -s "$LIB"
sha256sum "$LIB" | tee -a "$OUT/identity.log"
"$PYTHON" "$ROOT/tests/ppu_native_default_device.py" --library "$LIB" \
  --results "$OUT/results.json" 2>&1 | tee "$OUT/device.log"
