#!/usr/bin/env bash
# Run with bash, not source: failures never close the interactive container.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
OUT="${OUT:-/workspace/actlize-la-ppu10-default-${SHA:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
finish() {
  local rc=$?
  trap - EXIT
  local files=() name
  for name in identity.log build.log host.log retained-device.log default-device.log results.json; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  if ((rc == 0)); then
    echo '[PPU default box] PASS: public-default NUMERICS+ROUTING; serving/performance NOT_RUN'
  elif ((rc == 3)); then
    echo '[PPU default box] SKIP/NOT_ADMITTED: no device; not PASS'
  else
    echo "[PPU default box] FAIL rc=$rc; inspect preserved logs"
  fi
  echo "[PPU default box] artifacts=$OUT upload=$OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT:$ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
{
  date -u '+utc=%Y-%m-%d %H:%M:%S UTC'
  echo "sha=$SHA target=ppu10 source=$ROOT"
  git -C "$ROOT" status --short
  git -C "$ROOT" submodule status -- third_party/actlize
  "$PYTHON" -m actlize_la.serving.device --target ppu10
} 2>&1 | tee "$OUT/identity.log"

# Kernels did not change. EXTENSION may reuse an existing retained binary;
# the admission records its *loaded* binding + device DSO, not a guessed file.
if [[ -n "${EXTENSION:-}" ]]; then
  test -f "$EXTENSION"
else
  if [[ ! -x "$SDK/bin/hgcc" ]]; then
    echo "ERROR: missing $SDK/bin/hgcc; set PPU_SDK to the matching SDK" >&2
    exit 2
  fi
  "$SDK/bin/hgcc" --version | tee -a "$OUT/identity.log"
  cmake -S "$ROOT" -B "$OUT/build" -DGDN_QSA_TARGET=ppu10 \
    -DPPU_SDK_ROOT="$SDK" -DPython3_EXECUTABLE="$(command -v "$PYTHON")" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DGDN_QSA_BUILD_PPU_TESTS=OFF -DGDN_QSA_BUILD_PPU15_TESTS=OFF >"$OUT/build.log" 2>&1
  cmake --build "$OUT/build" --target _gdn_wy_ppu -j"${JOBS:-8}" >>"$OUT/build.log" 2>&1
  shopt -s nullglob
  binaries=("$OUT/build"/_gdn_wy_ppu*.so)
  if ((${#binaries[@]} != 1)); then
    echo 'ERROR: expected exactly one _gdn_wy_ppu extension' >&2
    exit 2
  fi
  EXTENSION="${binaries[0]}"
fi
sha256sum "$EXTENSION" | tee -a "$OUT/identity.log"
"$PYTHON" -m unittest discover -s "$ROOT/tests" -p test_gdn_route.py -v \
  >"$OUT/host.log" 2>&1
"$PYTHON" "$ROOT/tests/test_ppu_residual_backend.py" --extension "$EXTENSION" \
  --deliveries full-chunk 2>&1 | tee "$OUT/retained-device.log"
"$PYTHON" "$ROOT/tests/test_ppu_default_device.py" --extension "$EXTENSION" \
  --results "$OUT/results.json" 2>&1 | tee "$OUT/default-device.log"
