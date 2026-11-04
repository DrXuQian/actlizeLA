#!/usr/bin/env bash
# Run in a child shell: failures do not exit the user's interactive container.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-ppu15}"
PYTHON="${PYTHON:-python}"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/actlize-la-flare-${TARGET}-${SHA:0:8}-${STAMP}-$$}"
mkdir -p -- "$(dirname "$OUT")"
if ! mkdir -- "$OUT"; then
  echo "ERROR: OUT must be a new directory; refusing to mix old results: $OUT" >&2
  exit 2
fi

finish() {
  local rc=$?
  trap - EXIT
  # Small handoff: logs/results/identity only, not the build tree or caches.
  local files=()
  local name
  for name in identity.txt build.log host-tests.log device-tests.log results.json; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  if ((rc==0)); then
    echo "[Flare serving box] PASS: operator numerical + graph gates; full-model admission is separate"
  else
    echo "[Flare serving box] NOT_ADMITTED: rc=$rc; inspect logs (SKIP is not PASS)"
  fi
  echo "[Flare serving box] artifacts=$OUT upload=$OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT

export PYTHONPATH="$ROOT:$ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
{
  echo "utc=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "sha=$SHA target=$TARGET"
  git -C "$ROOT" status --short
  git -C "$ROOT" submodule status -- third_party/actlize
  "$PYTHON" -c 'import sys,torch; print("python="+sys.version); print("torch="+torch.__version__); print("runtime="+str(torch.version.cuda))'
} | tee "$OUT/identity.txt"

case "$TARGET" in
  ppu10|ppu15)
    SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
    if [[ ! -x "$SDK/bin/hgcc" ]]; then
      echo "ERROR: no hgcc at $SDK/bin/hgcc; set PPU_SDK to the matching SDK" >&2
      exit 2
    fi
    if [[ ! -f "$ROOT/third_party/actlize/CMakeLists.txt" ]]; then
      echo 'ERROR: missing pinned submodule; run git submodule update --init third_party/actlize before this script' >&2
      exit 2
    fi
    export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$SDK/bin/hgcc" --version | tee -a "$OUT/identity.txt"
    cmake -S "$ROOT/csrc/serving" -B "$OUT/build" -DALA_TARGET="$TARGET" \
      -DPPU_SDK_ROOT="$SDK" -DCMAKE_BUILD_TYPE=Release >"$OUT/build.log" 2>&1
    ;;
  cuda_sm120)
    cmake -S "$ROOT/csrc/serving" -B "$OUT/build" -DALA_TARGET=cuda_sm120 \
      -DCMAKE_CUDA_COMPILER="${CUDACXX:-nvcc}" -DCMAKE_BUILD_TYPE=Release >"$OUT/build.log" 2>&1
    ;;
  *) echo "ERROR: TARGET must be ppu10, ppu15 or cuda_sm120" >&2; exit 2 ;;
esac
cmake --build "$OUT/build" -j"${JOBS:-8}" >>"$OUT/build.log" 2>&1
LIB="$OUT/build/libactlize_la_serving.so"
test -s "$LIB"
sha256sum "$LIB" | tee -a "$OUT/identity.txt"
ALA_SERVING_TEST_LIBRARY="$LIB" "$PYTHON" -m pytest -q "$ROOT/tests/test_serving_contract.py" \
  2>&1 | tee "$OUT/host-tests.log"
"$PYTHON" "$ROOT/tests/serving_device.py" --target "$TARGET" --library "$LIB" \
  --results "$OUT/results.json" 2>&1 | tee "$OUT/device-tests.log"
