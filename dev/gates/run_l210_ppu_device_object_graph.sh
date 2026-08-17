#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
OUT="${L210_OUT:-/workspace/actlizeLA-l210-ppu-object-${SHA:0:8}-$$}"
SDK="$OUT/fake-sdk"
POS="$OUT/positive"
NEG="$OUT/negative"

mkdir -p "$SDK/bin" "$SDK/lib" "$SDK/include" \
  "$SDK/targets/x86_64-linux/include" "$POS" "$NEG"
cp "$ROOT/dev/gates/l210_fake_hgcc.sh" "$SDK/bin/hgcc"
chmod +x "$SDK/bin/hgcc"

for lib in hg_wrapper hggc_wrapper hggcrt1 hggc; do
  /usr/bin/cc -shared -fPIC -x c \
    "$ROOT/dev/gates/l210_empty_device_object.c" \
    -o "$SDK/lib/lib${lib}.so"
done

configure() {
  cmake -S "$ROOT" -B "$1" \
    -DACTLIZELA_ENABLE_PPU=ON \
    -DPPU_SDK_ROOT="$SDK" \
    -DCUTLASS_PPU_ARCHS=ppu0010 \
    -DCMAKE_BUILD_TYPE=Release
}

configure "$POS" >"$OUT/positive-configure.log"
L210_FIXTURE_SOURCE="$ROOT/dev/gates/l210_empty_device_object.c" \
  cmake --build "$POS" --target actlize_la_ppu -j8 \
  >"$OUT/positive-build.log" 2>&1

mapfile -t objects < <(find "$POS/ppu_obj" -maxdepth 1 -type f \
  -name 'ppu_chunked_gdn_backend_*.o' -print)
if [[ "${#objects[@]}" -ne 1 || ! -s "$POS/libactlize_la_ppu.so" ]]; then
  echo "[l210] FAIL: positive graph did not produce one object and one library" >&2
  exit 1
fi
build_make="$POS/CMakeFiles/actlize_la_ppu.dir/build.make"
if [[ "$(grep -Fc "$SDK/bin/hgcc" "$build_make")" -ne 1 ]]; then
  echo "[l210] FAIL: generated graph does not contain exactly one hgcc command" >&2
  exit 1
fi

configure "$NEG" >"$OUT/negative-configure.log"
if L210_DROP_OBJECT=1 \
   L210_FIXTURE_SOURCE="$ROOT/dev/gates/l210_empty_device_object.c" \
   cmake --build "$NEG" --target actlize_la_ppu -j8 \
     >"$OUT/negative-build.log" 2>&1; then
  echo "[l210] FAIL: missing-object plant unexpectedly linked" >&2
  exit 1
fi
if ! grep -Eq 'cannot find (ppu_obj/)?ppu_chunked_gdn_backend_[0-9a-f]+\.o' \
     "$OUT/negative-build.log"; then
  echo "[l210] FAIL: missing-object plant failed for another reason" >&2
  exit 1
fi

echo "[l210 negative] successful-hgcc-without-object=EXPECTED_RED/PASS"
echo "[l210] PASS: subdirectory compiler crossed into the shipping custom-command graph; object precedes link; artifacts=$OUT"
