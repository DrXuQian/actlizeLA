#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${L211_OUT:-/workspace/actlizeLA-l211-extension-name}"
BUILD="$OUT/build"
PREFIX="$OUT/prefix"
CONSUMER="$OUT/external-consumer"
CHECKER="$ROOT/dev/gates/check_actlize_extensions_name.py"

mkdir -p "$BUILD" "$PREFIX" "$CONSUMER"

python3 "$CHECKER" --root "$ROOT" --require-canonical \
  | tee "$OUT/canonical.log"

# Prove that the stale-name scanner recognizes the retired spelling rather
# than merely passing the current tree.  Construct the token in the fixture so
# this gate remains outside its own forbidden-token set.
STALE_TREE="$OUT/stale-negative"
mkdir -p "$STALE_TREE"
printf '#include <%s%s/cutlass/linear_attention/ppu_chunked_gdn_types.hpp>\n' \
  'quactlize_' 'extensions' >"$STALE_TREE/stale_consumer.cpp"
if python3 "$CHECKER" --root "$STALE_TREE" >"$OUT/stale-negative.log" 2>&1; then
  echo '[l211] FAIL: planted retired include root escaped the stale-name gate' >&2
  exit 1
fi
grep -Fq 'legacy include-root token remains in stale_consumer.cpp' \
  "$OUT/stale-negative.log" || {
    echo '[l211] FAIL: stale-name negative failed for the wrong reason' >&2
    cat "$OUT/stale-negative.log" >&2
    exit 1
  }
echo '[l211 negative] stale-name=EXPECTED_RED/PASS'

cmake -S "$ROOT" -B "$BUILD" \
  -DACTLIZELA_ENABLE_PPU=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  >"$OUT/configure.log"
cmake --install "$BUILD" >"$OUT/install.log"

CANONICAL_HEADER="$PREFIX/include/actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"
[[ -f "$CANONICAL_HEADER" ]] || {
  echo "[l211] FAIL: install omitted canonical header: $CANONICAL_HEADER" >&2
  exit 1
}
RETIRED_ROOT='quactlize_'"extensions"
if find "$PREFIX/include" -type d -name "$RETIRED_ROOT" -print -quit \
    | grep -q .; then
  echo '[l211] FAIL: install contains the retired extension include root' >&2
  exit 1
fi

cat >"$CONSUMER/consumer.cpp" <<'CPP'
#include <actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp>

using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
static_assert(Traits::ChunkSize == 64);
static_assert(Traits::HeadSizeK == 128);
static_assert(Traits::HeadSizeV == 128);

int main() { return 0; }
CPP

(
  cd "$CONSUMER"
  env -u CPATH -u CPLUS_INCLUDE_PATH "${CXX:-g++}" \
    -std=c++17 -Wall -Wextra -Werror \
    -I"$PREFIX/include" consumer.cpp -o consumer
  ./consumer
) >"$OUT/external-consumer.log" 2>&1

# A downstream source using the retired spelling must fail against the exact
# installed prefix.  No source-tree include directory is supplied.
printf '#include <%s%s/cutlass/linear_attention/ppu_chunked_gdn_types.hpp>\nint main() { return 0; }\n' \
  'quactlize_' 'extensions' >"$CONSUMER/retired_consumer.cpp"
if (
  cd "$CONSUMER"
  env -u CPATH -u CPLUS_INCLUDE_PATH "${CXX:-g++}" \
    -std=c++17 -I"$PREFIX/include" retired_consumer.cpp -o retired_consumer
) >"$OUT/retired-consumer-negative.log" 2>&1; then
  echo '[l211] FAIL: installed prefix still resolves the retired include root' >&2
  exit 1
fi
grep -Fq 'ppu_chunked_gdn_types.hpp' "$OUT/retired-consumer-negative.log" || {
  echo '[l211] FAIL: retired-path compile negative failed for the wrong reason' >&2
  cat "$OUT/retired-consumer-negative.log" >&2
  exit 1
}
echo '[l211 negative] installed-retired-path=EXPECTED_RED/PASS'

echo '[l211] PASS: canonical source tree + installed external consumer use only actlize_extensions'
