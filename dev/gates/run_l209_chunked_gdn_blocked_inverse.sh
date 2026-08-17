#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${QUACTLIZE_L209_OUT:-/workspace/actlizeLA-l209-blocked-inverse}
mkdir -p "$OUT"

CXX=${CXX:-g++}
command -v "$CXX" >/dev/null || {
  echo '[l209] SKIP: a C++17 compiler is required for the TF32 authority'
  exit 3
}

common=(
  -std=c++17 -O2
  -I"$ROOT/third_party/actlize/include"
  -I"$ROOT/dev/gates"
  "$ROOT/dev/gates/l209_chunked_gdn_blocked_inverse.cpp"
)

"$CXX" "${common[@]}" -o "$OUT/l209" >"$OUT/positive-build.log" 2>&1
"$OUT/l209" | tee "$OUT/positive-run.log"
grep -Fq '[l209] PASS:' "$OUT/positive-run.log" || {
  echo '[l209] FAIL: positive TF32 blocked-inverse witness changed' >&2
  exit 1
}

for plant in B_STRIDE MISSING_NEGATE SKIP_FINAL_MERGE; do
  binary="$OUT/negative-${plant,,}"
  log="$binary.log"
  "$CXX" "${common[@]}" -D"L209_PLANT_${plant}=1" \
      -o "$binary" >"$log" 2>&1
  if "$binary" >>"$log" 2>&1; then
    echo "[l209] FAIL: ${plant} negative unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq '[l209] FAIL:' "$log" || {
    echo "[l209] FAIL: ${plant} negative failed for the wrong reason" >&2
    exit 1
  }
  case "$plant" in
    B_STRIDE)
      grep -Fq 'block_products=6/6 exact_raw_bad=1488/4096 exact_residual=0.25' "$log" ;;
    MISSING_NEGATE)
      grep -Fq 'block_products=6/6 exact_raw_bad=1024/4096 exact_residual=1' "$log" ;;
    SKIP_FINAL_MERGE)
      grep -Fq 'block_products=4/4 exact_raw_bad=1024/4096 exact_residual=0.5' "$log" ;;
  esac || {
    echo "[l209] FAIL: ${plant} negative signature changed" >&2
    exit 1
  }
  echo "[l209 negative] ${plant}=EXPECTED_RED/PASS"
done

echo "[l209] PASS: TF32 seam and three blocked-inverse negatives closed; artifacts=$OUT"
