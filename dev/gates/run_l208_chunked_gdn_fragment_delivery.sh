#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${QUACTLIZE_L208_OUT:-/workspace/actlizeLA-l208-fragment-delivery}
mkdir -p "$OUT"

NVCC=${NVCC:-nvcc}
command -v "$NVCC" >/dev/null || {
  echo '[l208] SKIP: nvcc is required for the local __HGGCCC__ fragment oracle'
  exit 3
}

common=(
  -std=c++17 -arch=sm_120 --expt-relaxed-constexpr
  -D__HGGCCC__=1
  -I"$ROOT/include"
  -I"$ROOT/third_party/actlize/include"
  -I"$ROOT/dev/gates/stub_inc"
  "$ROOT/dev/gates/l208_chunked_gdn_fragment_delivery.cu"
)

"$NVCC" "${common[@]}" -o "$OUT/l208" >"$OUT/positive-build.log" 2>&1
"$OUT/l208" | tee "$OUT/positive-run.log"
grep -Fq '[l208] PASS:' "$OUT/positive-run.log" || {
  echo '[l208] FAIL: positive fragment delivery witness changed' >&2
  exit 1
}

for plant in PHYSICAL_SLOT_ROTATE B_TRANSPOSE; do
  binary="$OUT/negative-${plant,,}"
  log="$binary.log"
  "$NVCC" "${common[@]}" -D"L208_PLANT_${plant}=1" \
      -o "$binary" >"$log" 2>&1
  if "$binary" >>"$log" 2>&1; then
    echo "[l208] FAIL: ${plant} negative unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq '[l208] FAIL:' "$log" || {
    echo "[l208] FAIL: ${plant} negative failed for the wrong reason" >&2
    exit 1
  }
  case "$plant" in
    PHYSICAL_SLOT_ROTATE)
      grep -Eq 'bf16-64\[FAIL\].*A=\(visits:8192,holes:0,aliases:0,bad:8192,.*B=\(visits:8192,holes:0,aliases:0,bad:8192,' "$log" || {
        echo '[l208] FAIL: physical-slot negative did not rotate every BF16 donor' >&2
        exit 1
      }
      ;;
    B_TRANSPOSE)
      grep -Eq 'bf16-64\[FAIL\].*A=\(visits:8192,holes:0,aliases:0,bad:0,.*B=\(visits:8192,holes:0,aliases:0,bad:8064,' "$log" || {
        echo '[l208] FAIL: B-transpose negative did not preserve A while corrupting B' >&2
        exit 1
      }
      ;;
  esac
  echo "[l208 negative] ${plant}=EXPECTED_RED/PASS"
done

echo "[l208] PASS: BF16/TF32 logical slots and compact physical donors closed; artifacts=$OUT"
