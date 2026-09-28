#!/usr/bin/env bash
# Real CuTe ownership maps, executed on the host. No device kernel launch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUTLASS_ROOT="${CUTLASS_ROOT:-$ROOT/third_party/cutlass}"
OUT="${OUT:-/workspace/gdn-sm90-host-maps-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ -x "$NVCC" && -f "$CUTLASS_ROOT/include/cute/tensor.hpp" && ! -e "$OUT" ]] || {
  echo "[SM90 maps] require executable NVCC, CUTLASS_ROOT and fresh OUT" >&2
  exit 1
}
mkdir -p "$OUT/scratch"
export TMPDIR="$OUT/scratch"
export CUDA_VISIBLE_DEVICES=''
for row in "value_tile_map 1" "inverse_local_map 2" "paired_tail_map 3" "relative_gate_layout 3"; do
  read -r proof configuration <<< "$row"
  "$NVCC" -std=c++17 -O2 -arch=sm_90a --expt-relaxed-constexpr --extended-lambda \
    "-DGDN_SM90_CONFIGURATION=$configuration" -I"$ROOT/csrc/backends/sm90" \
    -I"$ROOT/csrc/backends/sm90/cula" -I"$CUTLASS_ROOT/include" \
    "$ROOT/dev/backends/sm90_$proof.cu" -o "$OUT/$proof" > "$OUT/$proof.build.log" 2>&1
  "$OUT/$proof" | tee "$OUT/$proof.log"
done
echo "[SM90 maps] PASS proofs=4 device=NOT_RUN artifacts=$OUT"
