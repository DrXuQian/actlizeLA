#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/workspace/actlizeLA-local-gates}"

mkdir -p "$OUT"

L203_BUILD_ROOT="$OUT/l203" \
  bash "$ROOT/dev/gates/run_l203_chunked_gdn_oracle.sh"
QUACTLIZE_L204_OUT="$OUT/l204" \
  bash "$ROOT/dev/gates/run_l204_chunked_gdn_device_compile.sh"
QUACTLIZE_L206_OUT="$OUT/l206" \
  bash "$ROOT/dev/gates/run_l206_chunked_gdn_global_dot_ownership.sh"
QUACTLIZE_L207_OUT="$OUT/l207" \
  bash "$ROOT/dev/gates/run_l207_chunked_gdn_resident_mma_ownership.sh"
QZ_GDN_SKIP_CUDA="${QZ_GDN_SKIP_CUDA:-0}" OUT="$OUT/l205" \
  bash "$ROOT/dev/gates/run_l205_ppu_chunked_gdn_abi.sh" --local

echo "[actlizeLA local] PASS: forward algebra, ABI, device type, and ownership gates"
