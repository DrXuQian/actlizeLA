#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/workspace/actlizeLA-local-gates}"

mkdir -p "$OUT"

L211_OUT="$OUT/l211" \
  bash "$ROOT/dev/gates/run_l211_actlize_extensions_install.sh"
L203_BUILD_ROOT="$OUT/l203" \
  bash "$ROOT/dev/gates/run_l203_chunked_gdn_oracle.sh"
QUACTLIZE_L204_OUT="$OUT/l204" \
  bash "$ROOT/dev/gates/run_l204_chunked_gdn_device_compile.sh"
QUACTLIZE_L206_OUT="$OUT/l206" \
  bash "$ROOT/dev/gates/run_l206_chunked_gdn_global_dot_ownership.sh"
QUACTLIZE_L207_OUT="$OUT/l207" \
  bash "$ROOT/dev/gates/run_l207_chunked_gdn_resident_mma_ownership.sh"
QUACTLIZE_L208_OUT="$OUT/l208" \
  bash "$ROOT/dev/gates/run_l208_chunked_gdn_fragment_delivery.sh"
QUACTLIZE_L209_OUT="$OUT/l209" \
  bash "$ROOT/dev/gates/run_l209_chunked_gdn_blocked_inverse.sh"
L210_OUT="$OUT/l210" \
  bash "$ROOT/dev/gates/run_l210_ppu_device_object_graph.sh"
QZ_GDN_SKIP_CUDA="${QZ_GDN_SKIP_CUDA:-0}" OUT="$OUT/l205" \
  bash "$ROOT/dev/gates/run_l205_ppu_chunked_gdn_abi.sh" --local

echo "[actlizeLA local] PASS: canonical installed headers, forward algebra, ABI, device type, ownership, fragment delivery, and TF32 inverse gates"
