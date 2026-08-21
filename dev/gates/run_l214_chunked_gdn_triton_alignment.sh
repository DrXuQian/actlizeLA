#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/workspace/actlizeLA-l214-triton-alignment}"
mkdir -p "$OUT"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/include" \
  "$ROOT/dev/gates/l214_chunked_gdn_triton_alignment.cpp" \
  -o "$OUT/l214_chunked_gdn_triton_alignment"
"$OUT/l214_chunked_gdn_triton_alignment"
