#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/workspace/actlizeLA-l213-four-stage-schedule}"
mkdir -p "$OUT"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/include" \
  "$ROOT/dev/gates/l213_chunked_gdn_four_stage_schedule.cpp" \
  -o "$OUT/l213_chunked_gdn_four_stage_schedule"
"$OUT/l213_chunked_gdn_four_stage_schedule"
