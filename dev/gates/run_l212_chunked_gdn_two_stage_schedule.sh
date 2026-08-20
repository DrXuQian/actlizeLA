#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${L212_OUT:-/workspace/actlizeLA-l212-gdn-two-stage}
mkdir -p "$OUT"

${CXX:-g++} -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$ROOT/include" \
  "$ROOT/dev/gates/l212_chunked_gdn_two_stage_schedule.cpp" \
  -o "$OUT/l212"
"$OUT/l212" | tee "$OUT/run.log"
