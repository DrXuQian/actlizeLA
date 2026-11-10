#!/usr/bin/env bash
# One conversion-only round; the existing capture protocol and thresholds stay.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT=native-bf16
exec bash "$ROOT/tools/run_ppu_gdn_parallel_wu_box.sh"
