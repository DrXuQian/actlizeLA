#!/usr/bin/env bash
# Same audited capture protocol; only the declared precision candidate changes.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT=solve-tf32x1
exec bash "$ROOT/tools/run_ppu_gdn_parallel_wu_box.sh"
