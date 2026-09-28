#!/usr/bin/env bash
# Counter capture and tar packaging only; no kernel/tactic changes or sweep.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/tools/collect_ppu_gdn_acu.py" "$@"
