#!/usr/bin/env bash
# Reference arm on the exact saved operands from the preceding core capture.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export TOKENS="${TOKENS:-8192}"
INPUT_RUN="$($PYTHON -c 'import os; from collect_serving_prefill_acu import select_core_input_run; print(select_core_input_run(os.environ.get("INPUT_RUN"), length=int(os.environ["TOKENS"]))["directory"])')"
export INPUT_RUN ROLE=fla
export OUT="${OUT:-/workspace/flashnext-gdn-fla-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
exec bash "$ROOT/tools/run_ppu_flashnext_core_acu_box.sh"
