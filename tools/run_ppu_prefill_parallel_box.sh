#!/usr/bin/env bash
# Isolated candidate only; never install a wheel or replace a passed binary.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export PYTHONPATH="$ROOT:$ROOT/tests:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
# Resolve and validate the baseline BEFORE creating/building a candidate.
BASELINE_SERVING_RUN="$($PYTHON -c 'import os; from pathlib import Path; from collect_serving_prefill_acu import select_run; p=os.environ.get("BASELINE_SERVING_RUN"); print(select_run(Path(p) if p else None)["directory"])')"
export BASELINE_SERVING_RUN
SHA="$(git -C "$ROOT" rev-parse --short=8 HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/actlize-la-prefill-parallel-${SHA}-${STAMP}-$$}"
mkdir -p -- "$(dirname "$OUT")"
if ! mkdir -- "$OUT"; then
  echo "ERROR: OUT must be a new directory: $OUT" >&2
  exit 2
fi
TARGET=ppu10 PREFILL_TOKEN_PARALLEL=ON OUT="$OUT/candidate" \
  bash "$ROOT/tools/run_flare_serving_box.sh"
# Existing full numerical/graph gate AND old/new scratch parity must pass
# before the existing ACU collector may consume the candidate library.
SERVING_RUN="$OUT/candidate" OUT="$OUT/profile" \
  bash "$ROOT/tools/run_ppu_serving_prefill_acu_box.sh" \
    --lengths 2048 --value-heads 32 48 "$@"
echo "[prefill parallel] candidate gate: $OUT/candidate/results.tar.gz"
echo "[prefill parallel] ACU results: $OUT/profile/results.tar.gz"
echo "[prefill parallel] No change to installed/default release routing."
