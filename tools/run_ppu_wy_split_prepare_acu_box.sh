#!/usr/bin/env bash
# One handoff: real build, unchanged device numerics, then three ACU captures.
# Public-API event timing is disabled; the user target is ACU sum only.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/gdn-wy-split-prepare-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ -e "$RUN" ]]; then
  echo "[WY split ACU] FAIL: output already exists: $RUN; choose a fresh OUT" >&2
  exit 1
fi
echo "[WY split ACU] target=FLA/subject>=1.5 metric=SUM-OF-ALL-ACU-KERNEL-DURATIONS"
echo "[WY split ACU] incumbent=aiu-state-output subject=split-prepare routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PERF=1 ADMISSION_ONLY=1 STATE_PIPELINE_AB=0 SPLIT_PREPARE_AB=1 AIU_AB=0 PREPARE_ROWS_AB=0 \
  STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
echo "[WY split ACU] API_TIMING=NOT_RUN; correctness complete, starting ACU comparison"
env -u EXTENSION OUT="$RUN/acu" bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" \
  --wy-run "$RUN" --wy-control aiu-state-output --wy-delivery split-prepare --gate -1.0
echo "[WY split ACU] completed; upload: $RUN/acu.tar.gz"
