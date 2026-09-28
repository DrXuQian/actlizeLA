#!/usr/bin/env bash
# Same-input old split/new state-pipeline/FLA: numeric admission, then ACU.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/gdn-wy-state-pipeline-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ -e "$RUN" ]]; then
  echo "[WY pipeline ACU] FAIL: output already exists: $RUN; choose a fresh OUT" >&2
  exit 1
fi
echo "[WY pipeline ACU] target=FLA/subject>=1.5 metric=SUM-OF-ALL-ACU-KERNEL-DURATIONS"
echo "[WY pipeline ACU] incumbent=split-prepare subject=state-pipeline routing=UNCHANGED"
env -u SAMPLES OUT="$RUN" PERF=1 ADMISSION_ONLY=1 STATE_PIPELINE_AB=1 SPLIT_PREPARE_AB=0 \
  AIU_AB=0 PREPARE_ROWS_AB=0 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  bash "$ROOT/tools/run_ppu_wy_fla_box.sh"
echo "[WY pipeline ACU] API_TIMING=NOT_RUN; correctness complete, starting ACU comparison"
env -u EXTENSION OUT="$RUN/acu" bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" \
  --wy-run "$RUN" --wy-control split-prepare --wy-delivery state-pipeline --gate -1.0
echo "[WY pipeline ACU] completed; upload: $RUN/acu.tar.gz"
