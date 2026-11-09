#!/usr/bin/env bash
# Child shell only; caller's interactive Docker shell is never exited.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${INPUT_RUN:-}" ]]; then
  echo 'Set INPUT_RUN to the previous ZW810 recheck directory (contains build-run/ and acu/).' >&2
  exit 2
fi
args=(--input-run "$INPUT_RUN" --device "${DEVICE:-0}" --jobs "${JOBS:-4}"
      --acu "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}")
if [[ -n "${OUT:-}" ]]; then args+=(--out "$OUT"); fi
if [[ -n "${REUSE_BUILD_RUN:-}" ]]; then args+=(--pla-build-run "$REUSE_BUILD_RUN"); fi
exec "${PYTHON:-python3}" "$ROOT/tools/collect_gdn_threeway_acu.py" "${args[@]}"
