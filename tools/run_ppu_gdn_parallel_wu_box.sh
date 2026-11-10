#!/usr/bin/env bash
# Run in a child shell; never exit the user's interactive Docker shell.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${INPUT_RUN:-}" ]]; then
  echo 'Set INPUT_RUN to the admitted ZW810 recheck directory (build-run/ + acu/).' >&2
  exit 2
fi
args=(--input-run "$INPUT_RUN" --device "${DEVICE:-0}" --jobs "${JOBS:-4}"
      --experiment "${EXPERIMENT:-parallel-wu}"
      --acu "${ACU:-acu}")
if [[ -n "${OUT:-}" ]]; then args+=(--out "$OUT"); fi
exec "${PYTHON:-python3}" "$ROOT/tools/collect_gdn_parallel_wu_acu.py" "${args[@]}"
