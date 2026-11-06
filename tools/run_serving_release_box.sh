#!/usr/bin/env bash
# Child shell; build/install into a NEW private environment, never replace Torch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=(--target "${TARGET:-ppu10}" --sdk "${PPU_SDK:-/usr/local/PPU_SDK}" --jobs "${JOBS:-8}"
      --out "${OUT:-/workspace/actlize-la-serving-release-$(date -u +%Y%m%dT%H%M%SZ)-$$}")
if [[ -n "${FLARE_ROOT:-}" ]]; then args+=(--flare "$FLARE_ROOT"); fi
if [[ "${HOST_ONLY:-0}" == 1 ]]; then args+=(--host-only); fi
exec "${PYTHON:-python}" "$ROOT/tools/build_install_serving.py" "${args[@]}" "$@"
