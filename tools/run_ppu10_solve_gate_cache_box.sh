#!/usr/bin/env bash
# Build/admit once; compare the same three arms at both decay settings.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-static-solve-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"

# The existing runner enforces a fresh output directory and numerical gates.
OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" GATE=-1.0 \
  CANDIDATE=residual-gate-cache-solve-static \
  bash "$ROOT/tools/run_ppu_residual_delivery_acu_box.sh"

# comparison.json already admitted both gates. Reuse its exact binary and
# fixtures instead of rebuilding an independently named weak-decay candidate.
env -u EXTENSION OUT="$RUN/acu-weak" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
  bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
  --wy-control residual-gate-cache --wy-delivery residual-gate-cache-solve-static --gate -0.1
printf '[PPU1.0 solve] numerical gates and both captures complete; performance requires report analysis\n'
printf 'strong-decay bundle: %s/acu/acu.tar.gz\nweak-decay bundle: %s/acu-weak/acu-weak.tar.gz\n' "$RUN" "$RUN"
