#!/usr/bin/env bash
# One build/admission, then both decay settings against the measured control.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
RUN="${OUT:-/workspace/actlizeLA-ppu10-full-chunk-${SHA}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"

OUT="$RUN" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" GATE=-1.0 \
  CANDIDATE=residual-full-chunk \
  bash "$ROOT/tools/run_ppu_residual_delivery_acu_box.sh"

# Reuse the exact admitted binary, inputs and compiler for the second capture.
env -u EXTENSION OUT="$RUN/acu-weak" PPU_SDK="$PPU_SDK_ROOT" ACU="$ACU" \
  bash "$ROOT/tools/run_ppu_gdn_fla_acu_box.sh" --wy-run "$RUN" \
  --wy-control residual-gate-cache-solve-static --wy-delivery residual-full-chunk --gate -0.1
printf '[PPU1.0 full chunk] gates and both captures complete; speed requires ACU analysis\n'
printf 'strong-decay bundle: %s/acu/acu.tar.gz\nweak-decay bundle: %s/acu-weak/acu-weak.tar.gz\n' "$RUN" "$RUN"
