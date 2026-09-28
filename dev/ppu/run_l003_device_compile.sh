#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" &&
      -x /root/autodl-tmp/root-relocated/ppu-sdk/2.1.1/bin/hgcc ]]; then
  PPU_SDK_ROOT=/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1
fi
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" ]]; then
  echo "[ppu device compile] SKIP: hgcc unavailable at $PPU_SDK_ROOT/bin/hgcc"
  exit 77
fi
BUILD_DIR="${OUT:-/workspace/gdn-qsa-ppu-device-compile}" \
  PPU_SDK="$PPU_SDK_ROOT" bash "$ROOT/scripts/build_ppu.sh"
