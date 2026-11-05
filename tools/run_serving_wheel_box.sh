#!/usr/bin/env bash
# Execute in a child shell. Never compiles or installs/replaces platform libraries.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TARGET="${TARGET:-ppu10}"
OUT="${OUT:-/workspace/actlize-la-wheel-${TARGET}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
unset PYTHONPATH
args=(--target "$TARGET" --out "$OUT")
if [[ -n "${FLARE_ROOT:-}" ]]; then args+=(--flare "$FLARE_ROOT"); fi
if [[ -n "${BLAS_LIBRARY:-}" ]]; then args+=(--blas "$BLAS_LIBRARY"); fi
if [[ "${HOST_ONLY:-0}" == 1 ]]; then args+=(--host-only); fi
exec "$PYTHON" -I "$ROOT/tools/qualify_serving_wheel.py" "${args[@]}"
