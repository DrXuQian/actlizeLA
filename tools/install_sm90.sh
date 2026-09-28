#!/usr/bin/env bash
# Default: all SM90 candidates plus automatic shape dispatch; no GPU launch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIGURATION="${CONFIGURATION:-auto}"
if [[ "$CONFIGURATION" == auto ]]; then
  OUT="${OUT:-$ROOT/build/sm90}"
else
  OUT="${OUT:-$ROOT/build/sm90-$CONFIGURATION}"
fi
case "$CONFIGURATION" in
  auto|control|value64|value64-local-inverse|value128-paired) ;;
  *) echo "[actlizeLA install] unknown configuration: $CONFIGURATION" >&2; exit 2 ;;
esac
mkdir -p "$OUT/scratch"
export TMPDIR="$(cd "$OUT/scratch" && pwd)"
export PIP_NO_CACHE_DIR=1
GDN_QSA_TARGET=python "$PYTHON" -m pip install --no-build-isolation --no-deps -e "$ROOT"
if [[ "$CONFIGURATION" == auto ]]; then
  "$PYTHON" "$ROOT/tools/build_sm90_bundle.py" --out "$OUT" --register "$@"
  echo '[actlizeLA install] ready: from actlize_la import gdn_forward'
else
  "$PYTHON" "$ROOT/tools/build_gdn_sm90.py" --target cuda_sm90 \
    --configuration "$CONFIGURATION" --out "$OUT" "$@"
  "$PYTHON" - "$OUT" <<'PY'
import sys
from actlize_la import load_sm90
forward = load_sm90(sys.argv[1])
print(f"[actlizeLA install] PASS configuration={forward.configuration} binary={forward.extension}")
print(f"from actlize_la import load_sm90; forward = load_sm90({sys.argv[1]!r})")
print("Build/link/import only; no GPU kernel was launched.")
PY
fi
