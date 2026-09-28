#!/usr/bin/env bash
# Python frontend plus exactly one selected SM90a configuration; no GPU launch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIGURATION="${CONFIGURATION:-value64}"
OUT="${OUT:-$ROOT/build/sm90-$CONFIGURATION}"
case "$CONFIGURATION" in
  control|value64|value64-local-inverse|value128-paired) ;;
  *) echo "[actlizeLA install] unknown configuration: $CONFIGURATION" >&2; exit 2 ;;
esac
GDN_QSA_TARGET=python "$PYTHON" -m pip install --no-build-isolation --no-deps -e "$ROOT"
"$PYTHON" "$ROOT/tools/build_gdn_sm90.py" --target cuda_sm90 \
  --configuration "$CONFIGURATION" --out "$OUT"
"$PYTHON" - "$OUT" <<'PY'
import sys
from actlize_la import load_sm90
forward = load_sm90(sys.argv[1])
print(f"[actlizeLA install] PASS configuration={forward.configuration} binary={forward.extension}")
print(f"from actlize_la import load_sm90; forward = load_sm90({sys.argv[1]!r})")
print("Build/link/import only; no GPU kernel was launched.")
PY
