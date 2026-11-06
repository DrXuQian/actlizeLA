#!/usr/bin/env bash
# Child shell only. Reuse a passed serving binary; do not rebuild kernels.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/tools/collect_serving_prefill_acu.py" "$@"
