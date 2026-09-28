#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/workspace/gdn-qsa-ppu-local}"
JOBS="${JOBS:-16}"

mkdir -p "$OUT"
cmake -S "$ROOT" -B "$OUT" -DBUILD_TESTING=ON
cmake --build "$OUT" -j"$JOBS"
ctest --test-dir "$OUT" --output-on-failure
"$OUT/l004_ppu_affine_pipeline"
"$OUT/l006_ppu_original_delivery"
python "$ROOT/dev/ppu/check_original_structure.py" --self-test
python "$ROOT/dev/ppu/check_retile_failure_signature.py"
python "$ROOT/tests/test_ppu_fla_benchmark_contract.py"
python "$ROOT/tests/test_ppu_gdn_acu_contract.py"
python "$ROOT/tests/test_ppu_hgcc_arch.py"

OUT="$OUT/l003" \
  bash "$ROOT/dev/ppu/run_l003_device_compile.sh"
