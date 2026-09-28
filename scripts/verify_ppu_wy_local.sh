#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/workspace/gdn-wy-local}"
PPU_SDK_ROOT="${PPU_SDK:-/usr/local/PPU_SDK}"
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" && -x /root/autodl-tmp/root-relocated/ppu-sdk/2.1.1/bin/hgcc ]]; then
  PPU_SDK_ROOT=/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1
fi
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" || ! -x "$PPU_SDK_ROOT/bin/hgobjdump" ]]; then
  echo "[WY local] SKIP: real hgcc/hgobjdump unavailable at $PPU_SDK_ROOT"
  exit 77
fi
mkdir -p "$OUT"
cmake -S "$ROOT" -B "$OUT" -DGDN_QSA_ENABLE_PPU=ON -DPPU_SDK_ROOT="$PPU_SDK_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$OUT" --target _gdn_wy_ppu l004_ppu_affine_pipeline l006_ppu_original_delivery l007_ppu_wy_ownership l008_ppu_wy_delivery l009_ppu_wy_tiles l010_ppu_wy_state_address l011_ppu_wy_stage_address l012_ppu_wy_prepare_rows l016_ppu_wy_aiu_pair l018_ppu_wy_split_prepare l019_ppu_wy_state_pipeline -j"${JOBS:-8}"
ctest --test-dir "$OUT" --output-on-failure
python "$ROOT/tests/test_ppu_wy_algebra.py"
python "$ROOT/tests/test_ppu_wy_contract.py"
python "$ROOT/dev/ppu/check_original_structure.py" --self-test
python "$ROOT/dev/ppu/check_split_prepare.py" --self-test
python "$ROOT/dev/ppu/check_state_pipeline.py" --self-test
python "$ROOT/tests/test_ppu_fla_benchmark_contract.py"
python "$ROOT/tests/test_ppu_gdn_acu_contract.py"
python "$ROOT/tests/test_ppu_hgcc_arch.py"
"$PPU_SDK_ROOT/bin/hgobjdump" --arch=ppu1.0 --dump-isa "$OUT/libgdn_wy_ppu.so" > "$OUT/gdn_wy_ppu.isa"
"$PPU_SDK_ROOT/bin/hgobjdump" --dump-resource-usage=all "$OUT/libgdn_wy_ppu.so" > "$OUT/gdn_wy_ppu.resources"
python "$ROOT/dev/ppu/check_wy_binary.py" "$OUT" --self-test
BUILD_DIR="$OUT" PPU_SDK="$PPU_SDK_ROOT" JOBS="${JOBS:-8}" bash "$ROOT/scripts/build_ppu.sh"
echo "[WY local] PASS; device numerics/performance NOT_RUN"
