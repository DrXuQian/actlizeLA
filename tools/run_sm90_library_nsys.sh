#!/usr/bin/env bash
# Physical H800 only. Install pinned references/dependencies before invoking.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${WORK:?set the prepared reference workspace}"
: "${FAMILY:?flashqla or flashinfer}"
: "${GATE:?natural-log gate -0.1 or -1.0}"
: "${OUT:?set a fresh result directory}"
: "${CUDA_EXTENSION:?set the admitted CUDA SM90 binary}"
: "${PPU_SOURCE_EXTENSION:?set the admitted PPU-fork CUDA source-check binary}"
case "$FAMILY" in
  flashqla) PYTHON="$WORK/venv/bin/python"; REF="$WORK/FlashQLA" ;;
  flashinfer) PYTHON="$WORK/fi-venv/bin/python"; REF="$WORK/flashinfer" ;;
  *) echo "unknown reference family: $FAMILY" >&2; exit 2 ;;
esac
NSYS=${NSYS:-/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys}
[[ -d "$WORK" && -x "$PYTHON" && -x "$NSYS" && -d "$REF" ]]
[[ ! -e "$OUT" ]] || { echo "refuse to overwrite result directory: $OUT" >&2; exit 2; }
mkdir -p -- "$OUT" "$WORK/scratch" "$WORK/cache"
export TMPDIR="$WORK/scratch"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TILELANG_CACHE_DIR="$WORK/cache/tilelang"
export TILELANG_TMP_DIR="$WORK/scratch/tilelang"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$WORK/cache"
export CUTE_DSL_CACHE_DIR="$WORK/cache/cute-dsl"
export CUTE_DSL_DUMP_DIR="$WORK/cache/cute-dsl-codegen"
export CUTE_DSL_KEEP=ptx,cubin
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o "$OUT/forward" \
  "$PYTHON" -u "$ROOT/benchmarks/profile_sm90_libraries.py" \
    --family "$FAMILY" --reference-root "$REF" \
    --source-archive "$WORK/$FAMILY-source.tar.gz" \
    --cuda-extension "$CUDA_EXTENSION" --ppu-source-extension "$PPU_SOURCE_EXTENSION" \
    --gate "$GATE" --out "$OUT" 2>&1 | tee "$OUT/run.log"
"$NSYS" export --type sqlite --output "$OUT/forward.sqlite" "$OUT/forward.nsys-rep"
"$PYTHON" "$ROOT/tools/analyze_sm90_nsys.py" --sqlite "$OUT/forward.sqlite" \
  --receipt "$OUT/receipt.json" --out "$OUT/result.json" | tee "$OUT/summary.log"
