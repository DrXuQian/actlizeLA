#!/usr/bin/env bash
# 使用本机匹配SDK构建源码默认路由，完成原设备门并捕获登记矩阵。
# 不安装包/覆盖SDK/发布wheel/改频率；输出只能是新目录。
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${PPU_SDK:?请指定匹配Torch CUDA版本的真实PPU_SDK}"
ACU="${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
DEVICE="${DEVICE:-0}"
PHASE="${MATRIX_PHASE:-1}"
OUT="${OUT:-/workspace/actlize-la-gdn-box-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
export CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONUNBUFFERED=1
export PATH="$SDK/CUDA_SDK/bin:$SDK/bin:$PATH"
export CUDA_HOME="$SDK/CUDA_SDK" CUDA_PATH="$SDK/CUDA_SDK"
export LD_LIBRARY_PATH="$SDK/CUDA_SDK/lib64:$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT:$ROOT/tests${FLA_ROOT:+:$FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$OUT/scratch" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OUT/triton-cache}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
PYTHON="${PYTHON:-python3}"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --short
  "$SDK/bin/hgcc" --version
} > "$OUT/identity.txt"
# Check the selected profiler before compiling, with the same binding as capture.
# A bare --version here can parse a foreign SDK's sections before capture starts.
"$PYTHON" "$ROOT/tools/acu_installation.py" --acu "$ACU" --out "$OUT/profiler" \
  2>&1 | tee "$OUT/profiler.log"
git -C "$ROOT" diff HEAD --binary > "$OUT/source.patch"
"$PYTHON" -m actlize_la.serving.device --target ppu10 >> "$OUT/identity.txt"
cmake -S "$ROOT/csrc/serving" -B "$OUT/build" -DALA_TARGET=ppu10 \
  -DPPU_SDK_ROOT="$SDK" -DCMAKE_BUILD_TYPE=Release > "$OUT/build.log" 2>&1
cmake --build "$OUT/build" -j"${JOBS:-16}" >> "$OUT/build.log" 2>&1
LIB="$OUT/build/libactlize_la_serving.so"
"$PYTHON" "$ROOT/tests/serving_device.py" --target ppu10 --library "$LIB" \
  --results "$OUT/results.json" > "$OUT/device-tests.log" 2>&1
"$PYTHON" "$ROOT/tests/aux_device.py" --target ppu10 --library "$LIB" \
  --results "$OUT/aux-results.json" > "$OUT/aux-device.log" 2>&1
"$OUT/build/actlize_la_native_graph" > "$OUT/native-graph.log" 2>&1
"$PYTHON" "$ROOT/tools/describe_serving_binary.py" --target ppu10 --library "$LIB" \
  --source-sha "$(git -C "$ROOT" rev-parse HEAD)+source.patch" --output "$OUT/binary.json" > "$OUT/describe.log"
if [[ -n "${BASELINE_SERVING_RUN:-}" ]]; then
  "$PYTHON" "$ROOT/tests/prefill_parallel_device.py" --direct-pack \
    --baseline-run "$BASELINE_SERVING_RUN" --candidate "$LIB" \
    --results "$OUT/public-route.json" > "$OUT/public-route.log" 2>&1
fi
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
  printf '基线验证目录：%s\n' "$OUT"
  exit 0
fi
OUT="$OUT/captures" "$PYTHON" "$ROOT/tools/collect_serving_prefill_acu.py" \
  --serving-run "$OUT" --manifest "$ROOT/benchmarks/gdn_prefill_matrix.json" \
  --manifest-sha256 48a7aa907067cf2a966594c899da9b9882316950ddd09d39a292e4810f5d7402 \
  --manifest-phase "$PHASE" --device "$DEVICE" --acu "$ACU" --repeats 3
printf '产物目录：%s\n' "$OUT"
