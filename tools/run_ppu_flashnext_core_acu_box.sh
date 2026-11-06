#!/usr/bin/env bash
# Same raw-input numerical admission, then exactly one public core call in ACU.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT:$ROOT/tests:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASIGHT_HOME=/sim/eec/shared/junfu.qx/asight
ACU="$ASIGHT_HOME/bin/acu"
LIB="$($PYTHON -c 'import os; from pathlib import Path; from collect_serving_prefill_acu import select_run; p=os.environ.get("SERVING_RUN"); print(select_run(Path(p) if p else None)["library"])')"
OUT="${OUT:-/workspace/flashnext-gdn-core-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
finish() {
  local rc=$?
  trap - EXIT
  local files=() name
  for name in admission.json core.json admission.log core-acu.log core.csv kernel-sum.json sections.log; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  if ((rc!=0)); then
    echo "[Flash-Next core ACU] NOT_ADMITTED rc=$rc; inspect $OUT"
    for name in sections.log admission.log core-acu.log; do
      if [[ -f "$OUT/$name" ]]; then tail -15 "$OUT/$name"; fi
    done
  fi
  echo "[Flash-Next core ACU] artifacts=$OUT upload=$OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT
"$ACU" --section-folder "$ASIGHT_HOME/sections" --set full --list-sections >"$OUT/sections.log" 2>&1
ARGS=(--library "$LIB" --directory "$OUT" --length "${TOKENS:-2048}"
      --value-heads 48 --gate=-0.1)
"$PYTHON" -u "$ROOT/benchmarks/profile_serving_prefill.py" "${ARGS[@]}" \
  --phase admit --receipt "$OUT/admission.json" >"$OUT/admission.log" 2>&1
"$ACU" --section-folder "$ASIGHT_HOME/sections" -f -o "$OUT/core" --set full --check-exit-code yes \
  "$PYTHON" -u "$ROOT/benchmarks/profile_serving_prefill.py" "${ARGS[@]}" \
  --phase subject --role default --receipt "$OUT/core.json" >"$OUT/core-acu.log" 2>&1
"$PYTHON" - "$OUT" "$ACU" <<'PY'
import json, os, sys
from pathlib import Path
from collect_serving_prefill_acu import inventory, parse_raw, validate_receipts
from collect_ppu_gdn_acu import report_file, run, sha
out, acu = Path(sys.argv[1]), Path(sys.argv[2])
validate_receipts(json.loads((out/'admission.json').read_text()),
                  json.loads((out/'core.json').read_text()), 'default')
report = report_file(out/'core')
run([acu, '--section-folder', acu.parent.parent/'sections', '--import', report,
     '--csv', '--page', 'raw'], out/'core.csv', os.environ.copy(), console=False)
measured = inventory('default', parse_raw((out/'core.csv').read_text()))
measured.update(report_sha256=sha(report), repeats=1, scope='single-ACU-core-call-not-API-latency')
(out/'kernel-sum.json').write_text(json.dumps(measured, indent=2)+'\n')
for kernel in measured['kernels']:
    print(f"{kernel['us']:.5f} us  {kernel['kernel']}")
print(f"[Flash-Next core ACU] sum={measured['total_us']:.5f} us inventory={measured['status']}")
if measured['status'] != 'ADMITTED':
    raise SystemExit(3)
PY
