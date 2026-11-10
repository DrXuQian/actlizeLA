#!/usr/bin/env bash
# Same raw-input numerical admission, then exactly one public core call in ACU.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ROLE="${ROLE:-default}"
case "$ROLE" in
  default) NAME=core ;;
  fla) NAME=fla ;;
  *) echo "ERROR: ROLE must be default or fla" >&2; exit 2 ;;
esac
export PYTHONPATH="$ROOT:$ROOT/tests:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
SDK="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACU="$(command -v -- "${ACU:-acu}")"
ACU="$(readlink -f -- "$ACU")"
ASIGHT_HOME="$(dirname "$(dirname "$ACU")")"
export ASIGHT_HOME
if [[ -n "${INPUT_RUN:-}" ]]; then
  LIB="$($PYTHON -c 'import os; from collect_serving_prefill_acu import core_input_run; print(core_input_run(os.environ["INPUT_RUN"], int(os.environ.get("TOKENS", "2048")))["library"])')"
else
  LIB="$($PYTHON -c 'import os; from pathlib import Path; from collect_serving_prefill_acu import select_run; p=os.environ.get("SERVING_RUN"); print(select_run(Path(p) if p else None)["library"])')"
fi
OUT="${OUT:-/workspace/flashnext-gdn-core-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p -- "$(dirname "$OUT")"
mkdir -- "$OUT"
finish() {
  local rc=$?
  trap - EXIT
  local files=() name
  for name in admission.json input-origin.json "$NAME.json" admission.log "$NAME-acu.log" "$NAME.csv" kernel-sum.json sections.log; do
    if [[ -f "$OUT/$name" ]]; then files+=("$name"); fi
  done
  if ((${#files[@]})); then tar -C "$OUT" -czf "$OUT/results.tar.gz" -- "${files[@]}"; fi
  if ((rc!=0)); then
    echo "[Flash-Next core ACU] NOT_ADMITTED rc=$rc; inspect $OUT"
    for name in sections.log admission.log "$NAME-acu.log"; do
      if [[ -f "$OUT/$name" ]]; then tail -15 "$OUT/$name"; fi
    done
  fi
  echo "[Flash-Next core ACU] artifacts=$OUT upload=$OUT/results.tar.gz"
  exit "$rc"
}
trap finish EXIT
"$ACU" --section-folder "$ASIGHT_HOME/sections" --set full --list-sections >"$OUT/sections.log" 2>&1
INPUTS="${INPUT_RUN:-$OUT}"
ARGS=(--library "$LIB" --directory "$INPUTS" --length "${TOKENS:-2048}"
      --value-heads 48 --gate=-0.1)
if [[ -n "${INPUT_RUN:-}" ]]; then
  "$PYTHON" - "$INPUT_RUN" "$OUT" "${TOKENS:-2048}" <<'PY'
import json, shutil, sys
from pathlib import Path
from collect_serving_prefill_acu import core_input_run
origin = core_input_run(sys.argv[1], int(sys.argv[3]))
out = Path(sys.argv[2])
shutil.copyfile(Path(origin['directory'])/'admission.json', out/'admission.json')
(out/'input-origin.json').write_text(json.dumps(origin, indent=2)+'\n')
print('[Flash-Next core ACU] reusing exact saved operands: ' + origin['directory'])
PY
else
  "$PYTHON" -u "$ROOT/benchmarks/profile_serving_prefill.py" "${ARGS[@]}" \
    --phase admit --receipt "$OUT/admission.json" >"$OUT/admission.log" 2>&1
fi
"$ACU" --section-folder "$ASIGHT_HOME/sections" -f -o "$OUT/$NAME" --set full --check-exit-code yes \
  "$PYTHON" -u "$ROOT/benchmarks/profile_serving_prefill.py" "${ARGS[@]}" \
  --phase subject --role "$ROLE" --receipt "$OUT/$NAME.json" >"$OUT/$NAME-acu.log" 2>&1
"$PYTHON" - "$OUT" "$ACU" "$ROLE" "$NAME" <<'PY'
import json, os, sys
from pathlib import Path
from collect_serving_prefill_acu import inventory, parse_raw, validate_receipts
from collect_ppu_gdn_acu import report_file, run, sha
out, acu = Path(sys.argv[1]), Path(sys.argv[2])
role, name = sys.argv[3:5]
validate_receipts(json.loads((out/'admission.json').read_text()),
                  json.loads((out/f'{name}.json').read_text()), role)
report = report_file(out/name)
run([acu, '--section-folder', acu.parent.parent/'sections', '--import', report,
     '--csv', '--page', 'raw'], out/f'{name}.csv', os.environ.copy(), console=False)
measured = inventory(role, parse_raw((out/f'{name}.csv').read_text()))
measured.update(role=role, report_sha256=sha(report), repeats=1, scope='single-ACU-core-call-not-API-latency')
(out/'kernel-sum.json').write_text(json.dumps(measured, indent=2)+'\n')
for kernel in measured['kernels']:
    print(f"{kernel['us']:.5f} us  {kernel['kernel']}")
print(f"[Flash-Next core ACU] role={role} sum={measured['total_us']:.5f} us inventory={measured['status']}")
if measured['status'] != 'ADMITTED':
    raise SystemExit(3)
PY
