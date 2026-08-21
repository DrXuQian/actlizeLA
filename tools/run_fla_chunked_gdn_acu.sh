#!/usr/bin/env bash
# Profile one FLA/Triton GDN forward at the same registered shape and fixture
# as actlizeLA's C++/PPU subject. Autotuning is completed into a private disk
# cache before ACU starts; a cache miss during the profiled process voids the
# report instead of silently mixing candidate launches into the measurement.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACU_BIN="${ACU_BIN:-/sim/eec/shared/junfu.qx/asight/bin/acu}"
SCOPE="${FLA_SCOPE:-post-cumsum}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
OUT="${OUT:-/workspace/actlizeLA-fla-gdn-acu-${SHA:0:8}-${STAMP}}"

case "$SCOPE" in
  full|post-cumsum) ;;
  *)
    echo "[FLA GDN ACU] FAIL: FLA_SCOPE must be full or post-cumsum" >&2
    exit 2
    ;;
esac

# FLA requires Python >=3.10. One known source of ``typing._ClassVar`` failures
# is an obsolete PyPI ``dataclasses`` backport shadowing the standard library.
# Reject that exact condition, but do not infer it merely from the exception:
# other packages can reach the same removed private symbol during lazy imports.
if ! "$PYTHON_BIN" - <<'PY'
import dataclasses
import pathlib
import sys

print(f"[FLA Python] executable={sys.executable} version={sys.version.split()[0]} dataclasses={dataclasses.__file__}")
if sys.version_info < (3, 10):
    raise SystemExit("FLA requires Python >=3.10; select the interpreter used by the working FLA installation with PYTHON_BIN")
path = pathlib.Path(dataclasses.__file__).resolve()
source = path.with_suffix(".py") if path.suffix == ".pyc" else path
try:
    text = source.read_text(errors="replace")
except OSError:
    text = ""
if "typing._ClassVar" in text:
    raise SystemExit(
        "obsolete dataclasses backport shadows the stdlib and accesses typing._ClassVar; "
        "select a clean FLA Python environment with PYTHON_BIN"
    )
PY
then
  echo "[FLA GDN ACU] FAIL: incompatible PYTHON_BIN=$PYTHON_BIN; no kernel was profiled" >&2
  exit 2
fi

# FLA_ROOT is an operator override, not a required identity field. With no
# override, measure the source authority actually imported by PYTHON_BIN.
if [[ -n "${FLA_ROOT:-}" ]]; then
  FLA_ROOT_SOURCE=operator
else
  FLA_ROOT="$($PYTHON_BIN - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("fla")
if spec is None or spec.origin is None:
    sys.exit(1)
print(pathlib.Path(spec.origin).resolve().parent.parent)
PY
  )" || {
    echo "[FLA GDN ACU] FAIL: PYTHON_BIN=$PYTHON_BIN cannot resolve the installed fla package; set FLA_ROOT only as an explicit fallback" >&2
    exit 2
  }
  FLA_ROOT_SOURCE=measured
fi
if [[ ! -d "$FLA_ROOT/fla" ]]; then
  echo "[FLA GDN ACU] FAIL: resolved FLA authority has no fla/ package: $FLA_ROOT (source=$FLA_ROOT_SOURCE)" >&2
  exit 2
fi
if [[ ! -x "$ACU_BIN" ]]; then
  echo "[FLA GDN ACU] FAIL: acu unavailable at $ACU_BIN" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  if [[ ! -d "$OUT" || -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[FLA GDN ACU] FAIL: OUT must be absent or an empty directory: $OUT" >&2
    exit 2
  fi
fi
mkdir -p "$OUT" "$OUT/triton-cache"

SUBJECT="$ROOT/tools/run_fla_chunked_gdn_acu_subject.py"
REPORT="$OUT/fla-${SCOPE}.report.acurep"
export PYTHONPATH="$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$OUT/triton-cache"
export TRITON_CACHE_AUTOTUNING=1
export FLA_CACHE_RESULTS=1
export FLA_CACHE_MODE=disabled
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_TILELANG=0
export FLA_FLASH_KDA=0
export FLA_INTRACARD_CP=0

subject_args=(
  "$SUBJECT"
  --device=cuda
  --sequences=1
  --length=2048
  --qk-heads=16
  --v-heads=32
  "--scope=$SCOPE"
)

{
  echo "actlizela_git_head=$SHA"
  echo "fla_git_head=$(git -C "$FLA_ROOT" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  echo "fla_root=$FLA_ROOT"
  echo "fla_root_source=$FLA_ROOT_SOURCE"
  echo "scope=$SCOPE"
  echo "shape=B1,T2048,H16,HV32,K128,V128,C64"
  echo "fixture=q:(i%7-3)/64,k:(i%5-2)/32,v:(i%9-4)/64,beta:0.5,gamma:-local/64,initial:(i%5-2)/1024,scale:0.5"
  echo "fla_disable_backend_dispatch=$FLA_DISABLE_BACKEND_DISPATCH"
  echo "triton_cache_dir=$TRITON_CACHE_DIR"
  "$PYTHON_BIN" -c 'import sys, torch, triton; print(f"python={sys.version.split()[0]}"); print(f"torch={torch.__version__}"); print(f"triton={triton.__version__}"); print(f"device={torch.cuda.get_device_name(0)!r}")'
  sha256sum "$SUBJECT" \
    "$FLA_ROOT/fla/ops/gated_delta_rule/chunk.py" \
    "$FLA_ROOT/fla/ops/gated_delta_rule/chunk_fwd.py" \
    "$FLA_ROOT/fla/ops/gated_delta_rule/wy_fast.py" \
    "$FLA_ROOT/fla/ops/common/chunk_delta_h.py" \
    "$FLA_ROOT/fla/ops/common/chunk_o.py"
} | tee "$OUT/identity.txt"

# First process: choose configurations and persist both compiled code and
# autotune timings. This process is deliberately outside ACU. Keep stderr in
# the artifact: lazy-import failures otherwise leave only their final line in
# copied logs, which is not enough to identify the responsible package.
set +e
TRITON_PRINT_AUTOTUNING=1 "$PYTHON_BIN" -X faulthandler "${subject_args[@]}" \
  2>&1 | tee "$OUT/prewarm.log"
prewarm_status=("${PIPESTATUS[@]}")
set -e
if [[ "${prewarm_status[0]}" -ne 0 || "${prewarm_status[1]}" -ne 0 ]]; then
  echo "[FLA GDN ACU] prewarm failed; locating private typing references in the exact PYTHON_BIN environment" \
    | tee "$OUT/python-private-typing-scan.log"
  "$PYTHON_BIN" - <<'PY' 2>&1 | tee -a "$OUT/python-private-typing-scan.log"
import dataclasses
import pathlib
import sys
import typing

print(f"python={sys.executable}")
print(f"typing={typing.__file__}")
print(f"dataclasses={dataclasses.__file__}")
roots = []
seen = set()
for entry in sys.path:
    if not entry:
        continue
    try:
        root = pathlib.Path(entry).resolve()
    except OSError:
        continue
    if root.is_dir() and root not in seen:
        roots.append(root)
        seen.add(root)

hits = []
for root in roots:
    for source in root.rglob("*.py"):
        try:
            text = source.read_text(errors="replace")
        except OSError:
            continue
        if "typing._ClassVar" in text:
            hits.append(source)

if hits:
    for source in sorted(set(hits)):
        print(f"typing_private_reference={source}")
else:
    print("typing_private_reference=NONE_IN_SYS_PATH")
    print("diagnosis=the complete traceback in prewarm.log is authoritative")
PY
  echo "[FLA GDN ACU] FAIL: FLA prewarm returned ${prewarm_status[0]}; no kernel was profiled; traceback=$OUT/prewarm.log" >&2
  exit 1
fi

expected=(
  chunk_gated_delta_rule_fwd_kkt_solve_kernel.autotune.json
  recompute_w_u_fwd_kernel.autotune.json
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64.autotune.json
  chunk_fwd_kernel_o.autotune.json
)
if [[ "$SCOPE" == full ]]; then
  expected+=(chunk_local_cumsum_scalar_kernel.autotune.json)
fi
for filename in "${expected[@]}"; do
  count="$(find "$TRITON_CACHE_DIR" -type f -name "$filename" -printf '.' | wc -c)"
  if [[ "$count" -ne 1 ]]; then
    echo "[FLA GDN ACU] FAIL: expected exactly one cached tuning decision for $filename, got $count" >&2
    exit 1
  fi
done
find "$TRITON_CACHE_DIR" -type f -name '*.autotune.json' -printf '%f\n' \
  | sort | tee "$OUT/autotune-cache-files.txt"

# Second process: ACU sees one forward call. Printing autotune decisions makes
# a cache miss mechanically visible; any such miss voids the report below.
TRITON_PRINT_AUTOTUNING=1 "$ACU_BIN" -f -o "$REPORT" --set full \
  "$PYTHON_BIN" "${subject_args[@]}" | tee "$OUT/acu.log"

if [[ ! -s "$REPORT" ]]; then
  echo "[FLA GDN ACU] FAIL: acu produced no nonempty report at $REPORT" >&2
  exit 1
fi
if grep -q 'Triton autotuning for function' "$OUT/acu.log"; then
  echo "[FLA GDN ACU] VOID: profiled process suffered an autotune cache miss" >&2
  exit 1
fi
if ! grep -q 'scope='"$SCOPE"'.*PASS$' "$OUT/acu.log"; then
  echo "[FLA GDN ACU] FAIL: subject did not publish a matching PASS witness" >&2
  exit 1
fi

echo "[FLA GDN ACU] PASS: report=$REPORT scope=$SCOPE timing=ACU-KERNEL-STAGES"
echo "[FLA GDN ACU] artifacts=$OUT"
