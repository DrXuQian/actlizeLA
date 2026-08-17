#!/usr/bin/env bash
set -Eeuo pipefail

out=""
while (($#)); do
  case "$1" in
    -o)
      out="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$out" ]]; then
  echo "[l210 fake hgcc] FAIL: missing -o" >&2
  exit 2
fi

# This plant models the exact box failure mode: a nominally successful device
# command that leaves the linker's declared external object absent.
if [[ "${L210_DROP_OBJECT:-0}" == 1 ]]; then
  exit 0
fi

exec /usr/bin/cc -fPIC -c "${L210_FIXTURE_SOURCE:?}" -o "$out"
