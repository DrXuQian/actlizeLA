#!/usr/bin/env bash
# Archive existing evidence, not current HEAD; refuse incomplete/reused bundles.
set -Eeuo pipefail
set -o noclobber
if [[ $# != 1 ]] || ! RUN="$(cd -- "$1" && pwd -P)"; then
  echo "usage: bash tools/pack_ppu10_paired_conversion.sh EXISTING_RUN_DIRECTORY" >&2
  exit 1
fi
files=(sha.txt binaries.sha256 comparison.json residual-correctness.log
       paired-conversion-edge.log paired-conversion-native.log l039_wy_paired_conversion.log)
for gate in -1.0 -0.1; do
  name="acu-paired-g$gate"
  if [[ -L "$RUN/$name" ]]; then
    echo "[paired conversion pack] FAIL: capture directory is a symlink: $name" >&2
    exit 1
  fi
  files+=("$name/$name.tar.gz")
done
for file in "${files[@]}"; do
  if [[ ! -f "$RUN/$file" || ! -s "$RUN/$file" || -L "$RUN/$file" ]]; then
    echo "[paired conversion pack] FAIL: missing/empty/nonregular evidence: $RUN/$file" >&2
    exit 1
  fi
done
measurement_sha="$(<"$RUN/sha.txt")"
if [[ ! "$measurement_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[paired conversion pack] FAIL: invalid measurement sha.txt" >&2
  exit 1
fi
manifest=paired-conversion.SHA256SUMS
archive="$RUN/paired-conversion.tar.gz"
if [[ -e "$archive" || -L "$archive" || -e "$RUN/$manifest" || -L "$RUN/$manifest" ]]; then
  echo "[paired conversion pack] FAIL: upload archive/manifest already exists; preserved: $RUN" >&2
  exit 1
fi
(cd "$RUN" && sha256sum -- "${files[@]}") > "$RUN/$manifest"
tar -czf - -C "$RUN" -- "$manifest" "${files[@]}" > "$archive"
sha256sum -- "$archive"
printf '[paired conversion pack] captures=2 measurement_sha=%s; upload only: %s\n' "$measurement_sha" "$archive"
