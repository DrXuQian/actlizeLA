#!/usr/bin/env bash
# Package an existing run without building, profiling or changing its SHA.
set -Eeuo pipefail
set -o noclobber
if [[ $# != 1 ]] || ! RUN="$(cd -- "$1" && pwd -P)"; then
  echo "usage: bash tools/pack_ppu10_full_chunk_stages.sh EXISTING_RUN_DIRECTORY" >&2
  exit 1
fi
files=(sha.txt binaries.sha256 comparison.json residual-correctness.log)
for stage in solve output both; do
  for gate in -1.0 -0.1; do
    name="acu-$stage-g$gate"
    if [[ -L "$RUN/$name" ]]; then
      echo "[full stages pack] FAIL: capture directory is a symlink: $name" >&2
      exit 1
    fi
    files+=("$name/$name.tar.gz")
  done
done
for file in "${files[@]}"; do
  if [[ ! -f "$RUN/$file" || ! -s "$RUN/$file" || -L "$RUN/$file" ]]; then
    echo "[full stages pack] FAIL: missing/empty/nonregular evidence: $RUN/$file" >&2
    exit 1
  fi
done
measurement_sha="$(<"$RUN/sha.txt")"
if [[ ! "$measurement_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[full stages pack] FAIL: invalid measurement sha.txt" >&2
  exit 1
fi
manifest=full-chunk-stages.SHA256SUMS
archive="$RUN/full-chunk-stages.tar.gz"
if [[ -e "$archive" || -L "$archive" || -e "$RUN/$manifest" || -L "$RUN/$manifest" ]]; then
  echo "[full stages pack] FAIL: upload archive/manifest already exists; preserved without overwrite: $RUN" >&2
  exit 1
fi
(cd "$RUN" && sha256sum -- "${files[@]}") > "$RUN/$manifest"
tar -czf - -C "$RUN" -- "$manifest" "${files[@]}" > "$archive"
sha256sum -- "$archive"
printf '[full stages pack] captures=6 measurement_sha=%s; upload only: %s\n' "$measurement_sha" "$archive"
