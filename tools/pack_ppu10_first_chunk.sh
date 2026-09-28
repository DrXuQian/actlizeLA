#!/usr/bin/env bash
# Repackage an existing measured run; never substitute the current git SHA.
set -Eeuo pipefail
set -o noclobber
if [[ $# != 1 ]] || ! RUN="$(cd -- "$1" && pwd -P)"; then
  echo "usage: bash tools/pack_ppu10_first_chunk.sh EXISTING_RUN_DIRECTORY" >&2
  exit 1
fi
files=(sha.txt binaries.sha256 comparison.json residual-correctness.log
       first-chunk-edge.log first-chunk-native.log l036_wy_first_chunk.log)
for gate in -1.0 -0.1; do
  name="acu-first-g$gate"
  if [[ -L "$RUN/$name" ]]; then
    echo "[first chunk pack] FAIL: capture directory is a symlink: $name" >&2
    exit 1
  fi
  files+=("$name/$name.tar.gz")
done
for file in "${files[@]}"; do
  if [[ ! -f "$RUN/$file" || ! -s "$RUN/$file" || -L "$RUN/$file" ]]; then
    echo "[first chunk pack] FAIL: missing/empty/nonregular evidence: $RUN/$file" >&2
    exit 1
  fi
done
measurement_sha="$(<"$RUN/sha.txt")"
if [[ ! "$measurement_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[first chunk pack] FAIL: invalid measurement sha.txt" >&2
  exit 1
fi
manifest=first-chunk.SHA256SUMS
archive="$RUN/first-chunk.tar.gz"
if [[ -e "$archive" || -L "$archive" || -e "$RUN/$manifest" || -L "$RUN/$manifest" ]]; then
  echo "[first chunk pack] FAIL: upload archive/manifest already exists; preserved without overwrite: $RUN" >&2
  exit 1
fi
(cd "$RUN" && sha256sum -- "${files[@]}") > "$RUN/$manifest"
tar -czf - -C "$RUN" -- "$manifest" "${files[@]}" > "$archive"
sha256sum -- "$archive"
printf '[first chunk pack] captures=2 measurement_sha=%s; upload only: %s\n' "$measurement_sha" "$archive"
