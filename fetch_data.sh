#!/usr/bin/env bash
# Fetch source corpora at pinned commits so every run sees identical data.
set -euo pipefail
mkdir -p raw && cd raw

fetch () {  # $1=url $2=dir $3=commit
  if [ ! -d "$2" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "$2" rev-parse --short HEAD)"
}

fetch https://github.com/center-for-threat-informed-defense/tram.git tram \
      f29793d8d665f7f552898696e00065ef24a29a20
fetch https://github.com/boschresearch/anno-ctr-lrec-coling-2024.git annoctr \
      d510b6949e1938d47c93a43eedd562dc538439dc
