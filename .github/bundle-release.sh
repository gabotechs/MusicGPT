#!/usr/bin/env bash

# WARNING: this script only works if executed from the root of the repo.


if [ -z "$1" ]; then
  echo "The archive name must be provided"
  exit 1
fi

ARCHIVE="$1"
BIN=target/release/musicgpt
LIBS=()

cargo build --release

# https://github.com/pykeio/ort dumps symlinked dynamic libs into
# target/release, so we must follow the links to the actual files.
for file in target/release/*.so*; do
  LIBS+=("$(readlink $file)")
done

tmpdir=$(mktemp -d)
mkdir -p "${tmpdir}/${ARCHIVE}" 

cp $BIN "${tmpdir}/${ARCHIVE}"

for lib in "${LIBS[@]}"; do
  cp $lib "${tmpdir}/${ARCHIVE}"
done

cwd=$(pwd)
pushd "${tmpdir}" >/dev/null
tar acf "${cwd}/${ARCHIVE}.tar.gz" "${ARCHIVE}"
popd >/dev/null
