#!/usr/bin/env bash
set -eEuo pipefail

# WARNING: this script only works if executed from the root of the repo.

if [ -z "$1" ]; then
  echo "The archive name must be provided"
  exit 1
fi

ARCHIVE="$1"
BIN=musicgpt
RELEASE_DIR=target/release
LIBS=()

if [ ! -f $RELEASE_DIR/$BIN ]; then
  RELEASE_DIR=target/$ARCHIVE/release
fi

if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "Could not find target release dir in $RELEASE_DIR or target/release"
  exit 1
fi

# https://github.com/pykeio/ort dumps symlinked dynamic libs into
# target/release, so we must follow the links to the actual files.
for file in $(ls $RELEASE_DIR); do
  if [[ "$file" == *.so* ]]; then
    LIBS+=("$(readlink $RELEASE_DIR/$file)")
  fi
done


tmpdir=$(mktemp -d)
mkdir -p "${tmpdir}/${ARCHIVE}/lib"

cp "${RELEASE_DIR}/${BIN}" "${tmpdir}/${ARCHIVE}"

for lib in "${LIBS[@]}"; do
  cp $lib "${tmpdir}/${ARCHIVE}/lib"
done

cwd=$(pwd)
pushd "${tmpdir}" >/dev/null
tar acf "${cwd}/${ARCHIVE}.tar.gz" "${ARCHIVE}"
popd >/dev/null
