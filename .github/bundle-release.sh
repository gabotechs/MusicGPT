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

if [ ! -f $RELEASE_DIR/$BIN ]; then
  RELEASE_DIR=target/$ARCHIVE/release
fi

if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "Could not find target release dir in $RELEASE_DIR or target/release"
  exit 1
fi


tmpdir=$(mktemp -d)
mkdir -p "${tmpdir}/${ARCHIVE}/lib"

echo "Copying file ${RELEASE_DIR}/${BIN} to ${tmpdir}/${ARCHIVE}..."
cp "${RELEASE_DIR}/${BIN}" "${tmpdir}/${ARCHIVE}"

cwd=$(pwd)
pushd "${tmpdir}" >/dev/null
tar acf "${cwd}/${ARCHIVE}.tar.gz" "${ARCHIVE}"
popd >/dev/null
