#!/usr/bin/env bash
set -eEuo pipefail

# WARNING: this script only works if executed from the root of the repo.

TARGET="$1"
VERSION="$2"

BIN=musicgpt
RELEASE_DIR=target/release
BUILD_HASH=$(cat $BUILD_HASH_FILE)

if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "No binary found in $RELEASE_DIR/$BIN, searching somewhere else..."
  RELEASE_DIR=target/$TARGET/release
fi
if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "No binary found in $RELEASE_DIR/$BIN"
  exit 1
fi
mv $RELEASE_DIR/$BIN $BIN-$TARGET
gh release upload v$VERSION $BIN-$TARGET

pushd $ONNXRUNTIME_BUILD_DIR/$BUILD_HASH >/dev/null
for file in $(ls *.{so,dylib,dll} 2> /dev/null); do
  mv $file $TARGET-$file
  gh release upload v$VERSION $TARGET-$file
done
popd >/dev/null
