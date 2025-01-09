#!/usr/bin/env bash
set -eEuo pipefail

# WARNING: this script only works if executed from the root of the repo.

TARGET="$1"
VERSION="$2"

BIN=musicgpt
RELEASE_DIR=target/release
BUILD_HASH=$(cat $BUILD_HASH_FILE)

if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "No binary found in $RELEASE_DIR/$BIN"
  RELEASE_DIR=target/$TARGET/release
  echo "Searching for $RELEASE_DIR/$BIN instead..."
else
  echo "Binary found in $RELEASE_DIR/$BIN"
fi

if [ ! -f $RELEASE_DIR/$BIN ]; then
  echo "No binary found in $RELEASE_DIR/$BIN, aborting"
  exit 1
else
  echo "Binary found in $RELEASE_DIR/$BIN"
fi

echo "Moving $RELEASE_DIR/$BIN to $BIN-$TARGET..."
mv "$RELEASE_DIR/$BIN" "$BIN-$TARGET"
echo "Upload $BIN-$TARGET file to github release v$VERSION..."

if [  -f "$BIN-$TARGET.exe" ]; then
  gh release upload "v$VERSION" "$BIN-$TARGET.exe"
else
  gh release upload "v$VERSION" "$BIN-$TARGET"
fi

pushd "$ONNXRUNTIME_BUILD_DIR/$BUILD_HASH" >/dev/null
for file in $(ls *.{so,dylib,dll} 2> /dev/null); do
  echo "Moving $file to $TARGET-$file..."
  mv "$file" "$TARGET-$file"
  echo "Uploading $TARGET-$file to github release v$VERSION..."
  gh release upload "v$VERSION" "$TARGET-$file"
done
echo "Done!"
popd >/dev/null
