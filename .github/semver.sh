#!/usr/bin/env bash

set -eu

BUMP=0
VERSION=""

while IFS= read -r line; do
  if [[ $line == *BREAKING\ CHANGE* ]]; then
    if [[ $BUMP -lt 3 ]]; then
      BUMP=3
      VERSION="major"
    fi
  elif [[ $line == *feat:* || $line == *feat\(*\):* ]]; then
    if [[ $BUMP -lt 2 ]]; then
      BUMP=2
      VERSION="minor"
    fi
  elif [[ $line != "" ]]; then
    if [[ $BUMP -lt 1 ]]; then
      BUMP=1
      VERSION="patch"
    fi
  fi
done <<< "$(git log --oneline $(git describe --tags --abbrev=0 @^)..@ | cat | awk '{print}')"

if [[ $VERSION == "" ]]; then
  echo "There is nothing new to release"
  exit 1
fi

echo $VERSION
