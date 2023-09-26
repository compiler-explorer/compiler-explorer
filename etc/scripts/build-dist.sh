#!/usr/bin/env bash

set -euo pipefail

ROOT="$(
  cd -- "$(dirname "$0")/../.." >/dev/null 2>&1
  pwd -P
)"
cd "$ROOT"

RELEASE_FILE_NAME=${GITHUB_RUN_NUMBER}
RELEASE_NAME=gh-${RELEASE_FILE_NAME}
HASH=$(git rev-parse HEAD)

# Clear the output
rm -rf out
mkdir -p out/dist
cd out/dist

echo "${HASH}" >git_hash
echo "${RELEASE_NAME}" >release_build

cp -R "${ROOT}"/etc \
      "${ROOT}"/examples \
      "${ROOT}"/views \
      "${ROOT}"/types \
      "${ROOT}"/package*.json \
      .
rm -rf "${ROOT}"/lib/storage/data

# Set up and build and webpack everything
cd "${ROOT}"
npm install --no-audit
npm run webpack
npm run ts-compile

# Now install only the production dependencies in our output directory
cd out/dist
npm install --no-audit --ignore-scripts --production
rm -rf node_modules/.cache/ node_modules/monaco-editor/
find node_modules -name \*.ts -delete

# Output some magic for GH to set the branch name and release name
echo "branch=${GITHUB_REF#refs/heads/}" >> "${GITHUB_OUTPUT}"
echo "release_name=${RELEASE_NAME}" >> "${GITHUB_OUTPUT}"

# Run to make sure we haven't just made something that won't work
node --no-warnings=ExperimentalWarning --loader ts-node/esm ./app.js --version --dist

rm -rf "${ROOT}/out/dist-bin"
mkdir -p "${ROOT}/out/dist-bin"
export XZ_OPT="-1 -T 0"
tar -Jcf "${ROOT}/out/dist-bin/${RELEASE_FILE_NAME}.tar.xz" .
pushd "${ROOT}/out/webpack"
tar -Jcf "${ROOT}/out/dist-bin/${RELEASE_FILE_NAME}.static.tar.xz" --transform="s,^static/,," static/*
popd
echo "${HASH}" >"${ROOT}/out/dist-bin/${RELEASE_FILE_NAME}.txt"
du -ch ./*
