#!/bin/bash
# usage: make-test-root.sh DEST_ETC
# Copies etc/config into DEST_ETC/config with the local properties' caches, storage, stats and Sentry made local, so
# a test instance can't write to the prod S3 cache. Turns on logging of compilation cache accesses.
set -euo pipefail

dest=${1:?usage: make-test-root.sh DEST_ETC}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)

mkdir -p "${dest}/config"
cp -r "${repo}"/etc/config/. "${dest}/config/"

props="${dest}/config/compiler-explorer.local.properties"
touch "${props}"
sed -i -E '/^(cacheConfig|executableCacheConfig|compilerCacheConfig|storageSolution|compilationStatsNotifier|sentryDsn|cookieDomainRe|logCompilationCacheAccesses)=/d' "${props}"
cat >> "${props}" <<'PROPS'
cacheConfig=InMemory(50)
executableCacheConfig=InMemory(50)
compilerCacheConfig=
storageSolution=local
logCompilationCacheAccesses=true
PROPS

echo "Test root ready: ${dest} (use as --root-dir)"
