#!/usr/bin/env bash
# Prints "<id> <exe>" for the newest compiler of a language that is installed locally.
# Usage: newest-local-compiler.sh <lang> <regex>
# <regex> matches the whole "<id> <exe>" line and picks the main x86-64 toolchain, e.g. for C++
# '^g[0-9]+ .*/gcc-[0-9.]+/bin/g\+\+$'. Sorting is by install path: ids don't sort by version.
set -euo pipefail

lang=${1:?language, e.g. c++}
regex=${2:?regex matching "<id> <exe>"}

grep -E '^compiler\.[^.]+\.exe=' "/opt/compiler-explorer/cemain/etc/config/${lang}.amazon.properties" |
    sed -E 's/^compiler\.([^.]+)\.exe=/\1 /' |
    while read -r id exe; do
        if [[ -x ${exe} ]]; then
            echo "${id} ${exe}"
        fi
    done |
    grep -E "${regex}" | sort -V -k2 | tail -1
