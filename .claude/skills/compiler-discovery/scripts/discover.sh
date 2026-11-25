#!/bin/bash
# usage: discover.sh [--exit-on-compiler-failure] ROOT_ETC OUTPUT_JSON LANGUAGE...
# Runs discovery-only against ROOT_ETC for the given languages and writes the discovered compilers to OUTPUT_JSON.
# With --exit-on-compiler-failure, as the prod discovery workflow runs it, any compiler that fails to load fails the
# run instead of being left out of the file.
set -euo pipefail

usage='usage: discover.sh [--exit-on-compiler-failure] ROOT_ETC OUTPUT_JSON LANGUAGE...'
extra=()
if [[ "${1:-}" == --exit-on-compiler-failure ]]; then
    extra+=(--exit-on-compiler-failure)
    shift
fi
root=${1:?$usage}
out=${2:?$usage}
shift 2
[[ $# -gt 0 ]] || { echo "Give at least one language" >&2; exit 1; }

log="${out%.json}.log"
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
if ! node --no-warnings=ExperimentalWarning --import=tsx app.ts --root-dir "${root}" --language "$@" --no-remote-fetch \
    --discovery-only "${out}" "${extra[@]}" > "${log}" 2>&1; then
    echo "Discovery failed (full log: ${log}):" >&2
    grep -E '^error: Failed to create' "${log}" >&2 || tail -n 10 "${log}" >&2
    grep -E '^(warn|error):.*compiler binary' "${log}" | sed -E 's/ \{".*$//' | head -n 20 >&2 || true
    exit 1
fi

python3 -I -c 'import json, sys; n = len(json.load(open(sys.argv[1]))); print(f"{n} compilers in {sys.argv[1]}"); sys.exit(n == 0)' "${out}"
