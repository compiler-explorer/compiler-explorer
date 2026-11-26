#!/bin/bash
# usage: start.sh ROOT_ETC PORT LOG PREDISCOVERED_JSON|- LANGUAGE...
# Starts a CE instance in the background and waits until it serves its compiler list. Pass - instead of a discovery
# file to let it discover its compilers itself. Fails, with the end of the log, if the instance exits first.
set -euo pipefail

usage='usage: start.sh ROOT_ETC PORT LOG PREDISCOVERED_JSON|- LANGUAGE...'
root=${1:?$usage}
port=${2:?$usage}
log=${3:?$usage}
prediscovered=${4:?$usage}
shift 4
[[ $# -gt 0 ]] || { echo "Give at least one language" >&2; exit 1; }

args=(--root-dir "${root}" --port "${port}" --language "$@" --no-remote-fetch)
[[ "${prediscovered}" == - ]] || args+=(--prediscovered "${prediscovered}")

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
nohup node --no-warnings=ExperimentalWarning --import=tsx app.ts "${args[@]}" > "${log}" 2>&1 &
pid=$!

until [[ $(curl -s -o /dev/null -w '%{http_code}' "localhost:${port}/api/compilers") == 200 ]]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "Instance on port ${port} exited during startup:" >&2
        tail -n 5 "${log}" >&2
        exit 1
    fi
    sleep 1
done
echo "Up on port ${port} (pid ${pid}), logging to ${log}"
