# Using gVisor for Sandboxing in Compiler Explorer

Compiler Explorer supports using [gVisor](https://gvisor.dev/) (`runsc`) as a sandboxing runtime. gVisor provides
a user-space kernel (called the "Sentry") that intercepts and implements system calls, providing a stronger
security boundary than traditional namespace-based sandboxes like `nsjail` or `firejail` which directly expose
the host kernel.

This is particularly useful if you are running Compiler Explorer for untrusted users and want defense-in-depth against kernel exploits.

## Why use gVisor?

Compared to the default `nsjail` setup:
*   **Stronger Isolation:** gVisor does not directly expose the host kernel to the sandboxed process.
    Syscalls are handled by the Sentry in Go, reducing the attack surface.
*   **Easier Setup (No Cgroups Configuration):** Unlike `nsjail`, which requires root access to set up cgroups
    on the host for resource limiting, gVisor can run fully rootless without any host cgroup configuration
    (resource limits are enforced internally by the Sentry).
*   **OCI Compliant:** gVisor uses standard OCI bundles, making it compatible with modern container infrastructure.

## Prerequisites

*   **Linux Host:** gVisor requires a Linux host.
*   **User Namespaces Enabled:** The host must have unprivileged user namespaces enabled (usually default on modern distributions).
*   **`runsc` Binary:** You must download and install the `runsc` binary.

## Step 1: Install `runsc`

Follow the official [gVisor installation guide](https://gvisor.dev/docs/user_guide/install/) to download the `runsc` binary.

A quick way to install the latest release:

```bash
(
  set -e
  ARCH=$(uname -m)
  URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
  curl -LO "${URL}/runsc"
  chmod a+rx runsc
  sudo mv runsc /usr/local/bin/runsc
)
```

Ensure it is readable and executable by the user running Compiler Explorer.

## Step 2: Configure Compiler Explorer

To enable gVisor sandboxing, create or edit `etc/config/execution.local.properties`:

```properties
# Enable gvisor for both user execution and compiler execution
sandboxType=gvisor
executionType=gvisor

# Optional: Path to runsc binary if not in PATH
#runsc=/usr/local/bin/runsc

# Config templates (already included in the repo)
gvisor.config.sandbox=etc/gvisor/user-execution.json
gvisor.config.execute=etc/gvisor/compilers-and-tools.json
```

## How it works

Compiler Explorer's gVisor integration works by:
1.  **OCI Bundle Creation:** For every execution, CE creates a temporary OCI bundle directory containing a
    `config.json` (generated from the templates) and an empty `rootfs` directory.
2.  **Rootless Execution:** CE runs `runsc --rootless --network=none run --bundle <bundle-dir> <container-id>`.
    *   `--rootless` maps the internal root user (`uid: 0`) to the host user running Compiler Explorer.
    *   `--network=none` disables network access inside the sandbox.
3.  **Mounts:** The default templates mount system directories (`/bin`, `/lib`, `/usr`, etc.) as read-only.
4.  **CWD Mount:** The compiler's working directory (where input/output files live) is dynamically bind-mounted
    as writable (`rw`) to `/app` (or the same host path) inside the sandbox, allowing the compiler
    to write its output.
5.  **Cleanup:** Once execution completes, the temporary bundle directory is deleted and the container
    is forced deleted (`runsc delete --force`).

## Customizing Configuration

You can customize the mounts and resource limits by editing `etc/gvisor/user-execution.json` and
`etc/gvisor/compilers-and-tools.json`.

These files are standard OCI runtime specs. You can configure:
*   **Memory limits:** `linux.resources.memory.limit` (in bytes).
*   **Pid limits:** `linux.resources.pids.limit`.
*   **File size limits:** `process.rlimits` (type `RLIMIT_FSIZE`).
*   **Mounts:** Add entries to the `mounts` array. You can use the custom `"skip_if_missing": true` property
    on bind mounts to ignore them if the source path does not exist on the host.
