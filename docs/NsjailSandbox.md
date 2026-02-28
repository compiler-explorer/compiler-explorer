# Using nsjail for Local Compiler Explorer Instances

[nsjail](https://github.com/google/nsjail) is a lightweight Linux process isolation tool that uses namespaces,
cgroups, rlimits, and seccomp-bpf to sandbox processes. Compiler Explorer uses nsjail on its production servers to
isolate both compiler execution and user binary execution. You can enable it on your local instance too, for added
security when running untrusted code.

## Why use nsjail?

Without sandboxing (`sandboxType=none`, the default for local development), compiled user binaries run directly on
your host with the same permissions as the Compiler Explorer process. This is fine when you're the only user, but
risky if you expose your instance to others. nsjail provides:

- **Filesystem isolation** -- user programs only see explicitly mounted paths
- **Resource limits** -- memory, CPU, file size, and process count caps via cgroups
- **Namespace isolation** -- separate PID, mount, UTS, and user namespaces
- **Network isolation** -- no network access for user binaries by default

## Prerequisites

- **Linux only** -- nsjail requires Linux kernel features (namespaces, cgroups, seccomp). It does not work on macOS
  or Windows (even under WSL 1; WSL 2 may work with caveats).
- **Kernel 4.6+** -- some namespace and cgroup features require at least kernel 4.6.
- **cgroup support** -- either cgroups v1 or v2 must be available and configured (see below).
- **Root access** -- needed for one-time cgroup setup; nsjail itself runs unprivileged after that.

## Step 1: Build nsjail

Compiler Explorer maintains its own fork of nsjail
([compiler-explorer/nsjail](https://github.com/compiler-explorer/nsjail)) which adds features used by the included
nsjail configs (e.g. `needs_mount_propagation` for bind mounts). Always use this fork rather than upstream
Google nsjail.

### Install build dependencies (Debian/Ubuntu)

```bash
sudo apt-get install autoconf bison flex gcc g++ git libprotobuf-dev \
  libnl-route-3-dev libtool make pkg-config protobuf-compiler
```

### Build from source

```bash
git clone https://github.com/compiler-explorer/nsjail.git
cd nsjail && git checkout ce && make
```

Then copy the resulting binary somewhere on your PATH:

```bash
sudo cp nsjail /usr/local/bin/nsjail
```

## Step 2: Set up cgroups

nsjail uses cgroups to enforce memory, CPU, and process count limits. Two cgroup hierarchies are needed:

| Cgroup name    | Used for                                      |
|----------------|-----------------------------------------------|
| `ce-compile`   | Compiler and tool execution (more permissive)  |
| `ce-sandbox`   | User binary execution (more restrictive)       |

The setup differs depending on whether your system uses cgroups v1 or v2. You can check with:

```bash
# If this directory exists with controller files, you have cgroups v2:
ls /sys/fs/cgroup/cgroup.controllers

# If you see /sys/fs/cgroup/memory/, /sys/fs/cgroup/pids/, etc., you have cgroups v1:
ls /sys/fs/cgroup/memory/
```

### Cgroups v2 (modern systems: Ubuntu 22.04+, Fedora 31+, etc.)

```bash
# Install cgroup tools if needed:
sudo apt-get install cgroup-tools   # Debian/Ubuntu
# or: sudo dnf install libcgroup-tools  # Fedora

# Create cgroups owned by your user:
sudo cgcreate -a $USER:$USER -g memory,pids,cpu:ce-sandbox
sudo cgcreate -a $USER:$USER -g memory,pids,cpu:ce-compile

# Allow your user to migrate processes into the root cgroup:
sudo chown $USER:root /sys/fs/cgroup/cgroup.procs
```

On **Ubuntu 24.04+** and other distributions with AppArmor restricting unprivileged user namespaces, you also need
to relax those restrictions for nsjail to work:

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_unconfined=0
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

### Cgroups v1 (older systems)

```bash
sudo cgcreate -a $USER:$USER -g memory,pids,cpu,net_cls:ce-sandbox
sudo cgcreate -a $USER:$USER -g memory,pids,cpu,net_cls:ce-compile
```

### Example init script

Here is a minimal script that handles all of the above for cgroups v2. Save it (e.g. as `init-cgroups.sh`) and run
it with `sudo` after each reboot:

```bash
#!/bin/sh

CE_USER=your-username

cgcreate -a ${CE_USER}:${CE_USER} -g memory,pids,cpu:ce-sandbox
cgcreate -a ${CE_USER}:${CE_USER} -g memory,pids,cpu:ce-compile
chown ${CE_USER}:root /sys/fs/cgroup/cgroup.procs

# Needed on Ubuntu 24.04+ / systems with AppArmor user namespace restrictions:
sysctl -w kernel.apparmor_restrict_unprivileged_unconfined=0
sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

### Making cgroup setup persistent

The cgroup directories are lost on reboot. You can either run the script above manually after each boot, or
automate it with a systemd oneshot service. Create `/etc/systemd/system/ce-cgroups.service`:

```ini
[Unit]
Description=Create Compiler Explorer cgroups
After=local-fs.target

[Service]
Type=oneshot
# Replace 'ce' with your username:
ExecStart=/bin/bash -c "cgcreate -a ce:ce -g memory,pids,cpu:ce-sandbox && cgcreate -a ce:ce -g memory,pids,cpu:ce-compile && chown ce:root /sys/fs/cgroup/cgroup.procs && sysctl -w kernel.apparmor_restrict_unprivileged_unconfined=0 && sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ce-cgroups.service
```

## Step 3: Configure Compiler Explorer

By default, sandboxing is disabled for local development (`sandboxType=none` and `executionType=none` in
`etc/config/execution.defaults.properties`). In production, `etc/config/execution.amazon.properties` enables nsjail:

```properties
sandboxType=nsjail
executionType=nsjail
wine=
wineServer=
firejail=
```

To do the same locally, create or edit `etc/config/execution.local.properties`:

```properties
# Enable nsjail for both compiler execution and user binary execution,
# matching the production configuration in execution.amazon.properties:
sandboxType=nsjail
executionType=nsjail

# Path to nsjail binary (default: 'nsjail', i.e. found on PATH):
#nsjail=/usr/local/bin/nsjail
```

The nsjail configuration files are already included in the repository:

| Property                  | Default value                           | Purpose                              |
|---------------------------|-----------------------------------------|--------------------------------------|
| `sandboxType`             | `none`                                  | Sandbox engine for user binaries     |
| `executionType`           | `none`                                  | Sandbox engine for compiler/tools    |
| `nsjail`                  | `nsjail`                                | Path to nsjail binary                |
| `nsjail.config.sandbox`   | `etc/nsjail/user-execution.cfg`         | Config for user binary sandboxing    |
| `nsjail.config.execute`   | `etc/nsjail/compilers-and-tools.cfg`    | Config for compiler sandboxing       |

## Step 4: Customising the nsjail configs (optional)

The included nsjail configs (`etc/nsjail/user-execution.cfg` and `etc/nsjail/compilers-and-tools.cfg`) are written
for the CE production environment but should work as-is on most local setups. Mounts for paths that don't exist on
your system (CEFS, NVIDIA devices, Intel/ARM/QNX compilers, etc.) are silently skipped by nsjail, so there is no
need to comment them out.

The main reason you might need to edit the configs is if your compilers are installed outside
`/opt/compiler-explorer`. In that case, add a bind mount for the relevant path:

```protobuf
mount {
    src: "/home/youruser/compilers"
    dst: "/home/youruser/compilers"
    is_bind: true
}
```

## Step 5: Verify it works

Start Compiler Explorer:

```bash
make dev
```

Then try compiling and executing a simple program. If everything is working, you should see normal output. If nsjail
fails, you'll typically see an error like:

```
Launching child process failed
```

### Troubleshooting

**"runChild():486 Launching child process failed"**

This usually means the cgroups aren't set up correctly. Verify:

```bash
# Cgroups v2:
ls -la /sys/fs/cgroup/ce-sandbox/
ls -la /sys/fs/cgroup/ce-compile/

# Cgroups v1:
ls -la /sys/fs/cgroup/memory/ce-sandbox/
ls -la /sys/fs/cgroup/pids/ce-sandbox/
```

The directories should exist and be owned by your user.

**"No such file or directory" for a mount source**

A non-optional mount source doesn't exist on your system. Either install the missing package, create the path, or
comment out the mount in the nsjail config.

**User namespaces not enabled**

Some distributions disable unprivileged user namespaces by default. Check:

```bash
sysctl kernel.unprivileged_userns_clone
```

If it returns `0`, enable it:

```bash
sudo sysctl -w kernel.unprivileged_userns_clone=1
# To make permanent:
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/99-userns.conf
```

**Permission denied errors**

nsjail needs to be able to create namespaces. If you're running inside a container (e.g. Docker), you may need
`--privileged` or specific capabilities (`CAP_SYS_ADMIN`, `CAP_SYS_PTRACE`).

## Understanding the two sandbox configs

Compiler Explorer uses two separate nsjail configurations with different security profiles:

### Compiler sandbox (`compilers-and-tools.cfg`)

Used when running compilers themselves. More permissive because compilers need significant resources:

| Resource          | Limit            |
|-------------------|------------------|
| Memory            | 1.25 GiB         |
| Max processes     | 72               |
| CPU               | 100% of one core |
| Max file size     | 1 GiB            |
| Open files        | 300              |
| Filesystem access | `/bin`, `/lib`, `/usr`, `/opt/compiler-explorer` (read-only) |

### User execution sandbox (`user-execution.cfg`)

Used when running user-compiled binaries. Much more restrictive:

| Resource          | Limit            |
|-------------------|------------------|
| Memory            | 200 MiB          |
| Max processes     | 14               |
| CPU               | 50% of one core  |
| Max file size     | 16 MiB           |
| Open files        | 100              |
| Filesystem access | `/lib`, `/usr/lib` only (no `/bin`, no `/usr/bin`) |
| `/tmp`            | 20 MiB tmpfs, noexec |

## Alternative: running without nsjail

If you can't use nsjail (e.g. on macOS or in a restricted environment), the default `sandboxType=none` works fine
for local development. For some extra safety without nsjail, you can restrict dangerous compiler flags by setting
`optionsForbiddenRe` in `etc/config/compiler-explorer.local.properties`:

```properties
optionsForbiddenRe=^(-W[alp],)?((--?(wrapper|fplugin.*|specs|load|plugin|include|fmodule-mapper)|(@.*)|-I|-i)(=.*)?|--)$
```

This blocks flags like `--plugin`, `-fplugin`, and `--wrapper` that could be used to execute arbitrary code via the
compiler.

## Further reading

- [nsjail GitHub repository](https://github.com/google/nsjail)
- [nsjail documentation](https://nsjail.dev)
- [Compiler Explorer nsjail fork](https://github.com/compiler-explorer/nsjail)
- [Compiler Explorer infra repository](https://github.com/compiler-explorer/infra) -- infrastructure, deployment, and compiler installation tooling for the live site
- [Configuration.md](Configuration.md) -- general CE configuration reference
