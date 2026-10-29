---
name: sync-windows-lib-props
description: Sync C++ library entries from libraries.yaml into etc/config/c++.amazonwin.properties (the Windows/MSVC library list), open the PR, and spot-check Windows conan builds. Use when asked for a "windows properties sync" or to bring Windows libraries up to date.
---

# Sync Windows C++ library properties

MSVC compilers read libraries from `etc/config/c++.amazonwin.properties`. Nothing updates it automatically, so it
drifts behind the Linux libraries. Infra's generator copies them across as `packagedheaders=true` entries.

1. Make sure cemain and `/opt/compiler-explorer/infra` are on an up-to-date `main`, then branch cemain as
   `windows-props-sync-<YYYY-MM-DD>`.
2. From the infra checkout, run this with `F=<cemain>/etc/config/c++.amazonwin.properties`:
   `bin/ce_install cpp-library generate-windows-props --input-file $F --output-file $F`
3. Review `git diff` and tell me what changed: new libraries and new versions, in a table. Point out anything odd,
   such as static libraries (`staticliblink`) that need a real MSVC build, or GPU/platform libraries copied over
   blindly. New libraries are appended after the last entry, not in alphabetical order. That's fine, because there's
   no `libs=` list.
4. Commit and open the PR after I say so (`git commit -n`, push the branch to origin).
5. Spot-check a few new entries with Windows conan builds, not all of them: a new header-only library, a new static
   library, and a new version of an existing static library. Use the process-lib-wizard-prs skill's
   `scripts/dispatch-lib-build.sh win <name> vcpp_v19_latest_x64` one at a time, and go by its exit code, because the
   run status lies. The builder reads infra's `libraries.yaml`, so builds don't wait for the PR to merge.
   A build tries every version of the library, and older versions often fail on Windows, so the "N packages built
   OK" line doesn't say which version built. Check the new version with
   `curl -s https://conan.compiler-explorer.com/failedbuilds/<lib>/<version>`: no `vcpp_v19_latest_x64` record means
   it built. The run log doesn't show why a build failed, but
   `.../getlogging_forcommit/<lib>/<ver>/<ver>/vcpp_v19_latest_x64/x86_64/libstdc%2B%2B` returns the full build log. `No binaries found to export` means the MSVC library isn't named
   `<staticliblink>.lib`; fix it with an `after_stage_script_pwsh` line in `libraries.yaml` that patches the
   library's CMake (see xad). The builder skips builds that failed before, so clear the records before a rebuild:
   `bin/ce_install build-status clear-for-library <lib> --version <ver>` (from infra).
6. Report the build results and list the remaining new entries that still need a build.
