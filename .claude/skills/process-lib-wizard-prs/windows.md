# Windows (C++ only)

## Review (step 2)

MSVC compilers read libraries from `etc/config/c++.amazonwin.properties`, not from the Linux file, and the wizard
doesn't touch it. Windows servers have no `/opt/compiler-explorer/libs`, so every entry there is
`packagedheaders=true` with no `.path=`: CE downloads the library's conan header package and adds
`-I<dir>/<id>/include`. Don't add Windows entries in the wizard PR: the proper way is infra's
`bin/ce_install cpp-library generate-windows-props --input-file <cemain>/etc/config/c++.amazonwin.properties
--output-file <same>`, which copies the Linux libraries into the Windows file. It reads the Linux libraries from
cemain `main`, so run it in its own PR after the wizard PRs merge, and review the whole diff: it also picks up
every other library that drifted since the last sync. Hand-write a block only when I say a library is a special
case. Either way, new Windows entries need a Windows build after merge (below).

## Build (step 6)

Only if the library was added to `c++.amazonwin.properties`. This is needed even for header-only libraries, because
Windows serves them as conan packages. Run `scripts/dispatch-lib-build.sh win <name> <compiler-id>`, with the bare
library name (no `libraries/c++/` prefix). For header-only libraries the builder always uses
`mingw64_ucrt_gcc_1220`, whatever compiler you pass. **The run's status means nothing**: it reports success when
the build failed, because the script drops ce_install's exit status. Go by the script's exit code, which checks
the log for `N packages built OK`. Header-only packaging runs the library's own CMake install; when that installs no
headers (no `CMakeLists.txt`, or no install rules), the builder copies the repo's `include/`, or else its root, into
the package instead.
