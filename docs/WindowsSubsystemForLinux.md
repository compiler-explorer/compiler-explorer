# Running on Windows Subsystem for Linux

Contact: [@AndrewPardoe](https://github.com/AndrewPardoe)

The Compiler Explorer ("CE" from here on) runs quite well on the
[Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/faq) ("WSL"). Running on WSL enables
Linux-based compilers to continue running natively while enabling Windows-based compilers to run in a real Windows
environment.

No special configuration is needed to run CE under WSL. Some configuration is required for hosting the Microsoft Visual
C++ ("MSVC") compiler. Testing has mainly been done on the Ubuntu distro but any distro should work.

## WSL/Windows interop and its limitations

WSL offers rich interop with Windows processes. You can run any Windows executable, such as "`cl.exe`", from a bash
shell. But this interop capability has some limitations.

- Windows volumes: While Windows executables can be run from bash, they cannot see Linux volumes. Windows executables
  that need to read or write files must be run on a Windows volume. This means all MSVC compiles must be done in the
  Windows `%TEMP%` directory instead of in the `bash` environment's temp directory.
- Path: The WSL path set in bash prepends the Windows path. While Linux filesystems support odd naming conventions such
  as spaces and parentheses, Windows' path uses these as a matter of course (e.g., `c:\Program Files (x86)`).
  Additionally, the Windows path delimiter is `\` instead of `/`, and it uses drive letters instead of mount points that
  are separated with a colon.
- Path names: A Windows path of `c:\tmp` is normally referred to as `/mnt/c/tmp` in `bash`. However, users can customize
  their `drvfs` mount points. A tool is provided in newer Windows releases, `/bin/wslpath`, that will convert paths
  between systems. Code in CE currently does the conversion between the standard conventions using string manipulation.
- Environment variables: While the Windows path is available in bash, Windows environment variables are not. CE uses
  `cmd.exe /c echo %TEMP%` to determine the Windows temporary directory.
- Execution environment: The execution environment cannot currently be set when doing `childprocess.spawn`. This is a
  serious issue for the MSVC compiler, which is highly environment-dependent (e.g., `%INCLUDE%`, `%LIBPATH%`, etc.)

## Configuration

This section is intended for the many WSL users who are new to Linux.

If you plan on debugging CE, you should clone the CE repo on a Windows volume.

CE is built on node.js ("node"). The easiest way to install node is using NVM, the Node Version Manager. Run the
following commands from a bash shell:

- `apt-get update` to make sure apt is up-to-date
- `apt-get install build-essential libssl-dev`, though you probably have these already
- Check https://github.com/creationix/nvm/releases for the latest NVM release, substituting it in the next command.
- `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash` to install NVM
- `source ~/.profile` to reload your profile, bringing NVM into your environment
- `nvm ls-remote --lts` to show the latest long-term supported (LTS) version of node.js
- `nvm install 10.15.3`, substituting the latest LTS version, to install node.js

At this point you can change into the directory where you cloned CE and `make`. `make` will install a bunch of node
packages and will finish with a message similar to this:

```
info: =======================================
info: Listening on http://localhost:10240/
info:   serving static files from 'static'
info:   git release bbf1407109d0439199f71bfdf4037fdeb0eb8393
info: =======================================
```

Now you can point your favorite web browser at http://localhost:10240 and see your own personal CE in action!

## Code changes

CE only required a few changes in order to run properly under WSL. Those changes are listed here:

- `app.ts`:
  - `process.env.wsl` is set if CE if the string "Microsoft" in found in the output of `uname -a`. This works for all
    WSL distros as they all run on the base Microsoft Linux kernel.
  - If the `-tmpDir` option is specified on the command line, os.tmpdir()'s return value is set to the specified value.
    Note that if this is specified as a non-Windows volume, Windows executables will fail to run properly. Otherwise,
    os.tmpdir() is set to the value of the Windows `%TEMP%` directory if CE can get the temp path from invoking
    `cmd.exe` from WSL.
- `lib/exec.ts`: Execute the compiler in the temporary directory.
- `lib/compilers/wsl-vc.ts`: See also `wine-vc.ts`, the Wine version of this compiler-specific file. These files provide
  custom behaviors for a compiler. This file does two interesting things:
  - The `CompileCl` function translates from Linux-style directories to Windows-style directories (`/mnt/c/tmp` to
    `c:/tmp`) so that `CL.exe` can find its input files.
- `etc/config/c++.defaults.properties`: Add a configuration (`&cl19`) for MSVC compilers. This edits in here are
  currently wrong in two ways, but it doesn't affect the main CE instance as it uses `amazon` properties files, and it
  doesn't affect anyone running a local copy of CE because CE will just fail silently when it can't find a compiler.
  - The locations of these are hardcoded to a particular install location. See **MSVC setup** below for more
    information.
  - Setting of the `%INCLUDE%` path is done with the `/I` switch. This is very clunky and will fall over when
    command-line limits are hit, but it's the only option currently as environments aren't passed through when starting
    a Windows process from WSL.

## Debugging

The only viable option for debugging under WSL is to use [VS Code](https://code.visualstudio.com). VSCode's 'Auto
Attach' option works on wsl and is the easiest way to start debugging. Make sure 'Auto Attach' is on (it is by default),
then at the VSCode terminal start an instance any way you prefer: `make` or `npm start` or similar. (`make` is required
at least for the first run).
