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
- `curl https://raw.githubusercontent.com/creationix/nvm/v0.33.8/install.sh | bash` to install NVM
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

- `app.js`:
  - `process.env.wsl` is set if CE if the string "Microsoft" in found in the output of `uname -a`. This works for all
    WSL distros as they all run on the base Microsoft Linux kernel.
  - If the `-tmpDir` option is specified on the command line, both `process.env.tmpDir` and `process.env.winTmp` are set
    to the specified value Note that if this is specified as a non-Windows volume, Windows executables will fail to run
    properly. Otherwise, `process.env.winTmp` is set to the value of the Windows `%TEMP%` directory.
- `lib/exec.js`: Execute the compiler in the temporary directory. If the compiler's binary is located on a mounted
  volume (`startsWith("/mnt"`)) and CE is running under WSL, run the compiler in the `winTmp` directory. Otherwise, use
  the Linux temp directory.
- `lib/compilers/wsl-vc.js`: See also `wine-vc.js`, the Wine version of this compiler-specific file. These files provide
  custom behaviors for a compiler. This file does two interesting things:
  - The `CompileCl` function translates from Linux-style directories to Windows-style directories (`/mnt/c/tmp` to
    `c:/tmp`) so that `CL.exe` can find its input files.
  - The `newTempDir` function creates a temporary directory in `winTmp`. CEs creates directories under the temp
    directory that start with `compiler-explorer-compiler` where the compiler and compiler output lives. This is similar
    to the function in `lib/base-compiler.js`.
- `etc/config/c++.defaults.properties`: Add a configuration (`&cl19`) for MSVC compilers. This edits in here are
  currently wrong in two ways, but it doesn't affect the main CE instance as it uses `amazon` properties files, and it
  doesn't affect anyone running a local copy of CE because CE will just fail silently when it can't find a compiler.
  - The locations of these are hardcoded to a particular install location. See **MSVC setup** below for more
    information.
  - Setting of the `%INCLUDE%` path is done with the `/I` switch. This is very clunky and will fall over when
    command-line limits are hit, but it's the only option currently as environments aren't passed through when starting
    a Windows process from WSL.

## Debugging

The only viable option for debugging under WSL is to use [VS Code](https://code.visualstudio.com). Because VS Code
doesn't currently run natively under WSL, you have to attach to a running CE instance. The following is a `launch.json`
that works for attaching to an instance of CE that was launched with the `--inspect` flag.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "attach",
      "name": "Attach to Process",
      "port": "9229",
      "address": "localhost",
      "protocol": "inspector",
      "localRoot": "${workspaceRoot}",
      "remoteRoot": "/mnt/c/src/compiler-explorer"
    }
  ]
}
```

Launch CE with `make NODE_ARGS="--inspect"` to have node listen on port 9229.

Because you can only attach to the process, as opposed to launching the process, you're limited to `printf` debugging
for startup code. Search the code for `logger.info` to see examples of how to `printf` debug.

## MSVC setup

TODO. There's no real MSVC setup at this point because there's no good way to pass the environment to an invocation of
`CL.exe`. Just point the `properties` file at your compiler binary and hack on the `/I` options until something works.

When I get this working in a generalized fashion, CE's config will expect that MSVC drops match the format used by the
daily NuGet compiler drops at https://visualcpp.myget/org. (NuGet packages are just renamed ZIP files plus metadata so
they make an easy distribution method for compiler toolset drops.)

## Putting it all together

This should be enough information to get you started running CE under WSL. If there's information that you wish you
would have had, please submit a PR to document. If there's information you're lacking to get running, please enter an
Issue on the CE repo or contact me directly.
