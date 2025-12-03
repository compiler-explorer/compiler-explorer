# Running on Windows

## Basic Setup

The setup on Windows should be fairly trivial: the only prerequisite is node. If you haven't yet installed node yet, you
can grab it from [here](https://nodejs.org/en/); get the Windows LTS release.

Once you've done this, and added `npm` to the path, run the following commands from any command line, in the directory
you want the Compiler Explorer (from here on, CE) to live:

```bat
git clone https://github.com/compiler-explorer/compiler-explorer.git
```

For quick setup with MSVC compilers, you can use the CE Properties Wizard to automatically configure your compiler:

```bat
cd compiler-explorer
etc\scripts\ce-properties-wizard\run.ps1 <path-to-cl.exe>
```

Alternatively, you can manually create a configuration file which points at your compilers and include directories. Copy
[`docs\WindowsLocal.properties`](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/WindowsLocal.properties)
to a new file, `etc\config\c++.local.properties`, and edit it, following the instructions in the comments.

For a comprehensive explanation of the configuration system, see [Configuration.md](Configuration.md).

## Actually Running the danged thing

Once you've finished setting it up, you can `cd` into the `compiler-explorer` directory, then run

```bat
npm i
npm run dev
```

For debugging, use `npm run debug` instead of `npm run dev`.

For production builds, use `npm start` (which runs webpack and starts the server in production mode).

Eventually, you'll see something that looks like

```
info: =======================================
info:   git release 96451ae8b92e420462137eaaec58f78d3cd6667b
info:   serving static files from 'static'
info:   Listening on http://localhost:10240/
info: =======================================
```

Now point your favourite web browser at http://localhost:10240, and you should be done!

You only have to run `npm i` the first time; every time after that, you should just be able to run `npm run dev` (or `npm start` for production).

## Debugging using VSCode

The easiest way to debug is to add a new terminal in VSCode called `JavaScript Debug Terminal` (via the terminal dropdown menu), then run `npm run dev` from that terminal. This will automatically attach the debugger.

## Setting up binary mode and execution

To create executables with Visual C++, it's required to install the Windows SDK.

You can find the Windows 10 SDK [here](https://developer.microsoft.com/en-US/windows/downloads/windows-10-sdk)

When you've installed the SDK, you'll need to set up the library and include paths in Compilers Explorer. Make sure that
in the previously discussed c++.local.properties you have added at least:

- to includePath
  - Windows Kits/10/include/_version_/ucrt
  - Windows Kits/10/include/_version_/shared
  - Windows Kits/10/include/_version_/um
- to libPath (for the x64 compiler)
  - Windows Kits/10/Lib/_version_/um/x64
  - Windows Kits/10/Lib/_version_/ucrt/x64
  - VC installation path/lib/x64

If needed, you can set in your properties file: `supportsExecute=true`

#### Binary mode

For binary mode, you will need a Windows version of Objdump. There are various versions of MingW available that will
offer binutils including objdump.

The version of objdump that we have tested with is shipped with MingW-64, you can find it for download
[here](https://sourceforge.net/projects/mingw-w64/)

When you use the installer for MingW-64, make sure you select the right architecture during installation.

When you use the zipped version, after unzipping you will need to add the bin folder to your Windows PATHS environment
variable. Be aware that this PATH needs to be added before any other folders that might contain an objdump. You cannot
just point to the .exe as the objdumper without having the proper PATH set, it will not work.

When you have everything installed, you can add to your properties file the following:

```
supportsBinary=true
objdumper=objdump
```

_Note that the 32 bit version of MingW does not support 64 bit binaries._

## Running a Production Build

For a production deployment, you'll want to build a distribution package and run it with more control over node parameters.

**Note:** This setup is intended for local or internal deployments only, not for publicly accessible websites.

First, build the distribution using the provided script:

```bat
etc\scripts\build-dist-win.ps1
```

This creates a ready-to-deploy package in `out/dist/`. You can then run node directly with custom parameters instead of using npm scripts. For example, as done in the [infra repository](https://github.com/compiler-explorer/infra/blob/main/init/run.ps1):

```bat
node.exe --max_old_space_size=6000 -- app.js --dist --port 10240 --language c++
```

Common parameters you might want to configure:
- `--max_old_space_size`: Node memory limit (in MB)
- `--dist`: Run in distribution mode
- `--port`: Server port (default: 10240)
- `--language`: Specify which languages to enable
- `--env`: Environment configuration to load

See `node app.js --help` for all available options.
