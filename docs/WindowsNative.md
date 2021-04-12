# Running on Windows

Contact: [Nicole Mazzuca](https://github.com/ubsan)

## Basic Setup

The setup on Windows should be fairly trivial:
the only prerequisite is node.
If you haven't yet installed node yet, you can grab it from
[here](https://nodejs.org/en/);
get the Windows LTS release.

Once you've done this,
and added `npm` to the path,
run the following commands from any command line,
in the directory you want the Compiler Explorer (from here on, CE)
to live:

```bat
git clone https://github.com/compiler-explorer/compiler-explorer.git
```

Then, we'll need to make a configuration file
which points at your compilers and include directories.
Copy `docs\WindowsLocal.properties` to a new file,
`etc\config\c++.local.properties`, and edit it,
following the instructions in the comments.
If you have any questions, please ping me on discord.


## Actually Running the danged thing

Once you've finished setting it up,
you can `cd` into the `compiler-explorer` directory,
then run

```bat
npm install
npm install webpack -g
npm install webpack-cli -g
npm update webpack
npm start
```

Eventually, you'll see something that looks like

```
info: =======================================
info:   git release 96451ae8b92e420462137eaaec58f78d3cd6667b
info:   serving static files from 'static'
info:   Listening on http://localhost:10240/
info: =======================================
```

Now point your favorite web browser at http://localhost:10240
and you should be done!

You only have to run `npm install` the first time;
every time after that, you should just be able to run `npm start`.

## Debugging using VSCode
Similar to [WindowsSubsystemForLinux](WindowsSubsystemForLinux.md), the following is a `launch.json` that works for attaching to an instance of CE that was launched with `npm run-script debugger` (launches with the `--inspect` flag). 


```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "node",
            "request": "attach",
            "name": "Attach to Process",
            "port": 9229,
            "address": "localhost",
            "protocol": "inspector",
            "localRoot": "${workspaceRoot}",
            "remoteRoot": "C:\\Users\\${username}\\compiler-explorer"
        }
    ]
}
```

Launch CE with `npm run-script debugger` to have node listen on port 9229. 

Because this only attaches to the process, as opposed to launching the process, in order to debug startup code you need to attach while npm is starting up.  The `debugger` script also enables `debug` logging level so debug print statements can be seen during the CE startup and run.

### Setting up binary mode and execution

To create executables with Visual C++, it's required to install the Windows SDK.

You can find the Windows 10 SDK [here](https://developer.microsoft.com/en-US/windows/downloads/windows-10-sdk)

When you've installed the SDK, you'll need to setup the library and include paths in Compilers Explorer.
Make sure that in the previously discussed c++.local.properties you have added at least:
 * to includePath
   - Windows Kits/10/include/*version*/ucrt
   - Windows Kits/10/include/*version*/shared
   - Windows Kits/10/include/*version*/um
 * to libPath (for the x64 compiler)
   - Windows Kits/10/Lib/*version*/um/x64
   - Windows Kits/10/Lib/*version*/ucrt/x64
   - VC installation path/lib/x64

If needed, you can set in your properties file: ```supportsExecute=true```

#### Binary mode

For binary mode, you will need a Windows version of Objdump. There are various
versions of MingW available that will offer binutils including objdump.

The version of objdump that we have tested with is shipped with MingW-64,
you can find it for download [here](https://sourceforge.net/projects/mingw-w64/)

When you use the installer for MingW-64, make sure you select the right architecture during installation.

When you use the zipped version, after unzipping you will need to add the bin folder to your Windows PATHS environment variable. Be aware that this PATH needs to be added before any other folders that might contain an objdump. You cannot just point to the .exe as the objdumper without having the proper PATH set, it will not work.

When you have everything installed, you can add to your properties file the following:
```
supportsBinary=true
objdumper=objdump
```

*Note that the 32 bit version of MingW does not support 64 bit binaries.*
