# Running EWARM compiler on windows host

This document will show a little insight on how to get the IAR ARM compiler working with compiler explorer _(Some line
highlighting is broken as EWAVR is different from EWARM)_

# Prerequisites

To run the IAR ARM compiler you will need:

- A valid installation of the IAR ARM compiler. [EWARM](https://www.iar.com/iar-embedded-workbench/#!?architecture=Arm)
  has a free 30-day trial on their website
- Technically you need a license as well to run the compiler, however you can get by with the 30-day free trial, as you
  will be able to run the compiler for 30 days for free
- MinGW C++ toolchain for `c++filt` and `objdump`

## IAR ARM Compiler

This compiler will be installed along with the IAR Embedded Workbench (EWARM). Default installation location is under
`C:\Program Files (x86)\IAR Systems\Embedded Workbench 8.2\arm\bin\iccarm.exe`

## MinGW C++ Toolchain

We need to install this toolchain through [MSYS2](https://www.msys2.org/), as it contains `c++filt` and `objdump`, which
are needed to demangle the assembly

- First download [MSYS2](https://www.msys2.org/)
- Install it and run it, entering the first command `pacman -Syuu`, this command will update all the internal MSYS2
  modules to their latest version. This command will also update all installed modules, such as the toolchain if it was
  installed.
- When you run the command for the first time, it will exit the bash console. You have to open it again and run the same
  command again `pacman -Syuu`
- After the second time, everything will be up to date. To install the MinGW toolchain, run
  `pacman -S mingw-w64-x86_64-toolchain`. This will install the toolchain to your MSYS2 default installation path under
  `C:\msys64\mingw64\bin`
- Add this path `C:\msys64\mingw64\bin` to windows global `PATH`
- Test by running `c++filt --help` from windows command prompt, if everything is set up correctly, then you should see
  all the command line options for c++filt

# Setup and Configuration

## Running compiler explorer on Windows

Refer to the [readme](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/WindowsNative.md) on running
Native on Windows for general setup of Compiler Explorer and other compilers.

## Setting up c++.local.properties

The next step is to create a `c++.local.properties` file under `etc/config` folder. The next step is going to be
different for everyone, as you can choose what compiler options you pass to the compiler and so on, but im going to
paste my template here, and you can just modify, what you need

```
# Default settings for C++
compilers=iar8.32.4

compiler.iar8.32.4.exe=C:\arm\bin\iccarm.exe
compiler.iar8.32.4.name=IAR8.32.4
compiler.iar8.32.4.supportsDemangler=true
compiler.iar8.32.4.supportsBinary=false
compiler.iar8.32.4.supportsExecute=false

compiler.iar8.32.4.options=--enable_restrict -IC:\arm\inc -IC:\arm\inc\c -IC:\arm\inc\cpp --dlib_config C:\arm\inc\c\DLib_Config_Full.h --c++ -e --no_exceptions --no_rtti --no_static_destruction --cpu Cortex-M4 --fpu VFPv4_sp --endian little --cpu_mode thumb
compiler.iar8.32.4.compilerType=ewarm

compiler.iar8.32.4.versionRe=IAR ANSI C\/C\+\+ Compiler.*ARM

defaultCompiler=iar8.32.4

demangler=c++filt
objdumper=objdump
demanglerType=default
postProcess=
binaryHideFuncRe=^(__.*|_(init|start|fini)|(de)?register_tm_clones|call_gmon_start|frame_dummy|\.plt.*|_dl_relocate_static_pie)$
needsMulti=false
stubRe=\bmain\b
stubText=int main(void){return 0;/*stub provided by Compiler Explorer*/}
```

**It's important to note that the `compiler.iar8.32.4.compilerType` field is set to `ewarm` this will be the custom
compiler key later on**

## Running Compiler Explorer

You should be able to just `cd` into the compiler explorer repository and run `npm start`. After that just head on to
[localhost:10240](http://localhost:10240)
